"""Who speaks when, and how: pyannote diarization + gender / age / emotion per line.

Lines are rebuilt from Scribe's word timings, but the speaker of every word comes from pyannote
(Scribe merges same-gender actors), so a line is split wherever the real voice changes.

Models:
  pyannote/speaker-diarization-3.1                            (gated: HF_TOKEN + accepted terms)
  prithivMLmods/Common-Voice-Gender-Detection                 gender
  audeering/wav2vec2-large-robust-24-ft-age-gender            age in years
  ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition   emotion (acted-speech model: a hint, not truth)
"""
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification, Wav2Vec2Model, Wav2Vec2PreTrainedModel

from .common import ANALYSIS_RATE as SR

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
GENDER_MODEL = "prithivMLmods/Common-Voice-Gender-Detection"
AGE_MODEL = "audeering/wav2vec2-large-robust-24-ft-age-gender"
EMOTION_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

MAX_GAP = 0.8  # seconds of silence that starts a new line
MAX_LEN = 8.0  # split a line at the next sentence end after this many seconds
MIN_FEATURE_SEC = 0.6  # shorter lines get no gender/age/emotion
MAX_FEATURE_SEC = 10.0  # classifiers see at most this much of a line
BLIP_SEC = 0.5  # a speaker change shorter than this between two runs of the same voice is diarization noise


def diarize(audio, device, num_speakers=None):
    from pyannote.audio import Pipeline

    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN missing in .env (and accept terms at hf.co/pyannote/speaker-diarization-3.1 "
                         "and hf.co/pyannote/segmentation-3.0)")
    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=token).to(torch.device(device))
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    annotation = pipeline({"waveform": torch.from_numpy(audio)[None], "sample_rate": SR}, **kwargs)
    del pipeline
    return [{"speaker": f"V{int(spk.rsplit('_', 1)[-1]):02d}", "start": round(t.start, 3), "end": round(t.end, 3)}
            for t, _, spk in annotation.itertracks(yield_label=True)]


def import_turns(data, start=0.0, end=None):
    """Turns from an earlier run: a diarization list or analyze_speakers.py output ({"segments": [...]}),
    shifted/trimmed to a --start/--end window. Only who-when is kept; features are recomputed per line."""
    turns = []
    for t in data["segments"] if isinstance(data, dict) else data:
        a, b = t["start"] - start, t["end"] - start
        if b <= 0 or (end is not None and a >= end - start):
            continue
        spk = t["speaker"]
        spk = spk if spk.startswith("V") else f"V{int(spk.rsplit('_', 1)[-1]):02d}"
        turns.append({"speaker": spk, "start": round(max(a, 0.0), 3), "end": round(b, 3)})
    return turns


def word_speakers(words, turns):
    """pyannote voice for every word: the turn overlapping it most, else the nearest one."""
    starts = np.array([t["start"] for t in turns])
    ends = np.array([t["end"] for t in turns])
    labels = [t["speaker"] for t in turns]
    out = []
    for w in words:
        overlap = np.minimum(ends, w["end"]) - np.maximum(starts, w["start"])
        if overlap.max() > 0:
            out.append(labels[int(overlap.argmax())])
        else:
            dist = np.maximum(starts - w["end"], w["start"] - ends)
            out.append(labels[int(dist.argmin())])
    # Smooth blips: a short run between two runs of the same voice joins them.
    runs = []  # [speaker, first index, last index]
    for i, spk in enumerate(out):
        if runs and runs[-1][0] == spk:
            runs[-1][2] = i
        else:
            runs.append([spk, i, i])
    for k in range(1, len(runs) - 1):
        spk, a, b = runs[k]
        if runs[k - 1][0] == runs[k + 1][0] != spk and words[b]["end"] - words[a]["start"] < BLIP_SEC:
            out[a:b + 1] = [runs[k - 1][0]] * (b - a + 1)
    return out


def build_lines(scribe_words, turns):
    """Scribe words -> dub lines, split on voice change, pauses and long sentences.

    A voice change is either pyannote's or Scribe's: each one sometimes merges two actors the other keeps
    apart, and the per-line gender/age check tells the script model who is who.
    """
    words = [w for w in scribe_words if w["type"] in ("word", "audio_event") and w.get("start") is not None]
    speakers = word_speakers(words, turns) if turns else [w.get("speaker_id") or "V00" for w in words]
    lines, cur, prev_end, scribe_spk = [], None, None, None
    for w, spk in zip(words, speakers):
        is_word = w["type"] == "word"
        text = w["text"] if is_word else f"[{w['text'].strip('()[] ')}]"
        changed = is_word and cur is not None and cur["voice"] is not None and (
            spk != cur["voice"] or w.get("speaker_id") != scribe_spk)
        new = (cur is None or changed or w["start"] - prev_end > MAX_GAP
               or (cur["end"] - cur["start"] > MAX_LEN and cur["en"].rstrip()[-1:] in ".?!"))
        if is_word:
            scribe_spk = w.get("speaker_id")
        if new:
            cur = {"start": w["start"], "end": w["end"], "voice": None, "en": "", "speech_start": None}
            lines.append(cur)
        cur["en"] += (" " if cur["en"] else "") + text.strip()
        cur["end"] = max(cur["end"], w["end"])
        prev_end = w["end"]
        if is_word:
            cur["voice"] = cur["voice"] or spk
            if cur["speech_start"] is None:
                cur["speech_start"] = w["start"]
    # Time each line from its first spoken word, so a leading [sound] doesn't start the dub early.
    for i, line in enumerate(lines):
        line["id"] = i
        speech_start = line.pop("speech_start")
        if speech_start is not None:
            line["start"] = speech_start
        line["voice"] = line["voice"] or "SFX"
    return lines


# --- audeering age head (custom architecture from its model card) ---
class _Head(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, x):
        return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))


class _AgeGender(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = _Head(config, 1)
        self.gender = _Head(config, 3)
        self.init_weights()

    def forward(self, input_values):
        return self.age(self.wav2vec2(input_values)[0].mean(dim=1))


class _EmotionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.output = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        return self.output(self.dropout(torch.tanh(self.dense(self.dropout(x)))))


class _Emotion(Wav2Vec2PreTrainedModel):
    """ehcalabres checkpoint: mean-pooled wav2vec2 + classifier.dense/classifier.output (not the stock HF head)."""

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = _EmotionHead(config)
        self.init_weights()

    def forward(self, input_values, attention_mask=None):
        return self.classifier(self.wav2vec2(input_values, attention_mask=attention_mask)[0].mean(dim=1))


class _Classifier:
    def __init__(self, name, device, model_class=Wav2Vec2ForSequenceClassification):
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        self.model = model_class.from_pretrained(name).to(device).eval()
        self.labels = self.model.config.id2label
        self.device = device

    @torch.inference_mode()
    def probs(self, wav):
        x = self.fe(wav, sampling_rate=SR, return_tensors="pt").to(self.device)
        out = self.model(**x)
        return torch.softmax(getattr(out, "logits", out), dim=-1)[0].cpu().numpy()


def age_group(years):
    return "child" if years < 13 else "teen" if years < 20 else "young_adult" if years < 35 \
        else "adult" if years < 55 else "senior"


def gender_word(p_female):
    return "female" if p_female >= 0.7 else "male" if p_female <= 0.3 else "uncertain"


@torch.inference_mode()
def add_features(lines, audio, device):
    """p_female, age (years) and emotion for every line long enough to judge."""
    gender = _Classifier(GENDER_MODEL, device)
    emotion = _Classifier(EMOTION_MODEL, device, _Emotion)
    age_fe = AutoFeatureExtractor.from_pretrained(AGE_MODEL)
    age_model = _AgeGender.from_pretrained(AGE_MODEL).to(device).eval()
    female = [k for k, v in gender.labels.items() if v == "female"][0]
    for line in tqdm(lines, desc="      voice features", ncols=90):
        if line["voice"] == "SFX" or line["end"] - line["start"] < MIN_FEATURE_SEC:
            continue
        s = int(line["start"] * SR)
        wav = audio[s:min(int(line["end"] * SR), s + int(MAX_FEATURE_SEC * SR))]
        line["p_female"] = round(float(gender.probs(wav)[female]), 3)
        x = age_fe(wav, sampling_rate=SR, return_tensors="pt").input_values.to(device)
        line["age"] = round(float(age_model(x)[0, 0]) * 100, 1)
        p = emotion.probs(wav)
        line["emotion"], line["emotion_score"] = emotion.labels[int(p.argmax())], round(float(p.max()), 3)
    del gender, emotion, age_model
    if device == "cuda":
        torch.cuda.empty_cache()


def profile(lines, key):
    """Per key (voice or character): speech seconds, duration-weighted p_female, median age."""
    groups = defaultdict(list)
    for line in lines:
        groups[line[key]].append(line)
    out = {}
    for name, ls in groups.items():
        judged = [l for l in ls if "p_female" in l]
        w = np.array([l["end"] - l["start"] for l in judged])
        info: dict = {"lines": len(ls), "seconds": round(sum(l["end"] - l["start"] for l in ls), 1)}
        if judged:
            p = float(np.average([l["p_female"] for l in judged], weights=w))
            years = float(np.median([l["age"] for l in judged]))
            info.update(p_female=round(p, 3), gender=gender_word(p), age=round(years, 1), age_group=age_group(years))
        out[name] = info
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["seconds"]))
