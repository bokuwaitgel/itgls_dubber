"""Who speaks when, and how: pyannote diarization + gender / age / emotion per line.

Lines are rebuilt from Scribe's word timings, but the speaker of every word comes from pyannote
(Scribe merges same-gender actors), so a line is split wherever the real voice changes.

Models:
  pyannote/speaker-diarization-3.1                            (gated: HF_TOKEN + accepted terms)
  pyannote/wespeaker-voxceleb-resnet34-LM                     speaker embedding per line (refine_voices)
  prithivMLmods/Common-Voice-Gender-Detection                 gender
  audeering/wav2vec2-large-robust-24-ft-age-gender            age in years + gender (female/male/child head)
  ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition   emotion (acted-speech model: a hint, not truth)

Gender per line fuses both gender models; where they disagree the line's voice decides (smooth_gender), so one
bad call doesn't flip a speaker, while a line both models agree on keeps its own gender.

Diarization also puts lines of one actor into another's voice (a mother's lines in her child's voice).
refine_voices compares each line's speaker embedding with its voice: a line that doesn't sound like it moves
to the voice it clearly matches, or to a new voice ("V21b") if it matches none.
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
EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
GENDER_MODEL = "prithivMLmods/Common-Voice-Gender-Detection"
AGE_MODEL = "audeering/wav2vec2-large-robust-24-ft-age-gender"
EMOTION_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

MAX_GAP = 0.8  # seconds of silence that starts a new line
EVENT_BRIDGE = 1.5  # sound events longer than this ([dragon roaring]) don't count as speech for line timing
MAX_WORD_SEC = 1.5  # Scribe sometimes stretches a word over the silence after it ("her." lasting 15 s)
MAX_LEN = 8.0  # split a line at the next sentence end after this many seconds
MIN_FEATURE_SEC = 0.6  # shorter lines get no gender/age/emotion
MAX_FEATURE_SEC = 10.0  # classifiers see at most this much of a line
BLIP_SEC = 0.5  # a speaker change shorter than this between two runs of the same voice is diarization noise
# Gender: audeering (robust to film audio) + Common Voice (read speech). A line is "sure" when both say the same
# with SURE_P confidence; otherwise its voice decides. Pitch is only a "low = male" hint: men shouting reach 300+ Hz.
W_AGE_GENDER = 0.6
SURE_P = 0.3  # both models below this (male) or above 1 - this (female)
VOICE_SHARE = 0.8  # a voice has a clear gender if this share of its sure seconds is one gender
VOICE_MIN_SEC = 3.0
LOW_F0 = 145.0  # Hz: an unsure line pitched below this on a voice of unclear gender leans male
F0_MIN, F0_MAX = 65.0, 450.0
# refine_voices (cosine similarity of WeSpeaker embeddings; lines of the same voice: median ~0.7, 5th pct ~0.4)
MIN_EMBED_SEC = 0.8  # shorter lines keep their voice
OUTLIER_SIM = 0.25  # a line below this to its own voice doesn't sound like it
MOVE_SIM, MOVE_MARGIN = 0.5, 0.2  # ...and moves to a voice it matches this well, by this much more than its own
NEW_VOICE_SIM = 0.5  # leftover outliers this similar to each other share one new voice
MIN_VOICE_LINES = 3  # voices with fewer embedded lines are neither checked nor moved to


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


def _cjk(ch):
    """Chinese/Japanese character or full-width punctuation: Scribe gives these one per word, joined without spaces."""
    return bool(ch) and ("\u3000" <= ch <= "\u9fff" or "\uff00" <= ch <= "\uffef")


def build_lines(scribe_words, turns):
    """Scribe words -> dub lines, split on voice change, pauses and long sentences.

    A voice change is either pyannote's or Scribe's: each one sometimes merges two actors the other keeps
    apart, and the per-line gender/age check tells the script model who is who.
    """
    words = [{**w, "end": min(w["end"], w["start"] + MAX_WORD_SEC)} if w["type"] == "word" else w
             for w in scribe_words if w["type"] in ("word", "audio_event") and w.get("start") is not None]
    speakers = word_speakers(words, turns) if turns else [w.get("speaker_id") or "V00" for w in words]
    lines, cur, prev_end, scribe_spk = [], None, None, None
    for w, spk in zip(words, speakers):
        is_word = w["type"] == "word"
        text = w["text"] if is_word else f"[{w['text'].strip('()[] ')}]"
        if not is_word and w["end"] - w["start"] > EVENT_BRIDGE:
            # A long sound ([dragon roaring] 8 s) is noted where it happens but is not speech: it must not glue
            # two sentences with a pause between them into one line, or stretch a line's end.
            if cur is None or w["start"] - prev_end > MAX_GAP:
                lines.append({"start": w["start"], "end": w["end"], "voice": None, "en": text, "speech_start": None})
            else:
                cur["en"] += " " + text
            continue
        changed = is_word and cur is not None and cur["voice"] is not None and (
            spk != cur["voice"] or w.get("speaker_id") != scribe_spk)
        new = (cur is None or changed or w["start"] - prev_end > MAX_GAP
               or (cur["end"] - cur["start"] > MAX_LEN and cur["en"].rstrip()[-1:] in ".?!。？！"))
        if is_word:
            scribe_spk = w.get("speaker_id")
        if new:
            cur = {"start": w["start"], "end": w["end"], "voice": None, "en": "", "speech_start": None}
            lines.append(cur)
        text = text.strip()
        cur["en"] += (" " if cur["en"] and not (_cjk(cur["en"][-1]) and _cjk(text[:1])) else "") + text
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


def embed_lines(lines, audio, device):
    """id -> L2-normalized speaker embedding, for every speech line long enough to embed."""
    from pyannote.audio import Inference, Model

    model = Model.from_pretrained(EMBEDDING_MODEL, use_auth_token=os.getenv("HF_TOKEN"))
    inference = Inference(model, window="whole", device=torch.device(device))
    out = {}
    for line in tqdm(lines, desc="      voice embeddings", ncols=90):
        if line["voice"] == "SFX" or line["end"] - line["start"] < MIN_EMBED_SEC:
            continue
        s = int(line["start"] * SR)
        wav = audio[s:min(int(line["end"] * SR), s + int(MAX_FEATURE_SEC * SR))]
        e = np.asarray(inference({"waveform": torch.from_numpy(np.ascontiguousarray(wav))[None], "sample_rate": SR}))
        if np.isfinite(e).all() and np.linalg.norm(e) > 0:
            out[line["id"]] = e / np.linalg.norm(e)
    del inference, model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def _centroid(ids, emb, lines_by_id):
    """Duration-weighted mean embedding, refit twice without the least similar quarter (the outliers)."""
    x = np.stack([emb[i] for i in ids])
    w = np.array([lines_by_id[i]["end"] - lines_by_id[i]["start"] for i in ids])
    keep = np.ones(len(ids), bool)
    for _ in range(3):
        c = (x[keep] * w[keep, None]).sum(0)
        c /= np.linalg.norm(c)
        if len(ids) < 8:
            break
        sim = x @ c
        keep = sim >= np.percentile(sim, 25)
    return c


def refine_voices(lines, emb):
    """Move lines that don't sound like their voice (see module doc). Moved lines keep the old id in voice_was.
    Returns the number of lines moved."""
    by_id = {l["id"]: l for l in lines}
    groups = defaultdict(list)
    for i in emb:
        groups[by_id[i]["voice"]].append(i)
    names = sorted(v for v, ids in groups.items() if len(ids) >= MIN_VOICE_LINES)
    if not names:
        return 0
    cents = np.stack([_centroid(groups[v], emb, by_id) for v in names])
    leftovers = defaultdict(list)  # voice -> outlier ids matching no other voice
    moved = 0
    for v in names:
        for i in groups[v]:
            sim = cents @ emb[i]
            own = sim[names.index(v)]
            by_id[i]["voice_sim"] = round(float(own), 3)
            if own >= OUTLIER_SIM:
                continue
            best = int(sim.argmax())
            if names[best] != v and sim[best] >= MOVE_SIM and sim[best] - own >= MOVE_MARGIN:
                by_id[i].update(voice_was=v, voice=names[best], voice_sim=round(float(sim[best]), 3))
                moved += 1
            else:
                leftovers[v].append(i)
    for v, ids in leftovers.items():  # greedy grouping into new voices V21b, V21c, ...
        clusters = []
        for i in sorted(ids, key=lambda i: -(by_id[i]["end"] - by_id[i]["start"])):
            match = next((c for c in clusters if emb[i] @ emb[c[0]] >= NEW_VOICE_SIM), None)
            if match:
                match.append(i)
            else:
                clusters.append([i])
        for k, c in enumerate(clusters):
            for i in c:
                by_id[i].update(voice_was=v, voice=f"{v}{chr(ord('b') + k)}" if k < 25 else f"{v}_{k}")
                moved += 1
    return moved


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
        hidden = self.wav2vec2(input_values)[0].mean(dim=1)
        return self.age(hidden), torch.softmax(self.gender(hidden), dim=-1)  # gender: female, male, child


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


def median_f0(wav, sr=SR):
    """Median pitch (Hz) of the voiced, louder frames of a clip, or None. Plain autocorrelation: a gender hint."""
    frame, hop = int(0.04 * sr), int(0.01 * sr)
    if len(wav) < frame:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(wav.astype(np.float32), frame)[::hop]
    frames = frames - frames.mean(axis=1, keepdims=True)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    frames = frames[rms > max(1e-4, np.percentile(rms, 40))]
    if len(frames) < 5:
        return None
    spec = np.fft.rfft(frames * np.hanning(frame), n=2 * frame)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:, :frame]
    ac = ac / np.maximum(ac[:, :1], 1e-9)
    lo, hi = int(sr / F0_MAX), int(sr / F0_MIN)
    lag = ac[:, lo:hi].argmax(axis=1) + lo
    voiced = ac[np.arange(len(ac)), lag] > 0.45
    if voiced.sum() < 5:
        return None
    return float(np.median(sr / lag[voiced]))


def smooth_gender(lines):
    """Unsure lines (the two models disagree) take their voice's gender when the voice is clearly one gender;
    sure lines keep their own call (an actor diarized into someone else's voice). Raw value stays in p_female_line."""
    groups = defaultdict(list)
    for line in lines:
        if "p_female" in line and line["voice"] != "SFX":
            groups[line["voice"]].append(line)
    for ls in groups.values():
        sure = [l for l in ls if l.get("gender_sure")]
        female = sum(l["end"] - l["start"] for l in sure if l["p_female"] > 0.5)
        total = sum(l["end"] - l["start"] for l in sure)
        clear = total >= VOICE_MIN_SEC and max(female, total - female) >= VOICE_SHARE * total
        if clear:
            side = [l["p_female"] for l in sure if (l["p_female"] > 0.5) == (female > total / 2)]
            voice_p = float(np.mean(side))
        for l in ls:
            if l.get("gender_sure"):
                continue
            if clear:
                l["p_female_line"] = l["p_female"]
                l["p_female"] = round(0.7 * voice_p + 0.3 * l["p_female"], 3)
            elif l.get("f0", 1e9) < LOW_F0:
                l["p_female_line"] = l["p_female"]
                l["p_female"] = min(l["p_female"], 0.3)


@torch.inference_mode()
def add_features(lines, audio, device):
    """p_female (two models + pitch), age (years) and emotion for every line long enough to judge."""
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
        p_cv = float(gender.probs(wav)[female])
        x = age_fe(wav, sampling_rate=SR, return_tensors="pt").input_values.to(device)
        age, g = age_model(x)
        f, m = float(g[0, 0]), float(g[0, 1])
        p_ag = f / (f + m) if f + m > 1e-6 else 0.5  # child mass left out: kids are cast by age, not gender
        f0 = median_f0(wav)
        line["p_female"] = round(W_AGE_GENDER * p_ag + (1 - W_AGE_GENDER) * p_cv, 3)
        line["gender_parts"] = {"age_gender": round(p_ag, 3), "common_voice": round(p_cv, 3)}
        line["gender_sure"] = max(p_ag, p_cv) < SURE_P or min(p_ag, p_cv) > 1 - SURE_P
        if f0:
            line["f0"] = round(f0, 1)
        line["age"] = round(float(age[0, 0]) * 100, 1)
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
