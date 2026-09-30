"""Diarize audio, then tag each speaker with gender, age group and emotion.

Models:
  Diarization: pyannote/speaker-diarization-3.1 (gated; needs HF_TOKEN + accepted terms)
  Gender:      prithivMLmods/Common-Voice-Gender-Detection
  Age:         audeering/wav2vec2-large-robust-24-ft-age-gender
               (audeering/wav2vec2-large-robust-12-ft-age-group does not exist on HF)
  Emotion:     ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition

Usage:
  python analyze_speakers.py village_craft_stall.mp3 [--out result.json] [--num-speakers N]
"""

import argparse
import json
import os
import subprocess
from collections import Counter, defaultdict

import imageio_ffmpeg
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

SR = 16000
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
GENDER_MODEL = "prithivMLmods/Common-Voice-Gender-Detection"
AGE_MODEL = "audeering/wav2vec2-large-robust-24-ft-age-gender"
EMOTION_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

MIN_SEG_SEC = 1.0   # skip shorter turns for classification
MAX_SEG_SEC = 10.0  # crop longer turns (wav2vec2 memory)
MAX_SEGS_PER_SPEAKER = 40  # speaker-level gender/age uses the longest N turns


def load_audio(path: str) -> np.ndarray:
    """Decode any format to 16 kHz mono float32 via bundled ffmpeg."""
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", path,
        "-ac", "1", "-ar", str(SR), "-f", "f32le", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


# --- audeering age/gender head (custom architecture from the model card) ---
class RegressionHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = RegressionHead(config, 1)
        self.gender = RegressionHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        hidden = self.wav2vec2(input_values)[0].mean(dim=1)
        return self.age(hidden), torch.softmax(self.gender(hidden), dim=1)


def age_group(years: float) -> str:
    if years < 13:
        return "child"
    if years < 20:
        return "teen"
    if years < 35:
        return "young_adult"
    if years < 55:
        return "adult"
    return "senior"


class Classifier:
    def __init__(self, name: str, device: str):
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(name).to(device).eval()
        self.labels = self.model.config.id2label
        self.device = device

    @torch.inference_mode()
    def probs(self, wav: np.ndarray) -> np.ndarray:
        inputs = self.fe(wav, sampling_rate=SR, return_tensors="pt").to(self.device)
        return torch.softmax(self.model(**inputs).logits, dim=-1)[0].cpu().numpy()


class AgeEstimator:
    def __init__(self, device: str):
        self.fe = AutoFeatureExtractor.from_pretrained(AGE_MODEL)
        self.model = AgeGenderModel.from_pretrained(AGE_MODEL).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def years(self, wav: np.ndarray) -> float:
        x = self.fe(wav, sampling_rate=SR, return_tensors="pt").input_values.to(self.device)
        age, _ = self.model(x)
        return float(age[0, 0]) * 100


def diarize(audio: np.ndarray, device: str, num_speakers: int | None):
    from pyannote.audio import Pipeline

    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN missing. Add it to .env and accept terms at "
            "https://hf.co/pyannote/speaker-diarization-3.1 and "
            "https://hf.co/pyannote/segmentation-3.0"
        )
    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=token)
    pipeline.to(torch.device(device))
    waveform = {"waveform": torch.from_numpy(audio)[None], "sample_rate": SR}
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    annotation = pipeline(waveform, **kwargs)
    return [
        {"speaker": spk, "start": round(turn.start, 3), "end": round(turn.end, 3)}
        for turn, _, spk in annotation.itertracks(yield_label=True)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--out", default=None)
    ap.add_argument("--num-speakers", type=int, default=None)
    args = ap.parse_args()
    load_dotenv()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = args.out or os.path.splitext(args.audio)[0] + "_speakers.json"

    print(f"[1/3] Decoding {args.audio} ({device})")
    audio = load_audio(args.audio)
    print(f"      {len(audio) / SR / 60:.1f} min")

    print("[2/3] Diarizing")
    cache_path = os.path.splitext(args.audio)[0] + "_diarization.json"
    if os.path.exists(cache_path):
        print(f"      using cached {cache_path} (delete to re-diarize)")
        with open(cache_path, encoding="utf-8") as f:
            segments = json.load(f)
    else:
        segments = diarize(audio, device, args.num_speakers)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(segments, f)
    print(f"      {len(segments)} turns, {len({s['speaker'] for s in segments})} speakers")

    print("[3/3] Classifying")
    gender_clf = Classifier(GENDER_MODEL, device)
    emotion_clf = Classifier(EMOTION_MODEL, device)
    age_est = AgeEstimator(device)

    def clip(seg):
        s = int(seg["start"] * SR)
        e = min(int(seg["end"] * SR), s + int(MAX_SEG_SEC * SR))
        return audio[s:e]

    # Emotion per turn (turns >= MIN_SEG_SEC).
    for seg in tqdm(segments, desc="emotion"):
        if seg["end"] - seg["start"] < MIN_SEG_SEC:
            seg["emotion"] = None
            continue
        p = emotion_clf.probs(clip(seg))
        i = int(p.argmax())
        seg["emotion"] = emotion_clf.labels[i]
        seg["emotion_score"] = round(float(p[i]), 3)

    # Gender/age per speaker, averaged over their longest turns.
    by_speaker = defaultdict(list)
    for seg in segments:
        by_speaker[seg["speaker"]].append(seg)

    speakers = {}
    for spk, segs in tqdm(by_speaker.items(), desc="speakers"):
        usable = [s for s in segs if s["end"] - s["start"] >= MIN_SEG_SEC]
        usable = sorted(usable, key=lambda s: s["end"] - s["start"], reverse=True)[:MAX_SEGS_PER_SPEAKER]
        total = sum(s["end"] - s["start"] for s in segs)
        info: dict = {"total_speech_sec": round(total, 1), "turns": len(segs)}
        if usable:
            g = np.mean([gender_clf.probs(clip(s)) for s in usable], axis=0)
            years = float(np.median([age_est.years(clip(s)) for s in usable]))
            emotions = Counter(s["emotion"] for s in segs if s.get("emotion"))
            info.update(
                gender=gender_clf.labels[int(g.argmax())],
                gender_score=round(float(g.max()), 3),
                age_years=round(years, 1),
                age_group=age_group(years),
                dominant_emotion=emotions.most_common(1)[0][0],
                emotion_counts=dict(emotions),
            )
        speakers[spk] = info

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"speakers": speakers, "segments": segments}, f, indent=2)

    print(f"\nSaved {out_path}")
    for spk, info in speakers.items():
        print(f"  {spk}: {info.get('gender')} ({info.get('gender_score')}), "
              f"{info.get('age_group')} (~{info.get('age_years')}y), "
              f"{info.get('dominant_emotion')}, {info['total_speech_sec']}s")


if __name__ == "__main__":
    main()
