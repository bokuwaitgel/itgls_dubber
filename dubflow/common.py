"""Shared helpers: ffmpeg I/O, timestamps, the dub script format, JSON."""
import json
import os
import re
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
MIX_RATE = 44100  # original, stems and final mix
ANALYSIS_RATE = 16000  # diarization and classifiers

HEADER = re.compile(r"(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)\s*\[(.+?)\]")
TAG = re.compile(r"\[[^\]]*\]")
HEADER_SRT = re.compile(r"(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)")


@contextmanager
def gpu_lock(device):
    """One GPU-heavy stage at a time across every running pipeline (an 8 GB card can't hold two jobs' models).
    A file lock, so a killed process releases it."""
    if device != "cuda":
        yield
        return
    path = ROOT / "cache" / "gpu.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as f:
        waiting = False
        while True:
            try:
                _lock_file(f, True)
                break
            except OSError:
                if not waiting:
                    print("      waiting for the GPU (another job is using it)", flush=True)
                    waiting = True
                time.sleep(2)
        try:
            yield
        finally:
            _lock_file(f, False)


def _lock_file(f, lock):
    if os.name == "nt":
        import msvcrt
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK if lock else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(f, (fcntl.LOCK_EX | fcntl.LOCK_NB) if lock else fcntl.LOCK_UN)


def ffmpeg(*args):
    subprocess.run([FFMPEG, "-v", "error", "-y", *map(str, args)], check=True)


def decode(path, rate, channels=1, filters=None):
    """Whole file as float32 samples, shape (n,) for mono or (n, channels)."""
    cmd = [FFMPEG, "-v", "error", "-i", str(path)]
    if filters:
        cmd += ["-filter:a", filters]
    cmd += ["-ac", str(channels), "-ar", str(rate), "-f", "f32le", "-"]
    x = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32)
    return x if channels == 1 else x.reshape(-1, channels)


def duration(path):
    return sf.info(str(path)).duration if Path(path).suffix == ".wav" else _probe_duration(path)


def _probe_duration(path):
    err = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True, text=True, errors="ignore").stderr
    h, m, s = err.split("Duration: ")[1].split(",")[0].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def stamp(t):
    ms = round(t * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def to_seconds(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def read_script(path):
    """Blocks of "00:00:05,200 --> 00:00:05,920 [Speaker]" + text, separated by blank lines."""
    lines = []
    for block in re.split(r"\n\s*\n", Path(path).read_text(encoding="utf-8").strip()):
        rows = block.strip().splitlines()
        match = HEADER.match(rows[0]) if rows else None
        if match:
            g = match.groups()
            lines.append({"start": to_seconds(*g[0:4]), "end": to_seconds(*g[4:8]), "speaker": g[8],
                          "text": " ".join(rows[1:]).strip()})
    return lines


def write_script(path, lines, key="text"):
    Path(path).write_text(
        "\n\n".join(f"{stamp(d['start'])} --> {stamp(d['end'])} [{d['speaker']}]\n{d[key]}" for d in lines) + "\n",
        encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
