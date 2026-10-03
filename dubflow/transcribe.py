"""English transcript with word timings from ElevenLabs Scribe v2.

A whole film is one 60-90 MB upload, which home connections often drop half way (connection reset). So the audio
goes up in ~10 minute pieces, cut in the quietest moment near each boundary so no word is split. Each piece is
retried on network errors and kept on disk once transcribed, so a rerun never pays for a piece twice.
"""
import os
import time

import numpy as np

from .common import decode, ffmpeg, load_json, save_json

PIECE_SEC = 600      # target piece length
SEARCH_SEC = 30      # look this far either side of a target cut for a pause
ENERGY_RATE = 4000   # Hz, enough to find pauses
FRAME_SEC = 0.05
TRIES = 4


def cut_points(audio):
    """Piece boundaries in seconds: [0, ..., duration], each inner cut at the quietest 50 ms near a multiple of
    PIECE_SEC."""
    x = decode(audio, ENERGY_RATE)
    hop = int(ENERGY_RATE * FRAME_SEC)
    n = len(x) // hop
    energy = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
    total = len(x) / ENERGY_RATE
    cuts = [0.0]
    while total - cuts[-1] > PIECE_SEC * 1.5:  # the last piece takes the rest, 5-15 min, never a tiny scrap
        target = cuts[-1] + PIECE_SEC
        lo, hi = int((target - SEARCH_SEC) / FRAME_SEC), int((target + SEARCH_SEC) / FRAME_SEC)
        cuts.append((lo + int(np.argmin(energy[lo:hi]))) * FRAME_SEC)
    return cuts + [total]


def _scribe(client, path):
    import httpx

    for attempt in range(1, TRIES + 1):
        try:
            with path.open("rb") as f:
                return client.speech_to_text.convert(
                    file=f, model_id="scribe_v2", diarize=True, tag_audio_events=True,
                    timestamps_granularity="word", request_options={"timeout_in_seconds": 1800},
                ).model_dump()
        except httpx.TransportError as e:  # reset, timeout, DNS: the network, not the request
            if attempt == TRIES:
                raise
            wait = 15 * attempt
            print(f"      network error ({type(e).__name__}), retrying in {wait}s", flush=True)
            time.sleep(wait)


def merge(parts, cuts):
    """One Scribe-shaped transcript from per-piece results: times shifted to the film, and each piece's speaker ids
    kept apart (Scribe numbers speakers per request; pyannote does the real speaker work later)."""
    words = []
    for n, (part, start) in enumerate(zip(parts, cuts)):
        for w in part["words"]:
            words.append({**w,
                          "start": None if w["start"] is None else round(w["start"] + start, 3),
                          "end": None if w["end"] is None else round(w["end"] + start, 3),
                          "speaker_id": w.get("speaker_id") and f"p{n}_{w['speaker_id']}"})
    return {**parts[0], "text": " ".join(p["text"].strip() for p in parts), "words": words,
            "audio_duration_secs": cuts[-1], "transcription_id": None}


def transcribe(audio, work, out_json):
    from elevenlabs import ElevenLabs

    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    cuts = cut_points(audio)
    pieces = work / "scribe_pieces"
    pieces.mkdir(exist_ok=True)
    parts = []
    for n, (start, end) in enumerate(zip(cuts, cuts[1:])):
        done = pieces / f"{n:03d}_{start:.2f}.json"  # the start in the name: a different cut is a different piece
        if done.exists():
            parts.append(load_json(done))
            continue
        print(f"      piece {n + 1}/{len(cuts) - 1}  {start / 60:.1f}-{end / 60:.1f} min", flush=True)
        upload = pieces / f"{n:03d}.mp3"
        ffmpeg("-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", audio, "-ac", 1, "-b:a", "96k", upload)
        result = _scribe(client, upload)
        upload.unlink()
        save_json(done, result)
        parts.append(result)
    save_json(out_json, merge(parts, cuts))
    for p in pieces.iterdir():
        p.unlink()
    pieces.rmdir()


def import_transcript(src_json, out_json, start=0.0, end=None):
    """Reuse an existing Scribe JSON, shifted/trimmed to a --start/--end test window."""
    data = load_json(src_json)
    words = []
    for w in data["words"]:
        if w["start"] is None or w["start"] < start or (end is not None and w["start"] >= end):
            continue
        words.append({**w, "start": w["start"] - start, "end": w["end"] - start})
    save_json(out_json, {**data, "words": words})
