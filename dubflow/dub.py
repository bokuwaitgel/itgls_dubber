"""TTS every script line with ElevenLabs (eleven_v4 default), fit it to its slot, mix it over the film.

With Demucs stems the dub sits on the clean background (music + effects) and the original voices are muted
under each dub line (laughs, screams and anything undubbed between lines stay). Without stems the whole
original is ducked instead. Mixing streams in chunks, so a 2-hour film fits in memory.

Lines are cached in cache/tts/ keyed by model+voice+text (same key as the old dubbing/scripts/dub.py cache,
so --tts-cache ../dubbing/cache reuses lines already paid for).
"""
import hashlib
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import soundfile as sf

from .common import MIX_RATE, ROOT, TAG, decode, ffmpeg

MODEL_ID = "eleven_v4"  # overridden by --tts-model; no language_code needed, reads Cyrillic
TTS_RATE = 24000
STABILITY = 0.5
WORKERS = 4  # parallel TTS requests (ElevenLabs concurrency limit depends on the plan)
MAX_TEMPO = 1.35  # fastest speed-up allowed to fit a line before the next one starts
LINE_RMS = 0.1  # loudness every dub line is leveled to (about -20 dBFS), times --dub-gain
FADE = 0.15  # seconds of ramp into/out of ducking
MIX_CHUNK = 60.0  # seconds mixed per step
DEFAULT_CACHE = ROOT / "cache" / "tts"
# How far from the mic a line sounds: trimmed lows/highs, softer presence, short room echo, quieter.
DISTANCE = {
    "close": ("", 1.0),  # dry TTS as generated
    "medium": ("highpass=f=110,lowpass=f=7500,equalizer=f=3000:t=q:w=1.2:g=-3,aecho=0.8:0.7:22|41:0.22|0.12,", 0.8),
    "far": ("highpass=f=180,lowpass=f=5000,equalizer=f=2500:t=q:w=1.2:g=-5,aecho=0.8:0.6:35|70|110:0.35|0.25|0.15,", 0.6),
}


def _voice(voices, speaker):
    entry = voices[speaker]
    return (entry["voice_id"], entry.get("pitch", 0)) if isinstance(entry, dict) else (entry, 0)


def _cache_path(cache, text, voice_id, stability):
    parts = [MODEL_ID, voice_id, text] if stability == STABILITY else [MODEL_ID, voice_id, str(stability), text]
    return cache / f"{hashlib.sha1('|'.join(parts).encode()).hexdigest()[:16]}.wav"


def spoken_lines(script):
    return [l for l in script if TAG.sub("", l["text"]).strip()]  # sound-only lines aren't voiced


def estimate(script, voices, cache, stability=STABILITY):
    """(characters to generate, lines to generate, total lines) - ElevenLabs bills ~1 credit per character."""
    todo = [l for l in spoken_lines(script) if not _cache_path(cache, l["text"], _voice(voices, l["speaker"])[0],
                                                               stability).exists()]
    return sum(len(l["text"]) for l in todo), len(todo), len(spoken_lines(script))


def generate(lines, voices, cache, stability=STABILITY, workers=WORKERS):
    """TTS every uncached line, `workers` requests at a time; backs off and retries when rate-limited."""
    from elevenlabs import ElevenLabs, VoiceSettings

    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    todo = {}  # cache path -> (text, voice_id); identical lines are generated once
    for line in lines:
        voice_id = _voice(voices, line["speaker"])[0]
        path = _cache_path(cache, line["text"], voice_id, stability)
        if not path.exists():
            todo[path] = (line["text"], voice_id)
    if not todo:
        return 0

    def one(path, text, voice_id):
        for attempt in range(6):
            try:
                audio = client.text_to_speech.convert(
                    text=text, voice_id=voice_id, model_id=MODEL_ID, output_format=f"pcm_{TTS_RATE}",
                    voice_settings=VoiceSettings(stability=stability, similarity_boost=0.75))
                pcm = np.frombuffer(b"".join(audio), np.int16)
                break
            except Exception as e:  # 429 too many requests / 409 voice still being added / 5xx / network: retry
                status = getattr(e, "status_code", None)
                if status is not None and status < 500 and status not in (409, 429) or attempt == 5:
                    raise
                time.sleep(2 ** attempt + random.random())
        tmp = path.with_name(path.stem + ".part.wav")
        sf.write(str(tmp), pcm, TTS_RATE, subtype="PCM_16")
        tmp.replace(path)  # only complete files count as cached
        return len(text)

    credits, started = 0, time.time()
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(one, p, t, v) for p, (t, v) in todo.items()]
        for k, fut in enumerate(as_completed(futures), 1):
            credits += fut.result()
            if k % 25 == 0 or k == len(futures):
                eta = (time.time() - started) / k * (len(futures) - k) / 60
                print(f"      generated {k}/{len(futures)} lines, {credits} credits, ~{eta:.0f} min left", flush=True)
    return credits


def render(script, voices, cache, seg_dir, stability=STABILITY, dub_gain=1.3, distance="medium", workers=WORKERS):
    """Generate (or load) every line; returns [(start_sec, samples, duck_from, duck_to)]."""
    cache.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    lines = spoken_lines(script)
    credits = generate(lines, voices, cache, stability, workers)
    clips = []
    for i, line in enumerate(lines):
        voice_id, pitch = _voice(voices, line["speaker"])
        path = _cache_path(cache, line["text"], voice_id, stability)

        # Speed up (never slow down) so the line ends before the next one starts.
        next_start = lines[i + 1]["start"] if i + 1 < len(lines) else line["start"] + 60
        slot = max(next_start - line["start"] - 0.05, line["end"] - line["start"])
        tempo = min(max(sf.info(str(path)).duration / slot, 1.0), MAX_TEMPO)
        f = 2 ** (pitch / 12)  # pitch shift that keeps duration lets one voice cover another character
        shift = f"asetrate={TTS_RATE * f:.0f},aresample={TTS_RATE},atempo={1 / f:.4f}," if pitch else ""
        room, room_gain = DISTANCE[distance]
        samples = decode(path, MIX_RATE, filters=shift + room + f"atempo={tempo:.3f}")
        rms = float(np.sqrt(np.mean(samples ** 2))) or 1.0
        samples = samples * min(LINE_RMS * dub_gain * room_gain / rms, 0.95 / (np.abs(samples).max() or 1.0))

        name = f"{i + 1:04d}_{line['speaker'].replace(' ', '').lower()}_{int(line['start'])}s.wav"
        sf.write(str(seg_dir / name), samples, MIX_RATE, subtype="PCM_16")
        end = line["start"] + len(samples) / MIX_RATE
        # Duck across the original line too, so no English tail sticks out past a shorter dub.
        clips.append((line["start"], samples, line["start"], max(line["end"], end)))
        over = end - next_start
        note = (f"  x{tempo:.2f}" if tempo > 1.0 else "") + (f"  OVERLAPS next by {over:.2f}s" if over > 0 else "")
        print(f"      [{i + 1}/{len(lines)}] {line['speaker']:<18} {len(samples) / MIX_RATE:5.2f}s{note}", flush=True)
    print(f"      credits used this run: {credits}")
    return clips


def _duck_curve(clips, duck, total):
    """Breakpoints of a gain curve: 1 outside dub lines, `duck` under them, FADE-long ramps."""
    spans = sorted((max(0.0, a - FADE), b + FADE) for _, _, a, b in clips)
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    xs, ys = [0.0], [1.0]
    for a, b in merged:
        xs += [a, a + FADE, max(a + FADE, b - FADE), b]
        ys += [1.0, duck, duck, 1.0]
    return np.array(xs + [total + 1]), np.array(ys + [1.0])


def mix(clips, out_wav, duck, original=None, vocals=None, background=None):
    """Stems given: background + ducked vocals + dub. Else: ducked original + dub."""
    bed = background or original
    info = sf.info(str(bed))
    total = info.frames / MIX_RATE
    xs, ys = _duck_curve(clips, duck, total)
    step = int(MIX_CHUNK * MIX_RATE)
    readers = [sf.SoundFile(str(p)) for p in ([background, vocals] if background else [original])]
    try:
        with sf.SoundFile(str(out_wav), "w", MIX_RATE, 2, "PCM_16") as out:
            for a in range(0, info.frames, step):
                n = min(step, info.frames - a)
                gain = np.interp((a + np.arange(n)) / MIX_RATE, xs, ys).astype(np.float32)[:, None]
                parts = [r.read(n, dtype="float32", always_2d=True) for r in readers]
                block = parts[0] + parts[1] * gain if background else parts[0] * gain
                dub = np.zeros(n, np.float32)
                for start, samples, _, _ in clips:
                    s = int(start * MIX_RATE) - a
                    if s >= n or s + len(samples) <= 0:
                        continue
                    lo, hi = max(0, s), min(n, s + len(samples))
                    dub[lo:hi] += samples[lo - s:hi - s]
                out.write(np.clip(block + dub[:, None], -1, 1))
    finally:
        for r in readers:
            r.close()
    ffmpeg("-i", out_wav, "-b:a", "192k", out_wav.with_suffix(".mp3"))
