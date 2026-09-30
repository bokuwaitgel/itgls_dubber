"""Video in, stems out, video back: extract audio, split voice from background (Demucs), mux the dub."""
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .common import FFMPEG, MIX_RATE, duration, ffmpeg

SEP_CHUNK = 300.0  # seconds of audio per Demucs pass (bounds RAM on a 2-hour film)
SEP_PAD = 5.0  # context on both sides of a chunk, cut off afterwards so chunk edges are seamless


def cut(video, out, start, end):
    """Re-encoded excerpt of the video, for test runs on a few minutes."""
    ffmpeg("-ss", start, "-to", end, "-i", video, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-c:a", "aac", "-b:a", "192k", _part(out))
    _part(out).replace(out)


def _part(path):
    """Temp name next to path; stages write there and rename at the end, so a killed run never looks done."""
    return path.with_name(path.stem + ".part" + path.suffix)


def extract(video, out):
    ffmpeg("-i", video, "-vn", "-ac", 2, "-ar", MIX_RATE, "-c:a", "pcm_s16le", _part(out))
    _part(out).replace(out)


def separate(original, vocals_out, background_out, device):
    """Demucs htdemucs: vocals stem + everything else (music, effects) as background."""
    from demucs.api import Separator

    sep = Separator(model="htdemucs", device=device, progress=False)
    assert sep.samplerate == MIX_RATE
    info = sf.info(str(original))
    chunk, pad = int(SEP_CHUNK * MIX_RATE), int(SEP_PAD * MIX_RATE)
    with sf.SoundFile(str(original)) as src, \
            sf.SoundFile(str(_part(vocals_out)), "w", MIX_RATE, 2, "PCM_16") as voc, \
            sf.SoundFile(str(_part(background_out)), "w", MIX_RATE, 2, "PCM_16") as bg:
        for start in range(0, info.frames, chunk):
            a, b = max(0, start - pad), min(info.frames, start + chunk + pad)
            src.seek(a)
            x = torch.from_numpy(src.read(b - a, dtype="float32", always_2d=True).T.copy())
            _, stems = sep.separate_tensor(x, MIX_RATE)
            keep = slice(start - a, start - a + min(chunk, info.frames - start))
            vocals = stems["vocals"][:, keep]
            rest = torch.stack([s for name, s in stems.items() if name != "vocals"]).sum(0)[:, keep]
            voc.write(np.clip(vocals.T.numpy(), -1, 1))
            bg.write(np.clip(rest.T.numpy(), -1, 1))
            print(f"      separated {min(start + chunk, info.frames) / MIX_RATE / 60:5.1f} / "
                  f"{info.frames / MIX_RATE / 60:.1f} min", flush=True)
    _part(vocals_out).replace(vocals_out)
    _part(background_out).replace(background_out)
    del sep
    if device == "cuda":
        torch.cuda.empty_cache()


def video_size(path):
    err = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True, text=True, errors="ignore").stderr
    w, h = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", err).groups()
    return int(w), int(h)


def text_band(video, samples=60):
    """(top, bottom) as fractions of the height where the video already has subtitles burned in, or None.

    Samples frames and counts, per row, white pixels on sharp edges (outlined caption text). Subtitles come and go
    and change, so a row counts when it has text in some frames; rows with text in nearly every frame are a fixed
    logo or watermark and are ignored."""
    width, height = video_size(video)
    w = 270
    h = int(height * w / width) // 2 * 2
    total = duration(video)
    rows = []
    for k in range(samples):
        raw = subprocess.run([FFMPEG, "-v", "error", "-ss", f"{total * (k + 0.5) / samples:.2f}", "-i", str(video),
                              "-frames:v", "1", "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
                             capture_output=True).stdout
        if len(raw) != w * h:
            continue
        f = np.frombuffer(raw, np.uint8).reshape(h, w).astype(np.int16)
        rows.append(((f[:, 1:] > 200) & (np.abs(np.diff(f, axis=1)) > 60)).sum(1) / w)
    if len(rows) < samples // 2:
        return None
    rows = np.array(rows)
    present = (rows > 0.03).mean(0)
    score = np.percentile(rows, 75, axis=0) * (present < 0.85)  # constant overlays out
    lo = int(h * 0.45)
    peak = lo + int(np.argmax(score[lo:]))
    if score[peak] < 0.02 or present[peak] < 0.1:
        return None
    top = bottom = peak
    keep = lambda y: score[y] > 0.1 * score[peak] or 0.08 < present[y] < 0.85
    while top > lo and keep(top - 1):
        top -= 1
    while bottom < h - 1 and keep(bottom + 1):
        bottom += 1
    return top / h, (bottom + 1) / h


def _encoder():
    """GPU H.264 when this machine has NVENC (several times faster than the CPU), else x264."""
    probe = subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
                            "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True)
    if probe.returncode == 0:
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "21", "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]


def _burn_filter(ass_name, width, height, band):
    """ass subtitles; with `band`, the video's own subtitles there are blurred and darkened first."""
    if not band:
        return ["-vf", f"ass={ass_name}"]
    y = int(band[0] * height) // 2 * 2
    bh = max(48, int((band[1] - band[0]) * height) // 2 * 2)
    bar = (f"[0:v]split[v][b];[b]crop=iw:{bh}:0:{y},boxblur=luma_radius=20:luma_power=2:chroma_radius=10,"
           f"drawbox=x=0:y=0:w=iw:h=ih:color=black@0.45:t=fill[bar];[v][bar]overlay=0:{y},ass={ass_name}[out]")
    return ["-filter_complex", bar, "-map", "[out]"]


def mux(video, audio, out, subs=None, mode="soft", cover=True):
    """Original video with the dubbed audio. With an SRT in `subs`:
    soft  a Mongolian subtitle track viewers can switch off (video copied, no re-encode)
    burn  subtitles drawn into the picture, always visible (video re-encoded). With `cover`, subtitles already
          burned into the video are found, blurred under a dark bar, and the Mongolian ones go on that bar."""
    video, audio, out = (Path(p).resolve() for p in (video, audio, out))
    audio_args = ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]
    if not subs or mode == "none":
        ffmpeg("-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *audio_args, _part(out))
    elif mode == "soft":
        ffmpeg("-i", video, "-i", audio, "-i", Path(subs).resolve(), "-map", "0:v:0", "-map", "1:a:0", "-map", "2:s:0",
               "-c:v", "copy", "-c:s", "mov_text", "-metadata:s:s:0", "language=mon", "-metadata:s:s:0", "title=Монгол",
               "-disposition:s:0", "default", *audio_args, _part(out))
    else:
        from .subtitles import read_srt, write_ass
        ass = Path(subs).with_suffix(".ass")
        width, height = video_size(video)
        band = text_band(video) if cover else None
        if band:
            pad = 0.012
            band = (max(0.0, band[0] - pad), min(1.0, band[1] + pad))
            print(f"      covering the video's own subtitles at {band[0]:.0%}-{band[1]:.0%} of the height", flush=True)
        band = write_ass(read_srt(subs), ass, width, height, band)
        video_args = _burn_filter(ass.name, width, height, band)
        if video_args[0] == "-vf":
            video_args[:0] = ["-map", "0:v:0"]
        # The ass filter takes a bare file name: run from its folder so Windows drive colons need no escaping.
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(video), "-i", str(audio), *video_args, "-map", "1:a:0",
                        *_encoder(), *audio_args, str(_part(out))], check=True, cwd=ass.parent)
    _part(out).replace(out)
