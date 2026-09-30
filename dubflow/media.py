"""Video in, stems out, video back: extract audio, split voice from background (Demucs), mux the dub."""
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import FFMPEG, MIX_RATE, duration, ffmpeg, h264_encoder

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
    import torch
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


SAMPLE_W = 270  # frames are analysed at this width


def sample_frames(video, samples=60):
    """Grey frames spread over the whole video, SAMPLE_W wide: (frames, h)."""
    width, height = video_size(video)
    h = int(height * SAMPLE_W / width) // 2 * 2
    total = duration(video)
    frames = []
    for k in range(samples):
        raw = subprocess.run([FFMPEG, "-v", "error", "-ss", f"{total * (k + 0.5) / samples:.2f}", "-i", str(video),
                              "-frames:v", "1", "-vf", f"scale={SAMPLE_W}:{h},format=gray", "-f", "rawvideo", "-"],
                             capture_output=True).stdout
        if len(raw) == SAMPLE_W * h:
            frames.append(np.frombuffer(raw, np.uint8).reshape(h, SAMPLE_W).astype(np.int16))
    return frames, h


def _text_pixels(f):
    """White pixels on sharp edges: outlined caption or logo text."""
    return (f[:, 1:] > 200) & (np.abs(np.diff(f, axis=1)) > 60)


def text_band(frames, h):
    """(top, bottom) as fractions of the height where the video already has subtitles burned in, or None.

    Subtitles come and go and change, so a row counts when it has text in some frames; rows with text in nearly
    every frame are a fixed logo or watermark and are ignored."""
    if len(frames) < 20:
        return None
    rows = np.array([_text_pixels(f).sum(1) / SAMPLE_W for f in frames])
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


def logo_box(frames, h):
    """(x0, y0, x1, y1) as fractions of the frame around a channel logo that is in the same place in nearly every
    frame, near the top or bottom edge; None if there is none."""
    if len(frames) < 20:
        return None
    steady = np.mean([_text_pixels(f) for f in frames], axis=0) > 0.8
    for y0, y1 in ((int(h * 0.8), h), (0, int(h * 0.2))):  # bottom first: where these channels put theirs
        ys, xs = np.nonzero(steady[y0:y1])
        if len(ys) >= 15:
            return xs.min() / SAMPLE_W, (y0 + ys.min()) / h, (xs.max() + 2) / SAMPLE_W, (y0 + ys.max() + 1) / h
    return None


def _even(v):
    return int(round(v)) // 2 * 2


WATERMARK = {"auto": True, "x": 0.72, "y": 0.955, "size": 0.42, "opacity": 1.0, "box": True, "blur": True,
             "cover": True}


def _watermark_graph(cur, width, height, logo_size, found, style, logo_input):
    """Filter steps that hide the video's own logo (`cover`) and put ours on it (`auto`) or at x, y (centre, as
    fractions of the frame), `size` = logo width as a fraction of the frame width. Behind it the video can be
    blurred (`blur`, frosted glass) and/or darkened (`box`)."""
    st = {**WATERMARK, **(style or {})}
    lw = _even(width * st["size"])
    lh = _even(lw * logo_size[1] / logo_size[0])
    pad = _even(lh * 0.25)
    steps = []
    if found and (st["cover"] or st["auto"]):
        # thin script logos only half show up in detection: take a generous margin around what was found
        fx0, fx1 = max(0, found[0] * width - width * 0.05), min(width, found[2] * width + width * 0.05)
        fy0, fy1 = max(0, found[1] * height - height * 0.012), min(height, found[3] * height + height * 0.012)
        if st["auto"]:
            st["x"], st["y"] = (fx0 + fx1) / 2 / width, (fy0 + fy1) / 2 / height
        if st["cover"]:
            m = 0
            x, y = _even(max(0, fx0 - m)), _even(max(0, fy0 - m))
            w, h = _even(min(width - x, fx1 - fx0 + 2 * m)), _even(min(height - y, fy1 - fy0 + 2 * m))
            r = max(1, min(min(w, h) // 4, 20))  # boxblur: radius must fit the half-size chroma planes
            steps.append(f"[{cur}]split[ca][cb];[cb]crop={w}:{h}:{x}:{y},boxblur=luma_radius={r}:chroma_radius={max(1, r // 2)}:"
                         f"luma_power=2,drawbox=x=0:y=0:w=iw:h=ih:color=black@0.35:t=fill[cov];[ca][cov]overlay={x}:{y}[covered]")
            cur = "covered"
    bw, bh = lw + 2 * pad, lh + 2 * pad
    if st["auto"] and found and (st["box"] or st["blur"]):  # the backing must hide the whole old logo
        bw, bh = max(bw, _even(fx1 - fx0)), max(bh, _even(fy1 - fy0))
    bx = _even(min(max(st["x"] * width - bw / 2, 0), width - bw))
    by = _even(min(max(st["y"] * height - bh / 2, 0), height - bh))
    lx, ly = bx + (bw - lw) // 2, by + (bh - lh) // 2
    if st["blur"]:
        r = max(1, min(min(bw, bh) // 4, 20))  # boxblur: radius must fit the half-size chroma planes
        steps.append(f"[{cur}]split[ba][bb];[bb]crop={bw}:{bh}:{bx}:{by},boxblur=luma_radius={r}:"
                     f"chroma_radius={max(1, r // 2)}:luma_power=3[frost];[ba][frost]overlay={bx}:{by}[blurred]")
        cur = "blurred"
    if st["box"]:  # lighter over blur: the blur already makes the logo readable
        alpha = (0.3 if st["blur"] else 0.55) * st["opacity"]
        steps.append(f"[{cur}]drawbox=x={bx}:y={by}:w={bw}:h={bh}:color=black@{alpha:.2f}:t=fill[boxed]")
        cur = "boxed"
    steps.append(f"[{logo_input}:v]scale={lw}:-2,format=rgba,colorchannelmixer=aa={st['opacity']:.2f}[logo];"
                 f"[{cur}][logo]overlay={lx}:{ly}:shortest=1[marked]")
    return steps, "marked"


def mux(video, audio, out, subs=None, mode="soft", cover=True, logo=None, logo_style=None):
    """Original video with the dubbed audio, plus optionally:

    subs + mode  soft: a Mongolian subtitle track viewers can switch off.
                 burn: subtitles drawn into the picture. With `cover`, subtitles already burned into the video
                 are found, blurred under a dark bar, and the Mongolian ones go on that bar.
    logo         an image (PNG with transparency) as a watermark, placed by `logo_style` (see WATERMARK and
                 _watermark_graph): by default over the video's own channel logo on a dark box, hiding that one.
    Without burn or logo the picture is copied (instant); with either it is re-encoded once."""
    video, audio, out = (Path(p).resolve() for p in (video, audio, out))
    subs = Path(subs).resolve() if subs and mode != "none" else None
    logo = Path(logo).resolve() if logo else None
    burn = subs is not None and mode == "burn"
    args = ["-i", str(video), "-i", str(audio)]
    maps = ["-map", "1:a:0"]
    cwd = None
    if burn or logo:
        width, height = video_size(video)
        frames, h = sample_frames(video)
        graph, cur = [], "0:v"
        if burn:
            from .subtitles import read_srt, write_ass
            band = text_band(frames, h) if cover else None
            if band:
                band = (max(0.0, band[0] - 0.012), min(1.0, band[1] + 0.012))
                print(f"      covering the video's own subtitles at {band[0]:.0%}-{band[1]:.0%} of the height", flush=True)
            ass = subs.with_suffix(".ass")
            band = write_ass(read_srt(subs), ass, width, height, band)
            if band:
                y, bh = _even(band[0] * height), max(48, _even((band[1] - band[0]) * height))
                graph.append(f"[{cur}]split[sa][sb];[sb]crop=iw:{bh}:0:{y},boxblur=luma_radius=20:luma_power=2:"
                             f"chroma_radius=10,drawbox=x=0:y=0:w=iw:h=ih:color=black@0.45:t=fill[bar];"
                             f"[sa][bar]overlay=0:{y}[subbed]")
                cur = "subbed"
        if logo:
            st = {**WATERMARK, **(logo_style or {})}
            found = logo_box(frames, h) if st["auto"] or st["cover"] else None
            print(f"      watermark{' over the video’s own logo' if found and st['auto'] else ''}"
                  f"{'; the video’s own logo hidden' if found and st['cover'] else ''}", flush=True)
            args += ["-loop", "1", "-i", str(logo)]
            steps, cur = _watermark_graph(cur, width, height, video_size(logo), found, st, 2)
            graph += steps
        if burn:
            # The ass filter takes a bare file name: run from its folder so Windows drive colons need no escaping.
            graph.append(f"[{cur}]ass={ass.name}[subs]")
            cur, cwd = "subs", ass.parent
        args += ["-filter_complex", ";".join(graph)]
        maps = ["-map", f"[{cur}]"] + maps
        codec = h264_encoder()
    else:
        maps = ["-map", "0:v:0"] + maps
        codec = ["-c:v", "copy"]
    if subs and not burn:
        args += ["-i", str(subs)]
        maps += ["-map", f"{args.count('-i') - 1}:s:0"]
        codec += ["-c:s", "mov_text", "-metadata:s:s:0", "language=mon", "-metadata:s:s:0", "title=Монгол",
                  "-disposition:s:0", "default"]
    subprocess.run([FFMPEG, "-v", "error", "-y", *args, *maps, *codec, "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", str(_part(out))], check=True, cwd=cwd)
    _part(out).replace(out)
