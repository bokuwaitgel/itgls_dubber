"""The film's own burned-in English subtitles, read off the picture, as the true wording of each line.

Scribe mishears names and shouted lines ("Lin Weiwei!" -> "Lead way") and sometimes misses a line under screams or
music; the subtitle in the picture has it right. The subtitle band is scanned a few times a second, every change of
subtitle is one event, one crop per event goes to the script model (20 numbered crops per image, read by Codex at
no API cost), and each transcript line gets the text shown while it is spoken as "on_screen". Subtitles that no
line covers become new lines, so a missed line still gets dubbed.
"""
import hashlib
import io
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pydantic import BaseModel

from .common import FFMPEG, ROOT, load_json, save_json
from .media import _text_pixels, sample_frames, text_band, video_size

SCAN_FPS = 4  # frames per second scanned for subtitle changes
CROP_W = 720  # crops are read at this width
SAME = 0.8  # text masks overlapping at least this much (IoU) are the same subtitle; strict, because often only
#             the small English half changes under the same Chinese line ("Lin Weiwei! I'll haunt" -> "you from
#             my grave!"); repeats of one text are merged afterwards
MIN_EVENT = 0.3  # seconds; shorter flickers are dropped
BATCH = 40  # crops per image sent to the model (each request also carries ~20k tokens of Codex overhead)
WORKERS = 4  # images read in parallel
MERGE_GAP = 0.5  # the same text again within this many seconds is one subtitle
NEAR = 0.3  # seconds of slack when matching subtitles to lines
CACHE_DIR = ROOT / "cache" / "llm"

PROMPT = """The image is a stack of numbered strips cut from a video: each strip is the subtitle area of one moment,
with its number on the left. Most strips hold a Chinese line and an English line. For every number, return the
English subtitle text exactly as written (fix only obvious letter-level OCR slips; keep the punctuation). If a
strip has no English text, return an empty string for it."""


class Strip(BaseModel):
    n: int
    en: str


class Strips(BaseModel):
    items: list[Strip]


def _small(mask):
    """4x4-block 'any text' grid, for cheap same-subtitle comparisons."""
    h, w = mask.shape[0] // 4 * 4, mask.shape[1] // 4 * 4
    return mask[:h, :w].reshape(h // 4, 4, w // 4, 4).any(axis=(1, 3))


def events(video, band):
    """[(start, end, crop)] per subtitle shown in band (fractions of the height): crop is a grey uint8 image."""
    width, height = video_size(video)
    y0, bh = int(band[0] * height) // 2 * 2, max(2, int((band[1] - band[0]) * height)) // 2 * 2
    ch = int(bh * CROP_W / width) // 2 * 2
    proc = subprocess.Popen([FFMPEG, "-v", "error", "-i", str(video), "-an", "-vf",
                             f"fps={SCAN_FPS},crop=iw:{bh}:0:{y0},scale={CROP_W}:{ch},format=gray",
                             "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    out, cur, i, size = [], None, 0, CROP_W * ch
    while len(raw := proc.stdout.read(size)) == size:
        frame = np.frombuffer(raw, np.uint8).reshape(ch, CROP_W)
        mask = _text_pixels(frame.astype(np.int16))
        has = ((mask.sum(1) / CROP_W) > 0.02).sum() >= 2
        t = i / SCAN_FPS
        small = _small(mask) if has else None
        if cur and has and (small & cur["ref"]).sum() / max(1, (small | cur["ref"]).sum()) >= SAME:
            cur["end"] = t + 1 / SCAN_FPS
            if cur["n"] == 1:  # second frame of a subtitle: past any fade-in, the one to read
                cur["crop"] = frame.copy()
            cur["n"] += 1
        else:
            if cur:
                out.append(cur)
            cur = {"start": t, "end": t + 1 / SCAN_FPS, "ref": small, "crop": frame.copy(), "n": 1} if has else None
        i += 1
    proc.wait()
    if cur:
        out.append(cur)
    return [(e["start"], e["end"], e["crop"]) for e in out if e["end"] - e["start"] >= MIN_EVENT]


def _sheet(crops, first):
    """Numbered strips stacked into one PNG (bytes)."""
    from PIL import Image, ImageDraw, ImageFont

    label, gap = 70, 6
    h = sum(c.shape[0] for c in crops) + gap * len(crops)
    img = Image.new("L", (label + CROP_W, h), 255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    y = 0
    for k, c in enumerate(crops):
        img.paste(Image.fromarray(c), (label, y))
        draw.text((6, y + c.shape[0] // 2 - 14), str(first + k), fill=0, font=font)
        y += c.shape[0] + gap
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _read(png, first, n, model, effort):
    """Codex reads one sheet -> {number: text}; cached by the image."""
    from .script import _codex

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"ocr_{hashlib.sha1(png + PROMPT.encode() + model.encode()).hexdigest()[:16]}.json"
    if path.exists():
        return {int(k): v for k, v in load_json(path).items()}
    img = path.with_suffix(".png")
    img.write_bytes(png)
    try:
        want = set(range(first, first + n))
        for _ in range(2):
            parsed, _tokens = _codex(model, PROMPT, f"Strips {first}-{first + n - 1}.", Strips, effort, [img])
            got = {s.n: s.en.strip() for s in (parsed.items if parsed else []) if s.n in want}
            if set(got) == want:
                save_json(path, got)
                return got
        return got
    finally:
        img.unlink(missing_ok=True)


def read_subtitles(video, out_json, model="codex", effort="low"):
    """[{start, end, en}] of the video's own English subtitles, written to out_json; [] if it has none."""
    if out_json.exists():
        return load_json(out_json)
    frames, h = sample_frames(video)
    band = text_band(frames, h)
    if not band:
        print("      no burned-in subtitles found", flush=True)
        save_json(out_json, [])
        return []
    band = (max(0.0, band[0] - 0.012), min(1.0, band[1] + 0.012))
    found = events(video, band)
    print(f"      {len(found)} subtitles at {band[0]:.0%}-{band[1]:.0%} of the height; reading them", flush=True)
    batches = [(k, found[k:k + BATCH]) for k in range(0, len(found), BATCH)]
    done = [0]

    def one(item):
        k, part = item
        got = _read(_sheet([c for _, _, c in part], k), k, len(part), model, effort)
        done[0] += len(part)
        print(f"      ocr [{done[0]}/{len(found)}]", flush=True)
        return got

    texts = {}
    with ThreadPoolExecutor(WORKERS) as pool:
        for got in pool.map(one, batches):
            texts |= got
    if missing := len(found) - len(texts):  # not saved, so a rerun reads only the missing sheets (the rest are cached)
        raise SystemExit(f"ocr: {missing} of {len(found)} subtitles unread; run again to finish")
    subs = []
    for k, (a, b, _) in enumerate(found):
        text = " ".join(texts.get(k, "").split())
        if not text:
            continue
        if subs and text.lower() == subs[-1]["en"].lower() and a - subs[-1]["end"] <= MERGE_GAP:
            subs[-1]["end"] = round(b, 2)
        else:
            subs.append({"start": round(a, 2), "end": round(b, 2), "en": text})
    save_json(out_json, subs)
    return subs


def attach(lines, subs):
    """Each line gets "on_screen" (subtitles shown while it is said); subtitles no line covers become new lines
    with an empty "en". Returns the lines in time order."""
    used = set()
    for line in lines:
        shown = [k for k, s in enumerate(subs)
                 if s["start"] < line["end"] + NEAR and s["end"] > line["start"] - NEAR]
        if shown:
            line["on_screen"] = " ".join(subs[k]["en"] for k in shown)
            used |= set(shown)
    next_id = max((l.get("id", 0) for l in lines), default=0) + 1
    added = []
    for k, s in enumerate(subs):
        if k in used:
            continue
        added.append({"id": next_id, "start": s["start"], "end": s["end"], "voice": "?", "en": "",
                      "on_screen": s["en"], "from_subtitle": True})
        next_id += 1
    if added:
        print(f"      {len(added)} subtitles with no transcript line: added as lines", flush=True)
    return sorted(lines + added, key=lambda l: l["start"])
