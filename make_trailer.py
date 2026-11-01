"""Trailer from the opening of a finished dub: the script model picks the strongest scenes of the first minutes
(whole lines, in film order, opening on a hook and ending on a cliffhanger), and they are cut from the final video
(logo and subtitles included) and joined with short audio fades.

    python make_trailer.py work/<film>                      # 3-5 min from the first 10 min
    python make_trailer.py work/<film> --within 600 --min 180 --max 300 --out trailer.mp4
"""
import argparse
import json
import subprocess
from pathlib import Path

from pydantic import BaseModel

import dubflow.common  # noqa: F401  (loads .env)
from dubflow.common import FFMPEG, TAG, h264_encoder, read_script
from dubflow.media import _bit_rate
from dubflow.script import MODEL, _ask

PAD_IN, PAD_OUT = 0.3, 0.6  # seconds kept before a segment's first line and after its last
FADE = 0.12  # audio fade at each cut, so joins don't click

PROMPT = """You cut a trailer for a Mongolian-dubbed short drama from the numbered lines of its opening minutes
(time, speaker, Mongolian line). Pick segments, each a run of consecutive lines (first..last, inclusive), that
together run {lo}-{hi} seconds. Keep film order. Open with a strong hook (danger, shock, betrayal, a bold line),
keep the story understandable (who the hero is, what went wrong, what power or chance they get), and end on a
cliffhanger line that makes people want the full film. Skip dull small talk and lines that only make sense with
what was cut. Segments must not overlap; each should be at least 6 seconds."""


class Segment(BaseModel):
    first: int
    last: int
    why: str


class Segments(BaseModel):
    items: list[Segment]


def _spans(lines, segs, end_limit):
    """Line ranges -> (start, end) seconds, padded, clamped so a segment never reaches into the next kept line."""
    out = []
    for s in sorted(segs, key=lambda s: s.first):
        a, b = max(0, s.first), min(len(lines) - 1, s.last)
        if a > b or (out and a <= out[-1][2]):
            continue
        start = max(0.0, lines[a]["start"] - PAD_IN, lines[a - 1]["end"] if a else 0.0)
        end = min(lines[b]["end"] + PAD_OUT, lines[b + 1]["start"] if b + 1 < len(lines) else end_limit, end_limit)
        out.append((start, end, b))
    return [(a, b) for a, b, _ in out]


def pick(lines, lo, hi, within):
    listing = "\n".join(f"{i}. [{l['start']:.1f}-{l['end']:.1f}s] {l['speaker']}: {TAG.sub('', l['text']).strip()}"
                        for i, l in enumerate(lines))
    note = ""
    for _ in range(3):
        parsed, _tokens = _ask(None, MODEL, PROMPT.format(lo=lo, hi=hi), listing + note, None, Segments, "medium")
        spans = _spans(lines, parsed.items if parsed else [], within)
        total = sum(b - a for a, b in spans)
        if lo <= total <= hi:
            return spans
        note = (f"\n\nYour last pick ran {total:.0f} s, outside {lo}-{hi} s. "
                f"{'Add' if total < lo else 'Drop'} segments and try again.")
        print(f"      picked {total:.0f} s; asking again", flush=True)
    raise SystemExit(f"no trailer of {lo}-{hi} s after 3 tries (last: {total:.0f} s)")


def cut(video, spans, out):
    parts, labels = [], ""
    for k, (a, b) in enumerate(spans):
        d = b - a
        parts.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{k}];"
                     f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:d={FADE},afade=t=out:st={d - FADE:.3f}:d={FADE}[a{k}]")
        labels += f"[v{k}][a{k}]"
    graph = ";".join(parts) + f";{labels}concat=n={len(spans)}:v=1:a=1[v][a]"
    graph_file = out.with_suffix(".filters.txt")
    graph_file.write_text(graph, encoding="utf-8")
    codec = h264_encoder()
    if rate := _bit_rate(video):
        codec += ["-maxrate", str(int(rate * 1.25)), "-bufsize", str(int(rate * 2.5))]
    part = out.with_name(out.stem + ".part" + out.suffix)
    subprocess.run([FFMPEG, "-v", "error", "-y", "-t", f"{spans[-1][1] + 1:.3f}", "-i", str(video),
                    "-/filter_complex", str(graph_file), "-map", "[v]", "-map", "[a]", *codec,
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(part)], check=True)
    part.replace(out)
    graph_file.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work", type=Path, help="the film's work dir (script_mn.txt and <name>_mn.mp4)")
    ap.add_argument("--within", type=float, default=600, help="only the first this many seconds (default 600)")
    ap.add_argument("--min", type=float, default=180, help="shortest trailer, seconds (default 180)")
    ap.add_argument("--max", type=float, default=300, help="longest trailer, seconds (default 300)")
    ap.add_argument("--out", type=Path, help="default: <work>/<name>_trailer.mp4")
    args = ap.parse_args()
    video = args.work / f"{args.work.name}_mn.mp4"
    lines = [l for l in read_script(args.work / "script_mn.txt") if l["end"] <= args.within]
    spans = pick(lines, args.min, args.max, args.within)
    out = args.out or args.work / f"{args.work.name}_trailer.mp4"
    print(f"      {len(spans)} scenes, {sum(b - a for a, b in spans):.0f} s: "
          + ", ".join(f"{a:.0f}-{b:.0f}" for a, b in spans), flush=True)
    (out.with_suffix(".json")).write_text(json.dumps(spans), encoding="utf-8")
    cut(video, spans, out)
    print(f"      done: {out}")


if __name__ == "__main__":
    main()
