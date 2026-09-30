"""Mongolian subtitles from the dub script: SRT for a switchable track, ASS for burning into the picture,
WebVTT for the studio player. Tags ([warm], [laughs]) never show; sound-only lines get no subtitle."""
from pathlib import Path

from .common import HEADER_SRT, TAG, stamp, to_seconds

MIN_SEC = 1.0  # shortest time a subtitle stays up
GAP = 0.05  # seconds between one subtitle ending and the next starting
READ_CPS = 13  # characters per second, to guess how long a line is said before it has been dubbed
FONT = "Arial"  # ships with Windows and has Cyrillic; any installed font with Cyrillic works


def text_of(line):
    return " ".join(TAG.sub("", line["text"]).split())


def cues(lines, durations=None):
    """[start, end, text] per spoken line. `durations` (seconds of dub audio, same order as `lines`) gives exact
    ends; without it the end is guessed from the text length. A subtitle never runs into the next one."""
    out = []
    for i, line in enumerate(lines):
        text = text_of(line)
        if not text:
            continue
        said = durations[i] if durations else max(line["end"] - line["start"], len(text) / READ_CPS)
        out.append([line["start"], line["start"] + max(said, MIN_SEC), text])
    for a, b in zip(out, out[1:]):
        a[1] = max(a[0] + 0.2, min(a[1], b[0] - GAP))
    return out


def write_srt(cue_list, path):
    Path(path).write_text("\n".join(f"{i}\n{stamp(a)} --> {stamp(b)}\n{t}\n" for i, (a, b, t) in enumerate(cue_list, 1)),
                          encoding="utf-8")


def read_srt(path):
    out = []
    for block in Path(path).read_text(encoding="utf-8").strip().split("\n\n"):
        rows = block.strip().splitlines()
        m = HEADER_SRT.match(rows[1]) if len(rows) > 2 else None
        if m:
            g = m.groups()
            out.append([to_seconds(*g[0:4]), to_seconds(*g[4:8]), " ".join(rows[2:])])
    return out


def vtt(cue_list):
    ts = lambda t: stamp(t).replace(",", ".")
    return "WEBVTT\n\n" + "\n".join(f"{ts(a)} --> {ts(b)}\n{t}\n" for a, b, t in cue_list)


def _ass_time(t):
    cs = round(t * 100)
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def write_ass(cue_list, path, width, height, band=None):
    """Styled for the frame: sized to the short side; on vertical video it sits above the app buttons and
    captions that TikTok / Shorts / Reels draw over the bottom of the picture. With `band` (top, bottom as
    fractions of the height: a bar covering the video's own subtitles) every line is centred on that bar,
    which grows upward if needed to hold three lines. Returns the band used."""
    size = round(min(width, height) * 0.052)
    margin_v = round(height * (0.22 if height > width else 0.07))
    margin_h = round(width * 0.07)
    pos = ""
    if band:
        top, bottom = band
        top = max(0.0, min(top, bottom - 3.6 * size / height))  # room for three lines
        band = (top, bottom)
        pos = f"{{\\an5\\pos({width // 2},{round((top + bottom) / 2 * height)})}}"
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{FONT},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{max(2, size // 14)},{max(1, size // 28)},2,{margin_h},{margin_h},{margin_v},204

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = "".join(f"Dialogue: 0,{_ass_time(a)},{_ass_time(b)},Default,,0,0,0,,"
                     f"{pos}{t.replace('{', '(').replace('}', ')')}\n" for a, b, t in cue_list)
    Path(path).write_text(head + events, encoding="utf-8")
    return band
