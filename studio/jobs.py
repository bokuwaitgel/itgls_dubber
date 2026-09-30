"""Dub jobs: one folder per job under data/jobs/<id>, run one at a time (one GPU) by a worker thread.

A job goes: queued -> running "prepare" (extract .. cast, no TTS) -> review (script and voices editable in the
UI) -> queued -> running "dub" (TTS + mix + mux, spends ElevenLabs credits) -> done. The pipeline itself is
dub_video.py, run as a subprocess so a crash or a stop never takes the server down and GPU memory is freed.
"""
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

from dub_video import build_parser
from dubflow.common import ROOT, load_json, read_script, save_json, stamp, write_script

DATA = (ROOT / os.getenv("DUB_DATA", "data")).resolve() / "jobs"  # a relative DUB_DATA is inside the project
TTS_CACHE = ROOT / "cache" / "tts"
POOL = ROOT / "voices" / "pool.json"
LOG_TAIL = 60  # log lines sent with a job
WORKERS = max(1, int(os.getenv("DUB_WORKERS", "2")))  # jobs running at once; GPU stages still take turns
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
ACTIVE = ("queued", "running", "stopping")

# Options the UI may set -> (dub_video.py flag, type). Defaults come from dub_video.py's own parser.
OPTIONS = {"start": ("--start", float), "end": ("--end", float), "model": ("--model", str),
           "polish_model": ("--polish-model", str), "tts_model": ("--tts-model", str),
           "stability": ("--stability", float), "distance": ("--distance", str), "dub_gain": ("--dub-gain", float),
           "num_speakers": ("--num-speakers", int), "chunk": ("--chunk", int), "no_separate": ("--no-separate", bool),
           "translate_workers": ("--translate-workers", int), "polish_workers": ("--polish-workers", int),
           "effort": ("--effort", str), "subs": ("--subs", str)}
DISTANCES = ("close", "medium", "far")
EFFORTS = ("low", "medium", "high")
SUBS = ("soft", "burn", "none")

_lock = threading.RLock()
_queue = queue.Queue()
_procs = {}  # job id -> running Popen


def defaults():
    from dubflow.media import WATERMARK
    parser = build_parser()
    out = {k: parser.get_default(flag.lstrip("-").replace("-", "_")) for k, (flag, _) in OPTIONS.items()}
    return {**out, "watermark": {"enabled": False, **WATERMARK}}


LOGO_UPLOAD = DATA.parent / "logo.png"  # set from the studio
DEFAULT_LOGO = ROOT / "watermark-nobackground.png"


def logo_path():
    return next((p for p in (LOGO_UPLOAD, DEFAULT_LOGO) if p.exists()), None)


def _clean_watermark(wm):
    """{"enabled", "auto", "x", "y", "size", "opacity", "box", "cover"} with every value checked."""
    if not isinstance(wm, dict):
        raise ValueError("watermark must be an object")
    base = defaults()["watermark"]
    out = {}
    for key, default in base.items():
        value = wm.get(key, default)
        if isinstance(default, bool):
            out[key] = bool(value)
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"watermark {key} must be a number")
        low, high = {"size": (0.05, 1.0), "opacity": (0.1, 1.0)}.get(key, (0.0, 1.0))
        if not low <= value <= high:
            raise ValueError(f"watermark {key} must be between {low} and {high}")
        out[key] = value
    return out


def clean_options(raw):
    """Whitelisted, typed options; empty values dropped. Raises ValueError with a readable message."""
    out = {}
    if raw.get("watermark"):
        out["watermark"] = _clean_watermark(raw["watermark"])
    for key, value in raw.items():
        if key not in OPTIONS or value is None or value == "" or value is False:  # 0 is a real value
            continue
        kind = OPTIONS[key][1]
        try:
            out[key] = True if kind is bool else kind(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a {kind.__name__}")
    if out.get("distance", "medium") not in DISTANCES:
        raise ValueError(f"distance must be one of {', '.join(DISTANCES)}")
    if out.get("subs", "soft") not in SUBS:
        raise ValueError(f"subs must be one of {', '.join(SUBS)}")
    if out.get("effort", "medium") not in EFFORTS:
        raise ValueError(f"effort must be one of {', '.join(EFFORTS)}")
    for key in ("translate_workers", "polish_workers"):
        if not 1 <= out.get(key, 1) <= 16:
            raise ValueError(f"{key} must be between 1 and 16")
    if "start" in out and "end" in out and out["end"] <= out["start"]:
        raise ValueError("end must be after start")
    return out


# ---- store

def job_dir(jid):
    return DATA / jid


def load(jid):
    if not jid.replace("-", "").isalnum():
        raise FileNotFoundError(jid)
    return load_json(job_dir(jid) / "job.json")


def update(jid, **fields):
    with _lock:
        job = load(jid)
        job.update(fields, updated=time.time())
        save_json(job_dir(jid) / "job.json", job)
        return job


def all_jobs():
    if not DATA.exists():
        return []
    found = [load_json(p) for p in DATA.glob("*/job.json")]
    return sorted(found, key=lambda j: j["created"], reverse=True)


def create(title, film, options, video_suffix=".mp4", video=None, work=None):
    """New job folder. Without `video`/`work` both live inside it (the caller writes the video file)."""
    jid = time.strftime("%y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    folder = job_dir(jid)
    folder.mkdir(parents=True)
    save_json(folder / "film.json", {"title": title, **film})
    job = {"id": jid, "title": title, "created": time.time(), "updated": time.time(),
           "video": str(video or folder / f"input{video_suffix}"), "work": str(work or folder / "work"),
           "options": options, "status": "uploading", "phase": "prepare", "redo": [], "stage": None, "error": None}
    save_json(folder / "job.json", job)
    return job


def delete(jid):
    job = load(jid)
    if job["status"] in ACTIVE:
        raise RuntimeError("stop the job before deleting it")
    shutil.rmtree(job_dir(jid))  # an imported job's external video/work dir stays where it was


def paths(job):
    work = ROOT / job["work"]
    return {"work": work, "video": ROOT / job["video"], "film": job_dir(job["id"]) / "film.json",
            "script": work / "script_mn.txt", "script_json": work / "script.json", "voices": work / "voices.json",
            "log": job_dir(job["id"]) / "run.log"}


def source_video(job):
    """What the script's timings refer to: the excerpt when the job is a test clip, else the input video."""
    excerpt = ROOT / job["work"] / "source.mp4"
    return excerpt if excerpt.exists() else ROOT / job["video"]


def final_video(job):
    return next((p for p in sorted((ROOT / job["work"]).glob("*_mn.mp4"))), None)


def log_tail(job, n=LOG_TAIL):
    log = paths(job)["log"]
    if not log.exists():
        return []
    rows = log.read_text(encoding="utf-8", errors="replace").splitlines()
    return [r for r in rows if "Warning" not in r and "warnings.warn" not in r][-n:]


# ---- runner

def enqueue(jid, phase, redo=()):
    job = update(jid, status="queued", phase=phase, redo=list(redo), error=None)
    _queue.put(jid)
    return job


def stop(jid):
    job = load(jid)
    if job["status"] == "queued":
        return update(jid, status="stopped")
    if job["status"] != "running":
        return job
    job = update(jid, status="stopping")
    proc = _procs.get(jid)
    if proc:
        _kill(proc)
    return job


def command(job):
    p = paths(job)
    cmd = [sys.executable, "-u", str(ROOT / "dub_video.py"), str(p["video"]), "--film", str(p["film"]),
           "--work", str(p["work"])]
    for key, value in job["options"].items():
        if key not in OPTIONS:
            continue
        flag, kind = OPTIONS[key]
        cmd += [flag] if kind is bool else [flag, str(value)]
    wm, logo = job["options"].get("watermark") or {}, logo_path()
    if wm.get("enabled") and logo:
        style = {k: v for k, v in wm.items() if k != "enabled"}
        cmd += ["--watermark", str(logo), "--watermark-style", json.dumps(style)]
    cmd += ["--until", "cast"] if job["phase"] == "prepare" else ["--yes"]
    if job["redo"]:
        cmd += ["--redo", *job["redo"]]
    return cmd


def _group():
    return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}


def _kill(proc):
    """The whole process tree: dub_video.py and the ffmpeg it may be running."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        os.killpg(proc.pid, signal.SIGTERM)


def _run(job):
    jid, log_path = job["id"], paths(job)["log"]
    update(jid, status="running", stage=None, error=None, note=None)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} {job['phase']}\n")
        try:
            proc = subprocess.Popen(command(job), cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                    errors="replace", **_group())
        except OSError as e:
            update(jid, status="failed", error=f"could not start the pipeline: {e}")
            return
        _procs[jid] = proc
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if line.startswith("== "):
                update(jid, stage=line[3:].split(":")[0].strip(), note=None)
            elif "waiting for the GPU" in line:
                update(jid, note="Waiting for the GPU (another job is using it)")
        code = proc.wait()
    _procs.pop(jid, None)
    status = load(jid)["status"]
    if status == "stopping":
        update(jid, status="stopped", error="Stopped. Resume picks up from the last finished stage.")
    elif code:
        last = next((r for r in reversed(log_tail(job)) if r.strip()), f"exit code {code}")
        update(jid, status="failed", error=last.strip())
    else:
        update(jid, status="review" if job["phase"] == "prepare" else "done", redo=[], stage=None)


def _worker():
    while True:
        jid = _queue.get()
        try:
            job = load(jid)
        except FileNotFoundError:
            continue  # deleted while queued
        if job["status"] == "queued":
            try:
                _run(job)
            except Exception as e:  # never let one job kill the worker
                update(jid, status="failed", error=f"{type(e).__name__}: {e}")


def start():
    """Recover jobs left over from a previous server run, then start the worker."""
    DATA.mkdir(parents=True, exist_ok=True)
    for job in reversed(all_jobs()):  # oldest first, so the queue keeps its order
        if job["status"] in ("running", "stopping"):
            update(job["id"], status="stopped", error="The server restarted while this was running. Resume it.")
        elif job["status"] == "uploading":
            update(job["id"], status="failed", error="The upload didn't finish. Delete this job and upload again.")
        elif job["status"] == "queued":
            _queue.put(job["id"])
    for n in range(WORKERS):
        threading.Thread(target=_worker, daemon=True, name=f"dub-worker-{n}").start()


# ---- script and voices

def _voice_id(entry):
    return entry["voice_id"] if isinstance(entry, dict) else entry


def estimate(job):
    """(characters, lines) ElevenLabs would generate now; a speaker without a voice yet counts in full."""
    from dubflow import dub

    p = paths(job)
    if not p["script"].exists():
        return None
    voices = load_json(p["voices"]) if p["voices"].exists() else {}
    model = job["options"].get("tts_model") or defaults()["tts_model"]
    stability = job["options"].get("stability", defaults()["stability"])
    chars = n = 0
    with _lock:  # dub.MODEL_ID is part of the cache key
        dub.MODEL_ID = model
        for line in dub.spoken_lines(read_script(p["script"])):
            entry = voices.get(line["speaker"])
            if entry is None or not dub._cache_path(TTS_CACHE, line["text"], _voice_id(entry), stability).exists():
                chars, n = chars + len(line["text"]), n + 1
    return {"credits": chars, "lines": n}


def script_view(job):
    p = paths(job)
    lines = read_script(p["script"])
    english = {}
    if p["script_json"].exists():
        english = {stamp(l["start"]): l.get("en", "") for l in load_json(p["script_json"])}
    counts = Counter(l["speaker"] for l in lines)
    return {"lines": [{"i": i, "start": l["start"], "end": l["end"], "speaker": l["speaker"], "text": l["text"],
                       "en": english.get(stamp(l["start"]), "")} for i, l in enumerate(lines)],
            "speakers": [{"name": s, "lines": c} for s, c in counts.most_common()],
            "voices": load_json(p["voices"]) if p["voices"].exists() else {},
            "pool": load_json(POOL) if POOL.exists() else []}


def _clean_speaker(name):
    name = " ".join(str(name).split())
    if not name or any(c in name for c in "[]"):
        raise ValueError(f"speaker name {name!r} can't be empty or contain [ ]")
    return name


def save_script(job, edits):
    """Apply [{i, speaker?, text?}] to script_mn.txt, keeping every line's timing."""
    p = paths(job)
    with _lock:
        lines = read_script(p["script"])
        for e in edits:
            i = int(e["i"])
            if not 0 <= i < len(lines):
                raise ValueError(f"no line {i}")
            if "speaker" in e:
                lines[i]["speaker"] = _clean_speaker(e["speaker"])
            if "text" in e:
                lines[i]["text"] = " ".join(str(e["text"]).split())  # a blank line would split the block
        write_script(p["script"], lines)
    return len(edits)


def save_voices(job, voices):
    """{speaker: voice_id | {voice_id, pitch}} for the speakers given; others keep their voice."""
    p = paths(job)
    known = {v["voice_id"] for v in load_json(POOL)} if POOL.exists() else set()
    with _lock:
        current = load_json(p["voices"]) if p["voices"].exists() else {}
        for speaker, entry in voices.items():
            vid, pitch = (entry.get("voice_id"), int(entry.get("pitch", 0))) if isinstance(entry, dict) else (entry, 0)
            if known and vid not in known:
                raise ValueError(f"unknown voice {vid!r} for {speaker}")
            if not -12 <= pitch <= 12:
                raise ValueError("pitch must be between -12 and 12 semitones")
            current[_clean_speaker(speaker)] = {"voice_id": vid, "pitch": pitch} if pitch else vid
        save_json(p["voices"], current)
    return current


def subtitles_vtt(job):
    """Mongolian captions for the studio player, straight from the script (so edits show at once)."""
    from dubflow import subtitles
    return subtitles.vtt(subtitles.cues(read_script(paths(job)["script"])))


def frame_jpeg(job, at):
    """One frame of the source video as JPEG, for placing the watermark."""
    from dubflow.common import FFMPEG
    video = source_video(job)
    return subprocess.run([FFMPEG, "-v", "error", "-ss", f"{max(0.0, at):.2f}", "-i", str(video), "-frames:v", "1",
                           "-vf", "scale=540:-2", "-q:v", "4", "-f", "mjpeg", "-"], capture_output=True, check=True).stdout


def logo_spot(job):
    """Where the video's own channel logo is (fractions x0, y0, x1, y1), or None. Worked out once per job."""
    from dubflow import media
    cache = paths(job)["work"] / "logo_spot.json"
    if cache.exists():
        return load_json(cache)["box"]
    frames, h = media.sample_frames(source_video(job))
    box = media.logo_box(frames, h)
    box = [float(v) for v in box] if box else None
    cache.parent.mkdir(parents=True, exist_ok=True)
    save_json(cache, {"box": box})
    return box


def backup_script(job):
    """Copy of script_mn.txt before a translation re-run overwrites it."""
    p = paths(job)
    if p["script"].exists():
        dest = p["script"].with_name(f"script_mn.{time.strftime('%Y%m%d-%H%M%S')}.bak.txt")
        shutil.copy2(p["script"], dest)
        return dest.name
    return None
