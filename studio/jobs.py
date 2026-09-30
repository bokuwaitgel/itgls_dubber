"""Dub jobs: one folder per job under data/jobs/<id>, run one at a time (one GPU) by a worker thread.

A job goes: queued -> running "prepare" (extract .. cast, no TTS) -> review (script and voices editable in the
UI) -> queued -> running "dub" (TTS + mix + mux, spends ElevenLabs credits) -> done. The pipeline itself is
dub_video.py, run as a subprocess so a crash or a stop never takes the server down and GPU memory is freed.
"""
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

DATA = Path(os.getenv("DUB_DATA", ROOT / "data")) / "jobs"
TTS_CACHE = ROOT / "cache" / "tts"
POOL = ROOT / "voices" / "pool.json"
LOG_TAIL = 60  # log lines sent with a job
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
ACTIVE = ("queued", "running", "stopping")

# Options the UI may set -> (dub_video.py flag, type). Defaults come from dub_video.py's own parser.
OPTIONS = {"start": ("--start", float), "end": ("--end", float), "model": ("--model", str),
           "polish_model": ("--polish-model", str), "tts_model": ("--tts-model", str),
           "stability": ("--stability", float), "distance": ("--distance", str), "dub_gain": ("--dub-gain", float),
           "num_speakers": ("--num-speakers", int), "chunk": ("--chunk", int), "no_separate": ("--no-separate", bool)}
DISTANCES = ("close", "medium", "far")

_lock = threading.RLock()
_queue = queue.Queue()
_procs = {}  # job id -> running Popen


def defaults():
    parser = build_parser()
    return {k: parser.get_default(flag.lstrip("-").replace("-", "_")) for k, (flag, _) in OPTIONS.items()}


def clean_options(raw):
    """Whitelisted, typed options; empty values dropped. Raises ValueError with a readable message."""
    out = {}
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
    work = Path(job["work"])
    return {"work": work, "video": Path(job["video"]), "film": job_dir(job["id"]) / "film.json",
            "script": work / "script_mn.txt", "script_json": work / "script.json", "voices": work / "voices.json",
            "log": job_dir(job["id"]) / "run.log"}


def source_video(job):
    """What the script's timings refer to: the excerpt when the job is a test clip, else the input video."""
    excerpt = Path(job["work"]) / "source.mp4"
    return excerpt if excerpt.exists() else Path(job["video"])


def final_video(job):
    return next((p for p in sorted(Path(job["work"]).glob("*_mn.mp4"))), None)


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
        flag, kind = OPTIONS[key]
        cmd += [flag] if kind is bool else [flag, str(value)]
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
    update(jid, status="running", stage=None, error=None)
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
                update(jid, stage=line[3:].split(":")[0].strip())
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
    threading.Thread(target=_worker, daemon=True, name="dub-worker").start()


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


def backup_script(job):
    """Copy of script_mn.txt before a translation re-run overwrites it."""
    p = paths(job)
    if p["script"].exists():
        dest = p["script"].with_name(f"script_mn.{time.strftime('%Y%m%d-%H%M%S')}.bak.txt")
        shutil.copy2(p["script"], dest)
        return dest.name
    return None
