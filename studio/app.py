"""Web UI + JSON API over the dub pipeline. Run it with `python -m studio` (see README.md)."""
import base64
import hmac
import json
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from dubflow.common import ROOT, load_json, save_json

from . import jobs

STATIC = Path(__file__).parent / "static"
FILMS = ROOT / "films"
KEYS = ["ELEVENLABS_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "HF_TOKEN"]


@asynccontextmanager
async def lifespan(_app):
    jobs.start()
    yield


app = FastAPI(title="Dub Studio", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    """HTTP Basic auth when DUB_PASSWORD is set (required when the server listens beyond localhost)."""
    password = os.getenv("DUB_PASSWORD")
    if password:
        user, given = "", ""
        header = request.headers.get("authorization", "")
        if header.startswith("Basic "):
            try:
                user, _, given = base64.b64decode(header[6:]).decode().partition(":")
            except ValueError:
                pass
        if not (hmac.compare_digest(given.encode(), password.encode())
                and hmac.compare_digest(user.encode(), os.getenv("DUB_USER", "dub").encode())):
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Dub Studio"'})
    return await call_next(request)


def _job(jid):
    try:
        return jobs.load(jid)
    except FileNotFoundError:
        raise HTTPException(404, "No such job. It may have been deleted.")


def _idle(job):
    if job["status"] in jobs.ACTIVE:
        raise HTTPException(409, "This job is running. Stop it or wait until it finishes.")


def _bad(e):
    return HTTPException(400, str(e))


def detail(job):
    final = jobs.final_video(job)
    out = {**job, "log": jobs.log_tail(job), "has_final": final is not None,
           "has_script": jobs.paths(job)["script"].exists(), "estimate": None}
    if job["status"] not in jobs.ACTIVE and out["has_script"]:
        out["estimate"] = jobs.estimate(job)
    return out


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config():
    return {"defaults": jobs.defaults(), "distances": jobs.DISTANCES,
            "missing_keys": [k for k in KEYS if not os.getenv(k)],
            "films": sorted(p.stem for p in FILMS.glob("*.json"))}


@app.get("/api/films/{name}")
def film(name: str):
    path = FILMS / f"{name}.json"
    if not name.replace("_", "").replace("-", "").isalnum() or not path.exists():
        raise HTTPException(404, "No such film profile.")
    return load_json(path)


@app.get("/api/jobs")
def list_jobs():
    return [{k: j.get(k) for k in ("id", "title", "status", "phase", "stage", "created", "updated")}
            for j in jobs.all_jobs()]


@app.post("/api/jobs")
def create_job(video: UploadFile = File(...), title: str = Form(""), film: str = Form("{}"),
               options: str = Form("{}")):
    try:
        film_data, opts = json.loads(film or "{}"), jobs.clean_options(json.loads(options or "{}"))
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Film profile isn't valid JSON: {e.msg} (line {e.lineno}).")
    except ValueError as e:
        raise _bad(e)
    if not isinstance(film_data, dict):
        raise HTTPException(400, "Film profile must be a JSON object.")
    upload = Path(video.filename or "")
    if upload.suffix.lower() not in jobs.VIDEO_EXT:
        raise HTTPException(400, f"Upload a video file ({', '.join(sorted(jobs.VIDEO_EXT))}).")
    title = title.strip() or film_data.get("title") or upload.stem
    film_data.pop("title", None)
    job = jobs.create(title, film_data, opts, upload.suffix.lower())
    with open(job["video"], "wb") as f:
        shutil.copyfileobj(video.file, f, 1 << 20)
    return detail(jobs.enqueue(job["id"], "prepare"))


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    return detail(_job(jid))


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    _job(jid)
    try:
        jobs.delete(jid)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"deleted": jid}


@app.post("/api/jobs/{jid}/stop")
def stop_job(jid: str):
    _job(jid)
    return detail(jobs.stop(jid))


@app.post("/api/jobs/{jid}/resume")
def resume_job(jid: str):
    job = _job(jid)
    if job["status"] not in ("failed", "stopped"):
        raise HTTPException(409, "Only a failed or stopped job can be resumed.")
    return detail(jobs.enqueue(jid, job["phase"], job.get("redo", [])))


@app.post("/api/jobs/{jid}/dub")
def dub_job(jid: str, body: dict = Body(...)):
    """Spends ElevenLabs credits: the client must echo the credit estimate it showed the user."""
    job = _job(jid)
    _idle(job)
    if not jobs.paths(job)["script"].exists():
        raise HTTPException(409, "There is no script yet.")
    shown, now = body.get("credits"), (jobs.estimate(job) or {}).get("credits")
    if shown != now:
        raise HTTPException(409, f"The estimate changed to {now} credits. Check it and confirm again.")
    return detail(jobs.enqueue(jid, "dub"))


@app.post("/api/jobs/{jid}/retranslate")
def retranslate(jid: str):
    """Re-run the translation (e.g. after editing the film profile). The current script is backed up first."""
    job = _job(jid)
    _idle(job)
    backup = jobs.backup_script(job)
    return {**detail(jobs.enqueue(jid, "prepare", ["script"])), "backup": backup}


@app.get("/api/jobs/{jid}/film")
def get_film(jid: str):
    return load_json(jobs.paths(_job(jid))["film"])


@app.put("/api/jobs/{jid}/film")
def put_film(jid: str, body: dict = Body(...)):
    job = _job(jid)
    _idle(job)
    save_json(jobs.paths(job)["film"], body)
    return body


@app.get("/api/jobs/{jid}/script")
def get_script(jid: str):
    job = _job(jid)
    if not jobs.paths(job)["script"].exists():
        raise HTTPException(404, "The script isn't written yet.")
    return jobs.script_view(job)


@app.put("/api/jobs/{jid}/script")
def put_script(jid: str, body: dict = Body(...)):
    job = _job(jid)
    _idle(job)
    try:
        saved = jobs.save_script(job, body.get("edits", []))
    except (KeyError, ValueError) as e:
        raise _bad(e)
    return {"saved": saved, "estimate": jobs.estimate(job)}


@app.put("/api/jobs/{jid}/voices")
def put_voices(jid: str, body: dict = Body(...)):
    job = _job(jid)
    _idle(job)
    try:
        voices = jobs.save_voices(job, body)
    except (TypeError, ValueError) as e:
        raise _bad(e)
    return {"voices": voices, "estimate": jobs.estimate(job)}


@app.get("/api/jobs/{jid}/video/{kind}")
def video(jid: str, kind: str, download: bool = False):
    job = _job(jid)
    path = {"source": jobs.source_video, "final": jobs.final_video}.get(kind, lambda _: None)(job)
    if not path or not Path(path).exists():
        raise HTTPException(404, "That video doesn't exist yet.")
    name = f"{job['title']}{' (Mongolian)' if kind == 'final' else ''}{Path(path).suffix}"
    return FileResponse(path, media_type="video/mp4", filename=name if download else None,
                        content_disposition_type="attachment" if download else "inline")
