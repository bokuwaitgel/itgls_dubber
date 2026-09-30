"""Web UI + JSON API over the dub pipeline. Run it with `python -m studio` (see README.md)."""
import hashlib
import hmac
import html
import json
import os
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from dubflow.common import ROOT, clip as cut_clip, duration, load_json, save_json

from . import jobs

STATIC = Path(__file__).parent / "static"
FILMS = ROOT / "films"
KEYS = ["ELEVENLABS_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "HF_TOKEN"]
COOKIE = "dub_session"
SESSION_DAYS = 30
PUBLIC = ("/login", "/static/")  # reachable without signing in

LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Sign in: Dub Studio</title>
<link href="https://fonts.googleapis.com/css2?family=Golos+Text:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/style.css"></head>
<body><main class="login"><form method="post" action="/login" class="panel">
<h1>Dub Studio</h1><p class="muted">Enter the studio password (DUB_PASSWORD in .env).</p>
{error}<label class="field">Password<input type="password" name="password" autocomplete="current-password" required autofocus></label>
<button type="submit">Sign in</button></form></main></body></html>"""


def _password():
    return os.getenv("DUB_PASSWORD") or ""


def _token(password):
    """Session value derived from the password: changing DUB_PASSWORD signs everyone out."""
    return hmac.new(password.encode(), b"dub-studio-session", hashlib.sha256).hexdigest()


@asynccontextmanager
async def lifespan(_app):
    jobs.start()
    yield


app = FastAPI(title="Dub Studio", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def require_login(request: Request, call_next):
    """With DUB_PASSWORD set (required beyond localhost), everything but the login page needs a session cookie."""
    password = _password()
    path = request.url.path
    if password and not path.startswith(PUBLIC):
        if not hmac.compare_digest(request.cookies.get(COOKIE, ""), _token(password)):
            if path.startswith("/api/"):
                return JSONResponse({"detail": "Sign in first."}, status_code=401)
            return RedirectResponse("/login", status_code=303)
    return await call_next(request)


@app.get("/login", response_class=HTMLResponse)
def login_page():
    if not _password():
        return RedirectResponse("/", status_code=303)
    return LOGIN_PAGE.format(error="")


@app.post("/login")
def login(password: str = Form("")):
    wanted = _password()
    if not wanted or hmac.compare_digest(password.encode(), wanted.encode()):
        response = RedirectResponse("/", status_code=303)
        if wanted:
            response.set_cookie(COOKIE, _token(wanted), max_age=SESSION_DAYS * 86400, httponly=True, samesite="lax")
        return response
    time.sleep(1)  # slows down password guessing
    error = f'<div class="notice error">{html.escape("That password is wrong. Check DUB_PASSWORD in .env.")}</div>'
    return HTMLResponse(LOGIN_PAGE.format(error=error), status_code=401)


@app.get("/logout")
def logout():
    response = RedirectResponse("/login" if _password() else "/", status_code=303)
    response.delete_cookie(COOKIE)
    return response


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
    return {"defaults": jobs.defaults(), "has_logo": jobs.logo_path() is not None, "distances": jobs.DISTANCES, "efforts": jobs.EFFORTS, "login": bool(_password()),
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
    return [{k: j.get(k) for k in ("id", "title", "status", "phase", "stage", "note", "created", "updated")}
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
    """Rebuild the lines from the transcript and re-run the translation (e.g. after editing the film profile).
    The current script is backed up first. Speaker diarization is reused."""
    job = _job(jid)
    _idle(job)
    backup = jobs.backup_script(job)
    return {**detail(jobs.enqueue(jid, "prepare", ["analyze", "script"])), "backup": backup}


@app.get("/api/jobs/{jid}/film")
def get_film(jid: str):
    return load_json(jobs.paths(_job(jid))["film"])


@app.put("/api/jobs/{jid}/film")
def put_film(jid: str, body: dict = Body(...)):
    job = _job(jid)
    _idle(job)
    save_json(jobs.paths(job)["film"], body)
    return body


@app.put("/api/jobs/{jid}/options")
def put_options(jid: str, body: dict = Body(...)):
    """Change settings used by the next run (e.g. subtitles before rebuilding the video)."""
    job = _job(jid)
    _idle(job)
    try:
        options = jobs.clean_options({**job["options"], **body})
    except ValueError as e:
        raise _bad(e)
    return detail(jobs.update(jid, options=options))


@app.get("/api/jobs/{jid}/subtitles.vtt")
def subtitles_vtt(jid: str):
    job = _job(jid)
    if not jobs.paths(job)["script"].exists():
        raise HTTPException(404, "The script isn't written yet.")
    return Response(jobs.subtitles_vtt(job), media_type="text/vtt; charset=utf-8", headers={"Cache-Control": "no-store"})


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


_clip_lock = threading.Lock()


@app.get("/api/jobs/{jid}/clip")
def clip(jid: str, start: float = 0, end: float = 600):
    """Download a piece of the Mongolian video (default: the first 10 minutes). Cut once, then reused until the
    video is rebuilt."""
    job = _job(jid)
    final = jobs.final_video(job)
    if not final:
        raise HTTPException(404, "There is no Mongolian video yet. Dub the film first.")
    end = min(end, duration(final))
    if start < 0 or end - start < 1:
        raise HTTPException(400, "The clip must start at 0:00 or later and end at least a second after it starts.")
    out = jobs.paths(job)["work"] / "clips" / f"clip_{round(start * 10)}-{round(end * 10)}.mp4"
    with _clip_lock:
        if not out.exists() or out.stat().st_mtime < final.stat().st_mtime:
            out.parent.mkdir(exist_ok=True)
            cut_clip(final, out, start, end)
    stamp_ = lambda t: f"{int(t // 60)}-{int(t % 60):02d}"
    return FileResponse(out, media_type="video/mp4",
                        filename=f"{job['title']} (Mongolian) {stamp_(start)} to {stamp_(end)}.mp4")


@app.get("/api/logo")
def get_logo():
    path = jobs.logo_path()
    if not path:
        raise HTTPException(404, "No watermark image yet. Upload one.")
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@app.put("/api/logo")
def put_logo(image: UploadFile = File(...)):
    """New watermark image for every job from now on (PNG with a transparent background works best)."""
    data = image.file.read(20 * 1024 * 1024 + 1)
    if len(data) > 20 * 1024 * 1024 or not data.startswith(b"\x89PNG"):
        raise HTTPException(400, "Upload a PNG image under 20 MB (a transparent background looks best).")
    jobs.LOGO_UPLOAD.parent.mkdir(parents=True, exist_ok=True)
    jobs.LOGO_UPLOAD.write_bytes(data)
    return {"saved": True}


@app.get("/api/jobs/{jid}/frame")
def frame(jid: str, at: float = 10.0):
    job = _job(jid)
    try:
        return Response(jobs.frame_jpeg(job, at), media_type="image/jpeg")
    except Exception:
        raise HTTPException(404, "Couldn't read a frame at that time.")


@app.get("/api/jobs/{jid}/logo-spot")
def logo_spot(jid: str):
    return {"box": jobs.logo_spot(_job(jid))}


@app.get("/api/jobs/{jid}/video/{kind}")
def video(jid: str, kind: str, download: bool = False):
    job = _job(jid)
    path = {"source": jobs.source_video, "final": jobs.final_video}.get(kind, lambda _: None)(job)
    if not path or not Path(path).exists():
        raise HTTPException(404, "That video doesn't exist yet.")
    name = f"{job['title']}{' (Mongolian)' if kind == 'final' else ''}{Path(path).suffix}"
    return FileResponse(path, media_type="video/mp4", filename=name if download else None,
                        content_disposition_type="attachment" if download else "inline")
