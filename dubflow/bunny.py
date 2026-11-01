"""Finished dub -> Bunny Stream: create the video, upload the mp4 (streamed, so a 2 GB film never sits in RAM), add
the Mongolian subtitles as a caption track. The upload is recorded in <work>/bunny.json; the same file is never
uploaded twice, a rebuilt mp4 becomes a new video (the old one stays in the library until you delete it).

.env: BUNNY_LIBRARY_ID, BUNNY_STREAM_API_KEY; optional BUNNY_CDN_HOSTNAME (HLS link) and BUNNY_TOKEN_AUTH_KEY
(signed embed link, for libraries with token authentication on).
"""
import base64
import hashlib
import os
import time

from .common import load_json, save_json

API = "https://video.bunnycdn.com/library"
CHUNK = 8 * 1024 * 1024  # bytes per read while streaming the upload
EMBED_DAYS = 7  # signed embed link lifetime
STATUS = {0: "created", 1: "uploaded", 2: "processing", 3: "transcoding", 4: "finished", 5: "error",
          6: "upload failed"}


def configured():
    return bool(os.getenv("BUNNY_LIBRARY_ID") and os.getenv("BUNNY_STREAM_API_KEY"))


def _client():
    import httpx

    return httpx.Client(headers={"AccessKey": os.environ["BUNNY_STREAM_API_KEY"], "accept": "application/json"},
                        timeout=httpx.Timeout(60, write=None, read=600))


def _stream(path, size):
    sent, last = 0, 0.0
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            sent += len(chunk)
            if time.time() - last > 10 or sent == size:
                print(f"      uploaded {sent / 2**20:,.0f} / {size / 2**20:,.0f} MB", flush=True)
                last = time.time()
            yield chunk


def links(guid):
    lib = os.environ["BUNNY_LIBRARY_ID"]
    out = {"embed": f"https://iframe.mediadelivery.net/embed/{lib}/{guid}"}
    if os.getenv("BUNNY_TOKEN_AUTH_KEY"):
        expires = int(time.time()) + EMBED_DAYS * 86400
        token = hashlib.sha256(f"{os.environ['BUNNY_TOKEN_AUTH_KEY']}{guid}{expires}".encode()).hexdigest()
        out["embed_signed"] = f"{out['embed']}?token={token}&expires={expires}"
    if os.getenv("BUNNY_CDN_HOSTNAME"):
        out["hls"] = f"https://{os.environ['BUNNY_CDN_HOSTNAME']}/{guid}/playlist.m3u8"
    return out


def delete(guid):
    """Remove a video from the library (it can't be brought back)."""
    with _client() as c:
        r = c.delete(f"{API}/{os.environ['BUNNY_LIBRARY_ID']}/videos/{guid}")
        r.raise_for_status()
    print(f"      deleted Bunny video {guid}")


def upload(video, title, record, subs=None):
    """Upload video (+ subs .srt) unless record already holds this exact file. Returns the record dict."""
    stat = video.stat()
    same = {"file": video.name, "size": stat.st_size, "mtime": int(stat.st_mtime)}
    if record.exists():
        old = load_json(record)
        if all(old.get(k) == v for k, v in same.items()):
            print(f"      already on Bunny: {old['guid']}")
            return old
    lib = os.environ["BUNNY_LIBRARY_ID"]
    with _client() as c:
        r = c.post(f"{API}/{lib}/videos", json={"title": title})
        r.raise_for_status()
        guid = r.json()["guid"]
        print(f"      created Bunny video {guid} \"{title}\"", flush=True)
        r = c.put(f"{API}/{lib}/videos/{guid}", content=_stream(video, stat.st_size),
                  headers={"Content-Type": "application/octet-stream", "Content-Length": str(stat.st_size)})
        r.raise_for_status()
        if subs and subs.exists():
            r = c.post(f"{API}/{lib}/videos/{guid}/captions/mn",
                       json={"srclang": "mn", "label": "Монгол",
                             "captionsFile": base64.b64encode(subs.read_bytes()).decode()})
            if r.is_error:  # the video is up; a missing caption track isn't worth failing the run
                print(f"      captions not added ({r.status_code}): {r.text[:200]}")
        status = c.get(f"{API}/{lib}/videos/{guid}").json().get("status")
    out = {**same, "guid": guid, "title": title, "uploaded": time.strftime("%Y-%m-%d %H:%M:%S"), **links(guid)}
    save_json(record, out)
    print(f"      on Bunny ({STATUS.get(status, status)}; transcoding takes a few minutes): {out['embed']}")
    return out
