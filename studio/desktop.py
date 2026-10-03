"""Dub Studio as a desktop window.

    pythonw -m studio.desktop       runs the studio server in the background and opens it in its own window;
                                    closing the window stops the server (and any running jobs)

make_shortcut.ps1 puts a "Dub Studio" icon on the desktop that runs this.
"""
import os
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # dub_video.py and dubflow/ live in the repo root
os.chdir(ROOT)                 # .env, films/, voices/ and data/ are found relative to the repo

if sys.stdout is None or sys.stderr is None:  # pythonw has no console: keep the server's output in a log file
    log = ROOT / os.getenv("DUB_DATA", "data") / "desktop.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(log, "a", encoding="utf-8", buffering=1)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(port, server, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not server.started:
        time.sleep(0.1)
    if not server.started:
        sys.exit(f"Dub Studio server did not start on port {port}")


def main():
    import uvicorn
    import webview

    from dubflow import common  # noqa: F401  (loads .env)

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config("studio.app:app", host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="dub-server").start()
    _wait(port, server)

    webview.settings["ALLOW_DOWNLOADS"] = True  # the Download buttons save the finished video
    webview.create_window("Dub Studio", f"http://127.0.0.1:{port}/", width=1400, height=900, min_size=(900, 600))
    webview.start()

    server.should_exit = True
    os._exit(0)  # a running job shows as stopped next time, with a Resume button


if __name__ == "__main__":
    main()
