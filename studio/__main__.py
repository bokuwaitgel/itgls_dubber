"""Dub Studio server.

    python -m studio                          http://127.0.0.1:8000
    python -m studio --host 0.0.0.0           on your network (needs DUB_PASSWORD in .env)
    python -m studio import --video film.mp4 --work work/film --film films/film.json
                                              add a film dubbed earlier with dub_video.py to the studio
"""
import argparse
import ipaddress
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # dub_video.py and dubflow/ live in the repo root


def _local(host):
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def serve(args):
    import uvicorn

    from dubflow import common  # noqa: F401  (loads .env)

    if not _local(args.host) and not os.getenv("DUB_PASSWORD"):
        sys.exit("Refusing to listen on the network without a password: set DUB_PASSWORD in .env "
                 "(anyone who can open the page can spend your API credits).")
    print(f"Dub Studio: http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}")
    uvicorn.run("studio.app:app", host=args.host, port=args.port, log_level="warning")


def import_job(args):
    from dubflow.common import load_json

    from . import jobs

    for p in (args.video, args.work):
        if not p.exists():
            sys.exit(f"no such path: {p}")
    film = load_json(args.film) if args.film else {}
    title = args.title or film.pop("title", None) or args.video.stem
    film.pop("title", None)
    job = jobs.create(title, film, {}, video=args.video.resolve(), work=args.work.resolve())
    status = "done" if jobs.final_video(job) else "review" if jobs.paths(job)["script"].exists() else "stopped"
    jobs.update(job["id"], status=status, phase="dub" if status == "done" else "prepare")
    print(f"imported as {job['id']} ({status})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.getenv("DUB_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("DUB_PORT", "8000")))
    sub = ap.add_subparsers(dest="command")
    imp = sub.add_parser("import", help="add an existing dub_video.py work dir as a job")
    imp.add_argument("--video", type=Path, required=True)
    imp.add_argument("--work", type=Path, required=True)
    imp.add_argument("--film", type=Path)
    imp.add_argument("--title")
    args = ap.parse_args()
    import_job(args) if args.command == "import" else serve(args)


if __name__ == "__main__":
    main()
