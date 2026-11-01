"""Clone Mongolian voices from recordings you have permission to use, for the casting pool.

    python clone_voices.py extract a.mp3 b.mp3 ...   # -> voices/clone/S01.mp3 ... + speakers.json
    python clone_voices.py upload                    # clones the samples you gave a "name"

extract separates the voices from music (Demucs), finds who speaks when (pyannote), keeps each person's clean
stretches, matches the same person across files and writes one sample per person. Listen to the samples, put a
name on the ones to keep in voices/clone/speakers.json (fix "gender" / "pool_age" if they're wrong), then upload:
each becomes an ElevenLabs Instant Voice Clone and is added to voices/pool.json, so casting can pick it.
Intermediate files go to work/clone/<file>/ and are reused on the next run.
"""
import argparse
from pathlib import Path

from dubflow.clone import extract_samples, upload


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="recordings -> one sample per speaker")
    ex.add_argument("audio", nargs="+", type=Path)
    ex.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    up = sub.add_parser("upload", help="clone the named samples and add them to voices/pool.json")
    up.add_argument("--remove-noise", action="store_true", help="let ElevenLabs clean the samples too")
    args = ap.parse_args()

    if args.cmd == "extract":
        import torch

        missing = [p for p in args.audio if not p.exists()]
        if missing:
            raise SystemExit(f"not found: {', '.join(map(str, missing))}")
        extract_samples([p.resolve() for p in args.audio], args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print("\nListen to voices/clone/S*.mp3, put a \"name\" on the ones to keep in voices/clone/speakers.json, "
              "then: python clone_voices.py upload")
    else:
        upload(args.remove_noise)


if __name__ == "__main__":
    main()
