"""Video -> Mongolian dubbed video, one command.

    python dub_video.py "movie.mp4" --film films/village.json

Stages (each saves its output in the work dir and is skipped when that output exists):
    extract     audio out of the video                                   original.wav
    separate    Demucs: voices vs music/effects                          vocals.wav, background.wav
    transcribe  ElevenLabs Scribe v2, English words with timings         transcript_en.json
    analyze     pyannote voices + gender/age/emotion, lines split per voice  lines.json, voices_profile.json
    script      GPT: character names + Mongolian lines with v3 tags      script_mn.txt  <- edit freely
    cast        auto voice per character from voices/pool.json           voices.json    <- edit freely
    dub         ElevenLabs eleven_v4 TTS + mix (asks before spending)    dubbed.wav/.mp3
    mux         dubbed audio into the original video                     <name>_mn.mp4

Useful flags:
    --start 120 --end 300     test on a 3-minute excerpt (cut + re-encoded into the work dir)
    --transcript x.json       reuse an existing Scribe JSON instead of paying for transcription
    --diarization x.json      reuse speaker turns from an earlier run (e.g. village_craft_stall_speakers.json)
    --until cast              stop before any TTS spending, to review script_mn.txt and voices.json
    --redo script cast        re-run stages even though their output exists (script_mn.txt edits are lost!)
    --yes                     don't ask before spending TTS credits

After editing script_mn.txt (text or speaker names) just run the same command again: cast adds voices for
new names, dub only generates lines whose text or voice changed.
"""
import argparse
import re
import sys
import time
from pathlib import Path

from dubflow.common import ANALYSIS_RATE, ROOT, decode, load_json, read_script, save_json, stamp, write_script

STAGES = ["extract", "separate", "transcribe", "analyze", "script", "cast", "dub", "mux"]


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "film"


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--film", type=Path, help="film config JSON: title, genre, premise, names, aliases, notes, guide")
    ap.add_argument("--work", type=Path, help="work dir (default work/<video name>)")
    ap.add_argument("--start", type=float, help="test excerpt start (seconds)")
    ap.add_argument("--end", type=float, help="test excerpt end (seconds)")
    ap.add_argument("--transcript", type=Path, help="existing Scribe JSON for this video (skips paid transcription)")
    ap.add_argument("--diarization", type=Path,
                    help="existing speaker turns (analyze_speakers.py *_speakers.json or diarization.json)")
    ap.add_argument("--voices", type=Path, help="existing voice map to start from (character -> voice)")
    ap.add_argument("--num-speakers", type=int, help="exact number of voices, if known (helps diarization)")
    ap.add_argument("--no-separate", action="store_true", help="skip Demucs; duck the whole original instead")
    ap.add_argument("--model", default="gpt-5.4-mini", help="script model (OpenAI, or gemini-... to also send audio)")
    ap.add_argument("--chunk", type=int, default=40, help="lines per script-model request (lower if it drops lines)")
    ap.add_argument("--polish-model", help="model for the native-editor polish pass (default: --model)")
    ap.add_argument("--tts-model", default="eleven_v4", help="ElevenLabs TTS model id")
    ap.add_argument("--tts-workers", type=int, default=4, help="parallel TTS requests (raise if your plan allows)")
    ap.add_argument("--stability", type=float, default=0.5)
    ap.add_argument("--dub-gain", type=float, default=1.3)
    ap.add_argument("--distance", choices=["close", "medium", "far"], default="medium",
                    help="mic distance feel of the dub voices")
    ap.add_argument("--duck", type=float, help="original voice level under dub lines (default 0 with stems, 0.05 without)")
    ap.add_argument("--tts-cache", type=Path, default=ROOT / "cache" / "tts")
    ap.add_argument("--until", choices=STAGES, default="mux", help="stop after this stage")
    ap.add_argument("--redo", nargs="+", choices=STAGES, default=[], help="re-run these stages")
    ap.add_argument("--yes", action="store_true", help="don't ask before spending TTS credits")
    return ap


def main():
    args = build_parser().parse_args()

    if not args.video.exists():
        sys.exit(f"no such video: {args.video}")
    excerpt = args.start is not None or args.end is not None
    name = slug(args.video.stem) + (f"_{int(args.start or 0)}-{int(args.end or 0)}" if excerpt else "")
    work = args.work or ROOT / "work" / name
    work.mkdir(parents=True, exist_ok=True)
    film = load_json(args.film) if args.film else {"title": args.video.stem}
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    stems = not args.no_separate
    last = STAGES.index(args.until)
    f = {k: work / v for k, v in dict(
        source="source.mp4", original="original.wav", vocals="vocals.wav", background="background.wav",
        transcript="transcript_en.json", diarization="diarization.json", lines="lines.json",
        profile="voices_profile.json", script_json="script.json", script="script_mn.txt", script_en="script_en.txt",
        warnings="gender_warnings.txt", voices="voices.json", cast="cast.json", dubbed="dubbed.wav",
        final=f"{name}_mn.mp4").items()}
    print(f"work dir: {work}  (device: {device})")

    def todo(stage, *outputs):
        if STAGES.index(stage) > last:
            return False
        if stage in args.redo or not all(p.exists() for p in outputs):
            print(f"\n== {stage}", flush=True)
            return True
        print(f"== {stage}: done already ({', '.join(p.name for p in outputs)})")
        return False

    def timed(fn, *a, **kw):
        t = time.time()
        out = fn(*a, **kw)
        print(f"      ({(time.time() - t) / 60:.1f} min)")
        return out

    # extract
    video = f["source"] if excerpt else args.video
    if todo("extract", f["original"], *([f["source"]] if excerpt else [])):
        from dubflow import media
        if excerpt:
            end = args.end if args.end is not None else 10 ** 6
            timed(media.cut, args.video, f["source"], args.start or 0, end)
        timed(media.extract, video, f["original"])
    voice_src = f["vocals"] if stems else f["original"]

    # separate
    if stems and todo("separate", f["vocals"], f["background"]):
        from dubflow import media
        timed(media.separate, f["original"], f["vocals"], f["background"], device)

    # transcribe
    if todo("transcribe", f["transcript"]):
        from dubflow import transcribe
        if args.transcript:
            transcribe.import_transcript(args.transcript, f["transcript"], args.start or 0, args.end)
            print(f"      imported {args.transcript}")
        else:
            timed(transcribe.transcribe, voice_src, work, f["transcript"])

    # analyze
    if todo("analyze", f["lines"], f["profile"]):
        from dubflow import analyze
        audio = decode(voice_src, ANALYSIS_RATE)
        if args.diarization:
            turns = analyze.import_turns(load_json(args.diarization), args.start or 0, args.end)
            save_json(f["diarization"], turns)
            print(f"      imported {len(turns)} turns from {args.diarization}")
        elif f["diarization"].exists() and "analyze" not in args.redo:
            turns = load_json(f["diarization"])
        else:
            turns = timed(analyze.diarize, audio, device, args.num_speakers)
            save_json(f["diarization"], turns)
        lines = analyze.build_lines(load_json(f["transcript"])["words"], turns)
        timed(analyze.add_features, lines, audio, device)
        profile = analyze.profile([l for l in lines if l["voice"] != "SFX"], "voice")
        save_json(f["lines"], lines)
        save_json(f["profile"], profile)
        print(f"      {len(lines)} lines, {len(profile)} voices")
        for v, p in list(profile.items())[:12]:
            print(f"      {v}: {p.get('gender', '?'):<9} {p.get('age_group', '?'):<11} {p['seconds']:7.1f}s")

    # script
    if todo("script", f["script"]):
        from dubflow import script
        audio = voice_src if args.model.startswith("gemini") else None
        lines = timed(script.write_script, load_json(f["lines"]), film, load_json(f["profile"]), args.model, audio,
                      chunk_size=args.chunk, polish_model=args.polish_model)
        save_json(f["script_json"], lines)
        write_script(f["script"], lines, "mn")
        write_script(f["script_en"], lines, "en")

    if last < STAGES.index("cast"):
        return
    # cast (always: fills voices for names added by hand in script_mn.txt)
    print("\n== cast")
    from dubflow import cast, script as script_mod
    features = {stamp(l["start"]): l for l in load_json(f["script_json"])}
    edited = [{**features.get(stamp(l["start"]), {}), **l, "mn": l["text"]} for l in read_script(f["script"])]
    if args.voices and not f["voices"].exists():
        save_json(f["voices"], load_json(args.voices))
    report = cast.cast(edited, f["voices"], f["cast"])
    print(f"      {'character':<22}{'gender':<10}{'age':<12}{'lines':>6}{'sec':>8}  voice")
    for c, r in report.items():
        pitch = f" ({r['pitch']:+d})" if r["pitch"] else ""
        print(f"      {c:<22}{r.get('gender', '?'):<10}{r.get('age_group', '?'):<12}{r['lines']:>6}{r['seconds']:>8}  "
              f"{r['voice']}{pitch}")
    warnings = script_mod.gender_warnings(edited, report)
    f["warnings"].write_text("\n".join(warnings) + "\n", encoding="utf-8")
    if warnings:
        print(f"      {len(warnings)} lines heard as the other gender than their character -> {f['warnings'].name}")
    print(f"      review: {f['script']}  and  {f['voices']}")

    if last < STAGES.index("dub"):
        return
    # dub
    print("\n== dub")
    from dubflow import dub
    dub.MODEL_ID = args.tts_model  # part of the cache key: switching model regenerates lines
    lines, voices = read_script(f["script"]), load_json(f["voices"])
    chars, n_todo, n_all = dub.estimate(lines, voices, args.tts_cache, args.stability)
    print(f"      {n_all} lines, {n_all - n_todo} cached, {n_todo} to generate = ~{chars} ElevenLabs credits")
    if chars and not args.yes:
        try:
            ok = input(f"      spend ~{chars} credits? [y/N] ").strip().lower() == "y"
        except EOFError:
            ok = False
        if not ok:
            sys.exit("      stopped before TTS (nothing spent). Re-run with --yes to generate.")
    clips = timed(dub.render, lines, voices, args.tts_cache, work / "segments", args.stability, args.dub_gain,
                  args.distance, args.tts_workers)
    duck = args.duck if args.duck is not None else (0.0 if stems else 0.05)
    timed(dub.mix, clips, f["dubbed"], duck, original=f["original"],
          vocals=f["vocals"] if stems else None, background=f["background"] if stems else None)
    print(f"      {f['dubbed'].with_suffix('.mp3')}")

    if last < STAGES.index("mux"):
        return
    print("\n== mux")
    from dubflow import media
    timed(media.mux, video, f["dubbed"], f["final"])
    print(f"      done: {f['final']}")


if __name__ == "__main__":
    main()
