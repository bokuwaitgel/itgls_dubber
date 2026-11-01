"""Mongolian recordings -> one clean voice sample per speaker -> ElevenLabs Instant Voice Clones in the pool.

extract: Demucs vocals, pyannote speakers, the clean stretches of each speaker (no one else talking, not
quiet bleed), gender/age per speaker, and WeSpeaker embeddings to find the same person across files.
Each person gets voices/clone/S01.mp3 etc. (up to SAMPLE_SEC of their best stretches) and an entry in
voices/clone/speakers.json. Listen, give the ones to keep a "name", then upload clones those and adds
them to voices/pool.json.
"""
import os
import re

import numpy as np
import soundfile as sf

from .analyze import _centroid, add_features, diarize, embed_lines, profile
from .common import ANALYSIS_RATE as SR
from .common import MIX_RATE, ROOT, decode, ffmpeg, gpu_lock, load_json, save_json
from .media import extract, separate

WORK = ROOT / "work" / "clone"
OUT = ROOT / "voices" / "clone"
SPEAKERS = OUT / "speakers.json"
POOL = ROOT / "voices" / "pool.json"

MERGE_GAP = 0.25  # same speaker, shorter pause: one stretch
EDGE = 0.1  # seconds trimmed off both ends of a stretch (diarization edges bleed into the neighbour)
MIN_PIECE = 1.5  # shorter clean stretches are left out
QUIET_PCT = 25  # a speaker's quietest stretches (below this percentile) are dropped: likely bleed under a dub
FEATURE_PIECES = 40  # gender/age measured on this many of a speaker's longest stretches
SAME_PERSON = 0.6  # cosine of speaker centroids across files above this = the same person
MIN_MERGE_SEC = 60.0  # a speaker with less clean speech has too shaky a centroid to merge into someone else
MIN_SPEAKER_SEC = 30.0  # people with less clean speech get no sample (a clone needs about a minute)
SAMPLE_SEC = 150.0  # seconds of speech per sample (ElevenLabs IVC: 1-3 min is best, more doesn't help)
SAMPLE_PIECE = (3.0, 12.0)  # preferred stretch length in a sample
SAMPLE_RMS = 0.1  # every stretch leveled to this before joining
SAMPLE_GAP = 0.3  # silence between stretches in a sample
POOL_AGE = {"child": "child", "teen": "young", "young_adult": "young", "adult": "middle_aged", "senior": "old"}


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "audio"


def clean_pieces(turns):
    """Stretches where only one speaker talks: same-speaker turns merged, overlaps with anyone else cut out."""
    by_spk = {}
    for t in sorted(turns, key=lambda t: t["start"]):
        runs = by_spk.setdefault(t["speaker"], [])
        if runs and t["start"] - runs[-1][1] <= MERGE_GAP:
            runs[-1][1] = max(runs[-1][1], t["end"])
        else:
            runs.append([t["start"], t["end"]])
    pieces = []
    for spk, runs in by_spk.items():
        others = sorted((t["start"], t["end"]) for t in turns if t["speaker"] != spk)
        for a, b in runs:
            cur = a
            for oa, ob in others:
                if ob <= cur or oa >= b:
                    continue
                if oa > cur:
                    pieces.append((spk, cur, oa))
                cur = max(cur, ob)
            if cur < b:
                pieces.append((spk, cur, b))
    return [{"voice": spk, "start": round(a + EDGE, 3), "end": round(b - EDGE, 3)}
            for spk, a, b in pieces if b - a - 2 * EDGE >= MIN_PIECE]


def _rms(audio, piece):
    x = audio[int(piece["start"] * SR):int(piece["end"] * SR)]
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0


def analyze_file(path, device):
    """One recording -> {speaker: {"pieces", "centroid", profile...}} for its speakers with enough clean speech."""
    work = WORK / slug(path.stem)
    work.mkdir(parents=True, exist_ok=True)
    original, vocals = work / "original.wav", work / "vocals.wav"
    if not original.exists():
        print("    extract", flush=True)
        extract(path, original)
    if not vocals.exists():
        print("    separate (Demucs)", flush=True)
        with gpu_lock(device):
            separate(original, vocals, work / "background.wav", device)
    audio = decode(vocals, SR)
    turns_path = work / "turns.json"
    if turns_path.exists():
        turns = load_json(turns_path)
    else:
        print("    diarize", flush=True)
        with gpu_lock(device):
            turns = diarize(audio, device)
        save_json(turns_path, turns)

    pieces = clean_pieces(turns)
    for p in pieces:
        p["rms"] = _rms(audio, p)
    kept = []
    for spk in {p["voice"] for p in pieces}:
        mine = [p for p in pieces if p["voice"] == spk]
        floor = np.percentile([p["rms"] for p in mine], QUIET_PCT)
        kept += [p for p in mine if p["rms"] >= floor]
    for i, p in enumerate(kept):
        p["id"] = i

    seconds = {}
    for p in kept:
        seconds[p["voice"]] = seconds.get(p["voice"], 0) + p["end"] - p["start"]
    speakers = [s for s, sec in seconds.items() if sec >= MIN_SPEAKER_SEC]
    kept = [p for p in kept if p["voice"] in speakers]
    longest = [p for s in speakers
               for p in sorted((p for p in kept if p["voice"] == s), key=lambda p: p["start"] - p["end"])[:FEATURE_PIECES]]
    print(f"    {len(speakers)} speakers with {MIN_SPEAKER_SEC:.0f}+ s of clean speech; gender/age + embeddings", flush=True)
    with gpu_lock(device):
        add_features(longest, audio, device)
        emb = embed_lines(kept, audio, device)
    by_id = {p["id"]: p for p in kept}
    info = profile(longest, "voice")
    out = {}
    for s in speakers:
        ids = [p["id"] for p in kept if p["voice"] == s and p["id"] in emb]
        if not ids:
            continue
        c = _centroid(ids, emb, by_id)
        mine = []
        for p in kept:
            if p["voice"] == s:
                p["sim"] = round(float(emb[p["id"]] @ c), 3) if p["id"] in emb else 0.0
                mine.append({k: p[k] for k in ("start", "end", "rms", "sim")})
        out[s] = {"file": path.name, "vocals": str(vocals), "seconds": round(seconds[s], 1), "pieces": mine,
                  "centroid": c, **{k: info[s][k] for k in ("p_female", "gender", "age", "age_group") if k in info.get(s, {})}}
    return out


def group_people(per_file):
    """Same person across files: greedy, biggest first, by centroid cosine; two speakers of one file never merge
    (diarization already told them apart), and a speaker under MIN_MERGE_SEC stays on its own."""
    people = []
    for spk in sorted((s for f in per_file for s in f.values()), key=lambda s: -s["seconds"]):
        best = max(((float(spk["centroid"] @ p["centroid"]), p) for p in people
                    if spk["file"] not in {m["file"] for m in p["members"]}), key=lambda x: x[0], default=(0, None))
        if spk["seconds"] < MIN_MERGE_SEC:
            best = (0, None)
        if best[0] >= SAME_PERSON:
            p = best[1]
            p["members"].append(spk)
            w = np.array([m["seconds"] for m in p["members"]])
            c = (np.stack([m["centroid"] for m in p["members"]]) * w[:, None]).sum(0)
            p["centroid"] = c / np.linalg.norm(c)
        else:
            people.append({"members": [spk], "centroid": spk["centroid"]})
    return people


def write_sample(person, out):
    """Best stretches of a person (most like their own voice, preferred length) leveled and joined into one mp3."""
    lo, hi = SAMPLE_PIECE
    pieces = [(m["vocals"], p) for m in person["members"] for p in m["pieces"]]
    pieces.sort(key=lambda vp: (not lo <= vp[1]["end"] - vp[1]["start"] <= hi, -vp[1]["sim"]))
    gap, chunks, total = np.zeros(int(SAMPLE_GAP * MIX_RATE), np.float32), [], 0.0
    for vocals, p in pieces:
        if total >= SAMPLE_SEC:
            break
        end = min(p["end"], p["start"] + hi)
        x, _ = sf.read(vocals, start=int(p["start"] * MIX_RATE), stop=int(end * MIX_RATE), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        rms = float(np.sqrt(np.mean(x ** 2)))
        if rms < 1e-4:
            continue
        chunks += [np.clip(x * SAMPLE_RMS / rms, -1, 1), gap]
        total += end - p["start"]
    wav = out.with_suffix(".wav")
    sf.write(str(wav), np.concatenate(chunks), MIX_RATE, subtype="PCM_16")
    ffmpeg("-i", wav, "-ac", 1, "-b:a", "192k", out)
    wav.unlink()
    return round(total, 1)


def extract_samples(paths, device):
    OUT.mkdir(parents=True, exist_ok=True)
    per_file = []
    for path in paths:
        print(f"  {path.name}", flush=True)
        per_file.append(analyze_file(path, device))
    people = sorted(group_people(per_file), key=lambda p: -sum(m["seconds"] for m in p["members"]))
    old = {s["sample"]: s for s in load_json(SPEAKERS)} if SPEAKERS.exists() else {}
    report = []
    for i, person in enumerate(people, 1):
        sample = OUT / f"S{i:02d}.mp3"
        sec = write_sample(person, sample)
        members = person["members"]
        w = np.array([m["seconds"] for m in members])
        judged = [m for m in members if "p_female" in m]
        entry = {"sample": sample.name, "name": "", "seconds_in_sample": sec,
                 "clean_seconds": round(float(w.sum()), 1),
                 "found_in": [f"{m['file']} ({m['seconds']:.0f}s)" for m in members]}
        if judged:
            jw = np.array([m["seconds"] for m in judged])
            p_female = float(np.average([m["p_female"] for m in judged], weights=jw))
            age = float(np.average([m["age"] for m in judged], weights=jw))
            entry |= {"gender": "female" if p_female >= 0.5 else "male", "p_female": round(p_female, 2),
                      "age": round(age), "pool_age": POOL_AGE[max(judged, key=lambda m: m["seconds"])["age_group"]]}
        prev = old.get(sample.name, {})
        if prev.get("voice_id"):  # already cloned in an earlier run: keep it, the sample file may now be someone else
            entry |= {"note": "S-numbers were reassigned; check this entry", "previous": prev}
        report.append(entry)
        print(f"  {sample.name}  {entry.get('gender', '?'):6} ~{entry.get('age', '?')}y  "
              f"{entry['clean_seconds']:.0f}s clean  {', '.join(entry['found_in'])}", flush=True)
    save_json(SPEAKERS, report)
    return report


def upload(remove_noise=False):
    """Clone every speakers.json entry that has a name and no voice_id yet; add it to voices/pool.json."""
    from elevenlabs import ElevenLabs

    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    report = load_json(SPEAKERS)
    pool = load_json(POOL) if POOL.exists() else []
    todo = [s for s in report if s.get("name", "").strip() and not s.get("voice_id")]
    if not todo:
        print("nothing to upload: give the samples to keep a \"name\" in voices/clone/speakers.json")
        return
    for s in todo:
        gender, age = s.get("gender", "female"), s.get("pool_age", "young")
        with open(OUT / s["sample"], "rb") as f:
            r = client.voices.ivc.create(name=s["name"].strip(), files=[(s["sample"], f, "audio/mpeg")],
                                         remove_background_noise=remove_noise,
                                         labels={"language": "mn", "gender": gender, "age": age},
                                         description=f"Mongolian dub voice cloned from {', '.join(s['found_in'])}")
        s["voice_id"] = r.voice_id
        pool.append({"name": s["name"].strip(), "voice_id": r.voice_id, "gender": gender, "age": age})
        save_json(SPEAKERS, report)
        save_json(POOL, pool)
        print(f"  {s['sample']} -> {s['name'].strip()} ({gender}, {age})  {r.voice_id}", flush=True)
