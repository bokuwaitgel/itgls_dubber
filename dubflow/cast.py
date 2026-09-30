"""Auto voice map: every character gets a pool voice matching its measured gender and age.

Pool: voices/pool.json (made from your ElevenLabs Mongolian cloned voices on first run; edit freely).
Bigger roles pick first and get unshifted voices; smaller roles reuse a voice with a pitch shift
(dub stage keeps duration), so no two characters share the same voice+pitch while combinations last.
"""
import os
from collections import Counter

from .analyze import profile
from .common import ROOT, TAG, load_json, save_json

POOL = ROOT / "voices" / "pool.json"
AGE_TO_POOL = {"child": ["child"], "teen": ["young", "child"], "young_adult": ["young", "middle_aged"],
               "adult": ["middle_aged", "young"], "senior": ["old", "middle_aged", "young"]}
PITCHES = {"child": [0, 1, -1, 2], "teen": [2, 3, 1, 4], "young_adult": [0, 1, -1, 2, -2, 3],
           "adult": [0, -1, 1, -2, 2, -3], "senior": [-2, -3, -1, -4]}
FEMALE_WORDS = ("woman", "girl", "mother", "mom", "lady", "wife", "sister", "aunt", "daughter", "female", "maid")
MALE_WORDS = ("man", "boy", "father", "dad", "husband", "brother", "uncle", "son", "male", "guy", "soldier")


def load_pool():
    if POOL.exists():
        return load_json(POOL)
    from elevenlabs import ElevenLabs

    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    pool = []
    for v in client.voices.search(page_size=100).voices:
        labels, name = dict(v.labels or {}), (v.name or "").lower()
        if v.category != "cloned" or labels.get("language") != "mn" or any(w in name for w in ("test", "idk")):
            continue
        child = any(w in name for w in ("girl", "boy", "child", "kid"))
        pool.append({"name": v.name, "voice_id": v.voice_id, "gender": labels.get("gender", "female"),
                     "age": "child" if child else labels.get("age", "young")})
    if not pool:
        raise SystemExit("no Mongolian cloned voices in your ElevenLabs account; write voices/pool.json by hand")
    pool.sort(key=lambda v: "lead" not in v["name"].lower())  # file order = preference: leads go to biggest roles
    POOL.parent.mkdir(exist_ok=True)
    save_json(POOL, pool)
    print(f"      wrote {POOL} ({len(pool)} voices) - edit it to change the casting pool")
    return pool


def _gender(name, info):
    if "p_female" in info:
        return "female" if info["p_female"] >= 0.5 else "male"
    low = name.lower()
    return "female" if any(w in low for w in FEMALE_WORDS) and not any(f" {w}" in f" {low}" for w in MALE_WORDS) \
        else "male"


def _candidates(pool, gender, group):
    same = [v for v in pool if v["gender"] == gender]
    for ages in (AGE_TO_POOL[group], None):
        found = [v for age in ages for v in same if v["age"] == age] if ages else [v for v in same if v["age"] != "child"]
        if found:
            return found
    return [v for v in pool if v["age"] == "child"] if group == "child" else same or pool


def cast(lines, voices_path, cast_path):
    """Fill voices_path (character -> voice) for characters not in it yet; write a cast report."""
    voices = load_json(voices_path) if voices_path.exists() else {}
    pool = load_pool()
    by_id = {v["voice_id"]: v for v in pool}
    characters = profile([l for l in lines if TAG.sub("", l["mn"]).strip()], "speaker")  # skip sound-only lines

    used = Counter()
    for entry in voices.values():
        vid, pitch = (entry["voice_id"], entry.get("pitch", 0)) if isinstance(entry, dict) else (entry, 0)
        used[(vid, pitch)] += 1
    for name, info in characters.items():  # biggest roles first
        if name in voices:
            continue
        gender, group = _gender(name, info), info.get("age_group", "young_adult")
        options = [(v["voice_id"], p) for p in PITCHES[group] for v in _candidates(pool, gender, group)]
        vid, pitch = next((o for o in options if not used[o]), min(options, key=lambda o: used[o]))
        used[(vid, pitch)] += 1
        voices[name] = {"voice_id": vid, "pitch": pitch} if pitch else vid

    report = {}
    for name, info in characters.items():
        entry = voices[name]
        vid, pitch = (entry["voice_id"], entry.get("pitch", 0)) if isinstance(entry, dict) else (entry, 0)
        report[name] = {**info, "voice": by_id.get(vid, {}).get("name", vid), "pitch": pitch}
    save_json(voices_path, voices)
    save_json(cast_path, report)
    return report
