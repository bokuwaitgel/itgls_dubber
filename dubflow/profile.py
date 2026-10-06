"""Film profile from the transcript: before translating, the script model reads the whole English film and writes
down who is who (names with their Mongolian spelling, gender, age, relations, how they are addressed), who says
та or чи to whom, recurring terms, the setting and the plot. Whatever the user wrote in the film profile wins.

The profile is what keeps names, politeness and terms right and consistent across the many translation chunks.
"""
import hashlib
import json

from pydantic import BaseModel

from .common import ROOT, load_json, save_json

CACHE_DIR = ROOT / "cache" / "llm"

# Example text the old studio template put in every new profile: never pass it on as facts about the film.
PLACEHOLDERS = {
    "premise": ("Who the main characters are and what happens, in two or three sentences.",),
    "address": ("Who says та and who says чи to whom.",),
    "names": ({"Anna": "Анна"},),
    "aliases": ({"Anna": "Anna, Mom (female lead)"},),
    "narrator": ("the main character",),
    "genre": ("drama",),
}

PROMPT = """You prepare the Mongolian dub of an English-dubbed drama by reading its whole transcript first.
Each transcript row: start time, diarized voice id (noisy: one id can be several characters, one character can have
several ids), the gender/age heard on the voice, and the English text.

Return a profile the translators will follow for every line:
- genre: short, e.g. "modern romance short drama", "ancient-China palace drama", "fantasy".
- setting: "modern" or "period" (historical, ancient, fantasy kingdom).
- premise: 4-6 sentences: the main characters, how they are related, the main conflict and how it develops.
- narrator: name of the character who narrates in voice-over, or "" if there is none.
- characters: every named or recurring character (also important unnamed roles, e.g. "Mother", "Village Chief").
  name: as said in the English dialogue (for unnamed roles a short English role name).
  mongolian: how a Mongolian dub says the name, in Cyrillic: English names transliterated (Ethan -> Итан),
    Chinese names in the usual Mongolian Cyrillic form (Xiao Meng -> Сяо Мэн, Lin -> Линь); role names translated
    (Mother -> Ээж, Village Chief -> Тосгоны дарга).
  gender: female or male. age: child, teen, young adult, adult or senior.
  role: who they are and their relations ("Fiona's father, the emperor", "Anna's fake fiance").
  called: the ways others address or refer to them ("Dr. Montgomery", "Mommy", "Your Highness", "big sister").
- address: for each pair that talks to each other: speaker, listener and form. "та" goes UP: to elders, parents,
  parents-in-law, older in-laws, superiors, strangers, royalty. "чи" goes DOWN or ACROSS: parents and elders to
  children and to a daughter-in-law, between spouses, lovers, siblings and friends, and in contempt. Also
  the Mongolian address word if they use one (ээж ээ, аав аа, эгч ээ, ах аа, хүү минь, эзэн минь, хатан минь, эрхэм ээ).
- glossary: recurring special terms (titles, ranks, places, organisations, objects, illnesses, magic) with the
  Mongolian a dub should use.
Only state what the transcript supports. Do not invent names or relations."""


class Character(BaseModel):
    name: str
    mongolian: str
    gender: str
    age: str
    role: str
    called: list[str]


class Address(BaseModel):
    speaker: str
    listener: str
    form: str
    word: str


class Term(BaseModel):
    english: str
    mongolian: str


class Profile(BaseModel):
    genre: str
    setting: str
    premise: str
    narrator: str
    characters: list[Character]
    address: list[Address]
    glossary: list[Term]


def clean_film(film):
    """The user's profile without empty values and old template placeholders."""
    out = {}
    for key, value in film.items():
        if value in ("", None, {}, []) or value in PLACEHOLDERS.get(key, ()):
            continue
        out[key] = value
    return out


def _transcript(lines):
    from .analyze import age_group, gender_word
    rows = []
    for l in lines:
        t = int(l["start"])
        heard = f" {gender_word(l['p_female'])}/{age_group(l['age'])}" if "p_female" in l else ""
        rows.append(f"{t // 60}:{t % 60:02d} {l['voice']}{heard}: {l['en']}")
    return "\n".join(rows)


def build(lines, film, model, effort="medium"):
    """Profile as a film dict (names, aliases, address, glossary, premise, genre, narrator, notes). Cached."""
    from .script import _ask, _client

    known = clean_film(film)
    user = (f"Title: {film.get('title', '')}\n"
            + (f"What the user already knows (trust it): {json.dumps(known, ensure_ascii=False)}\n" if known else "")
            + "\nTranscript:\n" + _transcript(lines))
    path = CACHE_DIR / f"profile_{hashlib.sha1('|'.join([model, PROMPT, user]).encode()).hexdigest()[:16]}.json"
    if path.exists():
        return load_json(path)
    parsed, tokens = _ask(_client(model), model, PROMPT, user, None, Profile, effort)
    if parsed is None:
        raise RuntimeError("the script model returned no film profile")
    p = parsed
    out = {
        "genre": p.genre,
        "premise": p.premise,
        "narrator": p.narrator or "nobody (no voice-over)",
        "names": {c.name: c.mongolian for c in p.characters if c.mongolian},
        "aliases": {c.name: f"{', '.join(c.called) or c.name} ({c.gender}, {c.age}; {c.role})" for c in p.characters},
        "address": "; ".join(f"{a.speaker} to {a.listener}: {a.form}" + (f", says {a.word}" if a.word else "")
                             for a in p.address),
        "glossary": {t.english: t.mongolian for t in p.glossary},
        "setting": p.setting,
        "notes": ("Modern-day setting: modern words (phone, hospital, company, car) are right here."
                  if p.setting.lower().startswith("modern") else
                  "Period / fantasy setting: no modern slang or modern-world words."),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_json(path, out)
    print(f"      film profile: {len(p.characters)} characters, {len(p.address)} address rules, "
          f"{len(p.glossary)} terms ({tokens} tokens)", flush=True)
    return out


def merge(auto, film):
    """The user's own values win; names, aliases and glossary are merged entry by entry."""
    user = clean_film(film)
    out = {**auto, **user}
    for key in ("names", "aliases", "glossary"):
        out[key] = {**auto.get(key, {}), **user.get(key, {})}
    if auto.get("notes") and user.get("notes"):
        out["notes"] = f"{user['notes']}\n{auto['notes']}"
    return out
