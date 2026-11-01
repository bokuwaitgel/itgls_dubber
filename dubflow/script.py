"""English lines + voice evidence -> Mongolian dub script with character names and eleven_v3 tags.

Each line reaches the model with its pyannote voice id, the gender/age/emotion heard on it, and what that
voice was called in earlier chunks, so speaker names stay tied to real voices across the whole film.
Chunks are cached in cache/llm/ keyed by their full prompt, so re-runs are free.
"""
import hashlib
import json
import re
import subprocess
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

from .analyze import age_group, gender_word
from .common import FFMPEG, ROOT, TAG

MODEL = "codex"  # ChatGPT plan through the Codex CLI (see _codex); "gpt-6-luna" etc. to pay per token via the API
FALLBACK = "gpt-6-luna"  # API model for the rest of a run once the Codex plan limit is hit (needs OPENAI_API_KEY;
#                          "" to stop instead and wait for the limit to reset)
CHUNK = 40  # lines per request
DRAFT_WORKERS = 1  # film segments translated side by side: N is ~N times faster, but each segment after the
#                    first starts without the earlier names-per-voice history (slightly less consistent speakers)
POLISH_WORKERS = 6  # polish requests in flight; each starts as soon as its lines (and look-ahead) are drafted
EFFORT = "medium"  # OpenAI reasoning effort: low is faster, high is slower and more careful
MIN_CHUNK = 5  # a failing chunk is halved down to this size before giving up
CONTEXT = 15  # previous translated lines shown with each chunk
LOOKAHEAD = 8  # following lines shown, so sentences split over lines are translated whole
CHARS_PER_SEC = 15  # Mongolian characters that fit one second of dub at normal pace
EMOTION_MIN = 0.5  # emotion hints below this confidence are left out
AUDIO_PAD = 1.0  # Gemini: seconds of audio kept around each chunk
CACHE_DIR = ROOT / "cache" / "llm"

TAGS = ("quietly, whispers, softly, sad, crying, sobbing, hurt, bitter, sarcastic, angry, furious, shouting, "
        "screaming, excited, happy, playful, laughs, chuckles, sighs, gasps, nervous, worried, scared, trembling, "
        "confused, curious, surprised, shocked, hesitant, determined, serious, urgent, calm, gentle, reflective, "
        "cold, teasing, pleading, firm, warm, relieved, hopeful, tired, embarrassed")
# Speaker's own sounds that may also stay as tags.
SOUNDS = "giggles, laughing, sniffs, sniffles, grunts, groans, coughs, clears throat, panting, gulps, screams, sobs"
ALLOWED_TAGS = {t.strip() for t in (TAGS + ", " + SOUNDS).split(",")}
# Tags the model invents anyway -> nearest allowed one, so the line keeps its delivery instead of going flat.
TAG_MAP = {"concerned": "worried", "anxious": "worried", "tense": "nervous", "fearful": "scared",
           "panicked": "scared", "reassuring": "gentle", "sweet": "warm", "tender": "warm", "grateful": "warm",
           "sincere": "warm", "earnest": "serious", "grave": "serious", "somber": "sad", "emotional": "sad",
           "weary": "tired", "exhausted": "tired", "cheerful": "happy", "amused": "chuckles", "mocking": "sarcastic",
           "dry": "sarcastic", "threatening": "cold", "commanding": "firm", "formal": "calm", "polite": "calm",
           "professional": "calm", "matter-of-fact": "calm", "neutral": "calm", "apologetic": "hesitant",
           "frustrated": "angry", "annoyed": "angry", "irritated": "angry", "desperate": "pleading",
           "pained": "hurt", "breathless": "panting"}
FRAGMENT_SEC = 1.5  # max length of a piece join_fragments may hand back to the surrounding speaker
SPLIT_SURE = 0.3  # lines.json without "gender_sure" (older runs): a line this far from 0.5 counts as sure

SYSTEM = """You write the Mongolian dub script for an English-dubbed {genre} ("{title}").
{premise}

For every input line return: id (unchanged), speaker, text.

Evidence on each line:
- voice: the diarized voice id from the real audio. The same voice id is almost always the same character for
  the whole film; a voice actor may double a few minor roles. "voice_history" tells what each voice was called
  in earlier lines: keep that name unless the context clearly shows a different character.
- heard: gender and age group measured from the voice ("uncertain" = no clear call), plus the emotion the audio
  sounds like when it was clear. Gender is checked by two models and against the whole voice, so it is strong
  evidence: don't give a line heard as female to a male character or the other way round unless the words
  make it plain (the line names or answers the speaker, or clearly continues their sentence). The emotion
  model is trained on acted speech and over-reports "sad": trust the words and context first.
- on_screen: the film's own English subtitle shown while the line is said, read off the picture. It is the true
  wording: where "en" (speech recognition) mishears a name or a shouted line, translate on_screen instead. One
  subtitle can cover several lines and one line several subtitles: give each line only its own part. A line with
  an empty "en" was missed by speech recognition: translate its on_screen text and pick its speaker from context.

speaker:
- The character who says the line, in English. Reuse known character names. Narration is usually {narrator}.
- For unnamed minor roles use a short English role that includes the gender ("Village Woman 1",
  "Soldier Man", "Merchant Man") and keep it the same across lines.

text: the Mongolian dub line.
{style}
- Keep names English-style, spelled exactly: {names}. Other names: transliterate the English form and reuse
  the spelling from earlier lines.
- Fit the time slot: stay near the line's max_chars (shorter is fine, much longer is not).
- Sentences split over several lines: "next_lines" shows what follows. Translate the whole sentence, then
  cut the Mongolian at a natural point so each line still carries its part (Mongolian puts the verb last, so
  moving words between the pieces is fine). Every piece must still sound like speech, not a broken fragment.
  If a piece clearly continues the previous speaker's sentence, give it that speaker.
- Fix obvious transcription errors from context.

Tags (for the TTS voice):
- Start almost every line with one or two performance tags in English square brackets that describe how THIS
  speaker says it, e.g. [warm], [sad] [quietly], [angry] [shouting], [firm], [excited]. Allowed tags: {tags}.
  A tag can go mid-line when the emotion changes. Only a truly flat line goes without one.
- The speaker's own sounds in the input ([laughs], [sighs], [gasps], [sobbing]) become tags in the right place.
- Background sounds are NOT the speaker: drop [crowd murmuring], [birds chirping], [music], [door opens] etc.
- A line with only the speaker's own sound stays tag-only, e.g. "[laughs]"."""

STYLE = """- Sound like real Mongolians talking to each other today, the way a good Mongolian TV dub sounds - not a
  translation, not a book. Rephrase freely; keep the meaning, intent, emotion and who-is-addressed exactly, but
  never the English word order, idioms or sentence build.
- Everyday spoken Khalkha: short sentences, drop pronouns when obvious, and the particles and spoken endings a
  Mongolian really uses: шүү, шүү дээ, дээ, л, л дээ, биз дээ, юм, юм чинь, байхгүй юу, байх аа, шиг байна,
  тэ, аа/ээ/оо; -чихлаа, -чихсан, -чих, -ъя, -гаарай, -даа, -хгүй юу, -сан юм уу.
- Live-speech openers and reactions where the feeling calls for them: за, заа, өө, аан, хөөе, ёо, яахав,
  нээрээ юу, тэгэлгүй яахав, үгүй ээ, болъё доо, юу гэнээ?, яасан бэ?, одоо яах юм. Emotion can show as
  repetition or a broken start ("Би... би мэдээгүй"). Don't stuff them in every line - one where a person would.
- Not literary: in dialogue no -жээ/-чээ/-лээ шүү past ("мартжээ" -> "мартчихсан", "хэлжээ" -> "хэлсэн"), no -вч
  ("харагдавч" -> "харагдаж байгаа ч"), no мэт/төдий/нэгэн цагт/дэндүү/шахуу/авч одох -> шиг/л/урьд нь/хэтэрхий/
  бараг/авах. No bookish/official words (хэрэгжүүлэх, эрхэм, тухайлбал) unless the character is formal.
  Narration ("I/we" looking back) may be a little softer and storytelling, but still said aloud, not written.
- Examples: "You really are cruel." -> "Ямар хатуу хүн бэ чи." | "Did you eat lunch?" -> "Хоолоо идсэн үү?" |
  "I had told Noah too many gentle lies." -> "Би Ноад хэтэрхий олон зөөлөн худал хэлсэн." | "Are you serious?" ->
  "Нээрээ юу?" | "Don't worry." -> "Санаа зовох хэрэггүй ээ." | "Fine, let's go." -> "За за, явъя."
- Use Mongolian words, not Russian/English loans, when a common Mongolian word exists: отряд -> бүлэг/цэрэг,
  лагерь -> хуаран, бизнес -> наймаа/ажил, проблем -> асуудал, окей -> за, идея -> санаа.
- Grammar must be correct: case endings, vowel harmony, possessives (-аа/-ээ/-оо, чинь, минь).
- Politeness: та to elders, parents-in-law, superiors and strangers; чи between spouses, siblings, friends
  and to children. Family address words as a Mongolian would say them (ээж, аав, ах, эгч, дүү, хүү минь).
- Numbers and abbreviations as Mongolian words (2025 -> хоёр мянга хорин тав), no digits.
- Follow the setting in the notes: period/fantasy = no modern slang or modern-world words; modern = everyday
  modern speech is right."""

POLISH = """You are the native Mongolian script editor of a TV dub ("{title}", {genre}). Each line has the English
original, who says it, the gender/age heard on the voice, and a draft Mongolian dub line. Return every line
(same id) with the final speaker and text.

Speaker: keep it unless it is clearly wrong: nobody thanks, calls or answers themselves by name ("Thank you,
Isola" is not said by Isola); a short piece that just continues someone's unfinished sentence belongs to that
person; a line heard as female is not said by a male character and the other way round. Only use speaker
names that already appear in the lines or previous_lines.

Fix the draft so a Mongolian viewer hears natural, correct, living speech:
- Meaning: must match the English (intent, facts, who is addressed). Fix mistranslations and lost meaning. Where a
  line has "on_screen" (the film's own subtitle), that is the true English wording, not "en".
- Language: correct grammar, case endings and vowel harmony; replace calques, bookish words and needless
  Russian/English loans with what a Mongolian would really say.
{style}
- Sentences split over consecutive lines must read as one natural sentence when played in order.
- Keep each line near its max_chars (the dub must fit the time slot); shorten by dropping filler, not meaning.
- Read each line aloud in your head: if it sounds like a subtitle or a book, rewrite it the way that character
  would really say it in that moment.
- Keep names exactly as in the draft. Keep the square-bracket performance tags (they drive the TTS emotion); fix
  one only if it is wrong for the line; a background sound is not a tag. Tags stay in English and must be from:
  {tags}.
If a draft line is already good, return it unchanged."""

EDIT = """You are the final Mongolian dialogue editor (найруулагч) of a TV dub ("{title}", {genre}). The lines are
already translated and checked for meaning. Read them in order as one scene, the way a viewer hears them, and fix
only what a native Mongolian listener would stumble on:
- words that don't exist, are misspelled, or are the wrong word for the meaning ("чам шиг дараатай" for "dead
  weight like you" -> "чам шиг ачаа")
- wrong collocations, case endings, vowel harmony, verb forms, possessives
- phrasing that sounds translated, written or stiff; a line that doesn't follow naturally from the line before
- a sentence spread over several lines must read as one sentence when played in order
Keep the meaning, every [tag] (in English, from: {tags}), names exactly as written, and the length (stay near
max_chars, never much longer). Return every line (same id) with its text; a line that is already good comes back
unchanged."""

STYLE_FILE = ROOT / "dubflow" / "style_mn.txt"  # lines from human Mongolian dubs of this genre (one per line)
EDIT_CONTEXT = 10  # final lines before a chunk shown to the editor

AUDIO_NOTE = """
The audio of these lines is attached; each line's "at" is its start in seconds from the clip start.
Listen to it to confirm who speaks and how each line is delivered."""


class Line(BaseModel):
    id: int
    speaker: str
    text: str


class Chunk(BaseModel):
    lines: list[Line]


class Fixed(BaseModel):
    id: int
    speaker: str
    text: str


class FixedChunk(BaseModel):
    lines: list[Fixed]


class Edited(BaseModel):
    id: int
    text: str


class EditedChunk(BaseModel):
    lines: list[Edited]


def style_examples():
    """STYLE_FILE as a prompt block: how real Mongolian dubbers of this genre talk (voice, not content)."""
    if not STYLE_FILE.exists():
        return ""
    rows = [l.strip() for l in STYLE_FILE.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    return ("\n\nLines from human Mongolian dubs of this kind of drama. Match their voice - word choice, particles, "
            "rhythm, how people address each other - never their content:\n" + "\n".join(rows)) if rows else ""


def read_guide(path):
    """Plain text of a character/story guide (.docx or .txt)."""
    if not path or not Path(path).exists():
        return ""
    path = Path(path)
    if path.suffix == ".docx":
        import docx
        d = docx.Document(str(path))
        rows = [" | ".join(c.text.strip() for c in r.cells) for t in d.tables for r in t.rows]
        return "\n".join([p.text for p in d.paragraphs if p.text.strip()] + rows)
    return path.read_text(encoding="utf-8")


def heard(line):
    if "p_female" not in line:
        return None
    parts = [gender_word(line["p_female"]), age_group(line["age"])]
    if line.get("emotion_score", 0) >= EMOTION_MIN:
        parts.append(line["emotion"])
    return ", ".join(parts)


def audio_clip(audio, start, end):
    return subprocess.run([FFMPEG, "-v", "quiet", "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}", "-i", str(audio),
                           "-ac", "1", "-ar", "16000", "-b:a", "32k", "-f", "mp3", "-"],
                          capture_output=True, check=True).stdout


def _client(model):
    if model.startswith("codex"):
        return None
    if model.startswith("gemini"):
        from google import genai
        return genai.Client()
    from openai import OpenAI
    return OpenAI()


def _strict(node, defs=None):
    """Pydantic JSON schema -> the strict form structured outputs need: $refs inlined, every property required,
    no extra properties."""
    defs = node.get("$defs", {}) if defs is None else defs
    if "$ref" in node:
        return _strict(defs[node["$ref"].rsplit("/", 1)[-1]], defs)
    out = {k: v for k, v in node.items() if k not in ("$defs", "title", "default")}
    if out.get("type") == "object":
        out["properties"] = {k: _strict(v, defs) for k, v in out.get("properties", {}).items()}
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    if "items" in out:
        out["items"] = _strict(out["items"], defs)
    return out


def _codex_default_model():
    """The model set in ~/.codex/config.toml (still wanted when the rest of that config is skipped), or None."""
    import os
    import tomllib

    path = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "config.toml"
    try:
        return tomllib.loads(path.read_text(encoding="utf-8")).get("model")
    except (OSError, tomllib.TOMLDecodeError):
        return None


def _codex(model, system, user, schema, effort, images=()):
    """"codex" or "codex:<model>": the model of your ChatGPT plan through the Codex CLI, so no API billing (plan
    usage limits apply). It runs read-only in an empty temp dir and must answer in the schema's JSON. The user's
    config.toml is skipped: its plugins and MCP servers would add their tool definitions (~20k tokens) to every call."""
    import shutil
    import tempfile

    exe = shutil.which("codex")
    if not exe:
        raise SystemExit("codex CLI not found: npm install -g @openai/codex, then codex login")
    with tempfile.TemporaryDirectory() as tmp:
        schema_path, out_path = Path(tmp) / "schema.json", Path(tmp) / "out.json"
        schema_path.write_text(json.dumps(_strict(schema.model_json_schema())), encoding="utf-8")
        cmd = [exe, "exec", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check", "-s", "read-only", "-C", tmp,
               "--color", "never",
               "--json", "--output-schema", str(schema_path), "-o", str(out_path),
               "-c", f'model_reasoning_effort="{effort}"']
        if name := model.split(":", 1)[1] if ":" in model else _codex_default_model():
            cmd += ["-m", name]
        cmd += [f"--image={p}" for p in images]
        prompt = ("You are used here as a text engine, not a coding agent: don't run commands or read files, just "
                  f"answer with the JSON.\n\n{system}\n\n---\n\n{user}")
        r = subprocess.run(cmd + ["-"], input=prompt, capture_output=True, text=True, encoding="utf-8")
        tokens, error = 0, ""
        for row in r.stdout.splitlines():
            try:
                event = json.loads(row)
            except json.JSONDecodeError:
                continue
            usage = event.get("usage") or {}
            tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            if event.get("type") in ("error", "turn.failed"):
                error = event.get("message") or (event.get("error") or {}).get("message", "")
        if "usage limit" in error:  # nothing more will work until the plan resets; cached calls are kept for a rerun
            raise CodexLimit(f"codex: {error}")
        if not out_path.exists() or not out_path.read_text(encoding="utf-8").strip():
            print(f"      codex gave no answer (exit {r.returncode}): {(error or r.stderr.strip())[-300:]}", flush=True)
            return None, max(tokens, 1)
        try:
            return schema.model_validate_json(out_path.read_text(encoding="utf-8")), max(tokens, 1)
        except ValueError as e:
            print(f"      codex answer didn't fit the schema: {str(e)[:200]}", flush=True)
            return None, max(tokens, 1)


class CodexLimit(SystemExit):
    """The ChatGPT plan's usage limit: stops the run unless _ask can fall back to the API."""


_fallback = {}  # set once the Codex limit is hit: {"client": OpenAI client}; later calls go straight to FALLBACK


def _ask(client, model, system, user, clip, schema, effort=EFFORT):
    if model.startswith("codex"):
        if not _fallback:
            try:
                return _codex(model, system, user, schema, effort)
            except CodexLimit as e:
                import os
                if not (FALLBACK and os.environ.get("OPENAI_API_KEY")):
                    raise
                if not _fallback:
                    print(f"      {e}\n      -> the rest goes through the API ({FALLBACK})", flush=True)
                    _fallback["client"] = _client(FALLBACK)
        client, model = _fallback["client"], FALLBACK
    if model.startswith("gemini"):
        from google.genai import types
        parts = ([types.Part.from_bytes(data=clip, mime_type="audio/mp3")] if clip else []) + [user]
        r = client.models.generate_content(
            model=model, contents=parts,
            config=types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                               response_schema=schema))
        return r.parsed, r.usage_metadata.total_token_count
    r = client.responses.parse(model=model, reasoning={"effort": effort}, text_format=schema,
                               input=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.output_parsed, r.usage.total_tokens


def _cached(client, model, system, user, schema, n, clip=None, effort=EFFORT):
    """Model call cached by its full prompt; retried until it returns exactly ids 0..n-1.
    Returns (None, tokens) if it keeps failing, so the caller can split the chunk."""
    key = [model, system, user] + ([effort] if effort != EFFORT else [])  # default effort keeps old cache keys
    path = CACHE_DIR / f"{hashlib.sha1('|'.join(key).encode()).hexdigest()[:16]}.json"
    if path.exists():
        return schema.model_validate_json(path.read_text(encoding="utf-8")).lines, 0
    total = 0
    for attempt in range(2):
        parsed, used = _ask(client, model, system, user, clip, schema, effort)
        total += used
        result = parsed.lines if parsed else []
        if sorted(l.id for l in result) == list(range(n)):
            path.write_text(schema(lines=result).model_dump_json(indent=1), encoding="utf-8")
            return result, total
        print(f"      got {len(result)}/{n} lines back, retrying ({attempt + 1}/2)", flush=True)
    return None, total


def _chunks(total, size):
    return [(n, min(size, total - n)) for n in range(0, total, size)]


def _solve(n, size, fn):
    """fn(n, size) -> (applied, tokens). A range the model keeps failing is retried as two halves, in order."""
    applied, tokens = fn(n, size)
    if applied:
        return tokens
    if size <= MIN_CHUNK:
        raise RuntimeError(f"model keeps failing on lines {n}-{n + size}; check them in lines.json")
    half = size // 2
    print(f"      splitting lines {n}-{n + size} in two", flush=True)
    return tokens + _solve(n, half, fn) + _solve(n + half, size - half, fn)


def _film_notes(film):
    notes = f"\n\n{film['notes']}" if film.get("notes") else ""
    if film.get("glossary"):
        notes += f"\n\nGlossary (use these Mongolian terms): {json.dumps(film['glossary'], ensure_ascii=False)}"
    if film.get("address"):
        notes += f"\n\nHow characters address each other: {film['address']}"
    return notes


def max_chars(line):
    return max(10, int((line["end"] - line["start"]) * CHARS_PER_SEC))


def write_script(lines, film, voices, model, audio=None, chunk_size=CHUNK, polish_model=None,
                 draft_workers=DRAFT_WORKERS, polish_workers=POLISH_WORKERS, effort=EFFORT, edit=True):
    """Returns lines with "speaker", "draft" (first pass) and "mn" (after the polish and edit passes) added.

    Draft chunks run in order within each segment (each chunk sees the previous lines and what each voice was
    called so far); polish chunks run in parallel alongside, each as soon as the drafts it looks at exist; the
    editor then reads the finished lines as scenes and fixes wording only."""
    guide = read_guide(ROOT / film["guide"] if film.get("guide") else None)
    system = SYSTEM.format(title=film.get("title", "untitled"), genre=film.get("genre", "drama"),
                           premise=film.get("premise", ""), narrator=film.get("narrator", "the main character"),
                           names=json.dumps(film.get("names", {}), ensure_ascii=False), tags=TAGS, style=STYLE)
    system += _film_notes(film)
    system += f"\n\nCaption spellings of characters: {json.dumps(film['aliases'], ensure_ascii=False)}" \
        if film.get("aliases") else ""
    system += f"\n\nCharacter and story guide:\n{guide}" if guide else ""
    system += style_examples()
    client = _client(model)
    if model.startswith("gemini"):
        system += AUDIO_NOTE
    not_speakers = set(film.get("not_speakers", []))
    known = (set(film.get("names", {})) | set(film.get("aliases", {}))) - not_speakers
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    out = [None] * len(lines)  # drafted lines, filled in as chunks finish
    lock = threading.Lock()
    stats = {"tokens": 0, "drafted": 0, "polished": 0}
    polisher = Polisher(out, film, polish_model or model, chunk_size, effort, stats, lock, polish_workers)

    def draft_segment(ranges):
        history = defaultdict(Counter)  # voice -> Counter(character), within this segment
        context, characters = [], set(known)

        def one(n, size):
            chunk = lines[n:n + size]
            clip_start = max(0.0, chunk[0]["start"] - AUDIO_PAD)
            in_chunk = sorted({l["voice"] for l in chunk} - {"SFX"})
            payload = {
                "known_characters": sorted(characters),
                "voice_history": {v: {"voice": f"{voices[v].get('gender', '?')}, {voices[v].get('age_group', '?')}",
                                      "called_so_far": dict(history[v].most_common(4))}
                                  for v in in_chunk if v in voices},
                "previous_lines": [{"speaker": d["draft_speaker"], "voice": d["voice"], "en": d["en"],
                                    "mn": d["draft"]} for d in context[-CONTEXT:]],
                "lines": [{"id": i, "voice": l["voice"], "heard": heard(l), "seconds": round(l["end"] - l["start"], 1),
                           "max_chars": max_chars(l), "en": l["en"]}
                          | ({"on_screen": l["on_screen"]} if l.get("on_screen") else {})
                          | ({"at": round(l["start"] - clip_start, 1)} if audio else {})
                          for i, l in enumerate(chunk)],
                "next_lines": [l["en"] for l in lines[n + size:n + size + LOOKAHEAD]],
            }
            user = json.dumps(payload, ensure_ascii=False, indent=1)
            clip = audio_clip(audio, clip_start, chunk[-1]["end"] + AUDIO_PAD) if audio else None
            result, used = _cached(client, model, system, user, Chunk, len(chunk), clip, effort)
            if result is None:
                return False, used
            for r in sorted(result, key=lambda r: r.id):
                speaker = r.speaker.strip()
                line = {**chunk[r.id], "speaker": speaker, "draft_speaker": speaker, "draft": r.text.strip()}
                out[n + r.id] = line
                context.append(line)
                characters.add(speaker)
                if line["voice"] != "SFX":
                    history[line["voice"]][speaker] += 1
            with lock:
                stats["drafted"] += size
                print(f"      draft  [{stats['drafted']}/{len(lines)}] "
                      f"{'cached' if not used else f'{used} tokens'}", flush=True)
            polisher.schedule()
            return True, used

        return sum(_solve(n, size, one) for n, size in ranges)

    ranges = _chunks(len(lines), chunk_size)
    per = -(-len(ranges) // max(1, min(draft_workers, len(ranges))))  # chunks per segment, rounded up
    segments = [ranges[i:i + per] for i in range(0, len(ranges), per)]
    try:
        with ThreadPoolExecutor(len(segments)) as pool:
            for used in pool.map(draft_segment, segments):
                stats["tokens"] += used
        polisher.schedule()
        polisher.wait()
        if edit:
            stats["tokens"] += edit_pass(out, film, polish_model or model, chunk_size, effort, polish_workers)
    except RuntimeError as e:
        sys.exit(str(e))
    finally:
        polisher.close()
    join_fragments(out)
    moved = split_gender(out)
    if moved:
        print(f"      {moved} lines heard as the other gender than their character -> '<name> (female?)/(male?)'")
    print(f"      {stats['tokens']} tokens this run")
    return out


def _tag(m):
    tag = m.group(0)[1:-1].strip().lower()
    tag = TAG_MAP.get(tag, tag)
    return f"[{tag}]" if tag in ALLOWED_TAGS else ""


def clean_tags(text):
    """Keep only tags the TTS should perform (near-synonyms mapped onto allowed ones); background sounds
    ([crowd murmuring], [birds chirping]) go."""
    return re.sub(r"\s{2,}", " ", TAG.sub(_tag, text)).strip()


def join_fragments(lines):
    """A short piece sandwiched inside one speaker's unfinished sentence is that speaker's (diarization jitter)."""
    for a, b, c in zip(lines, lines[1:], lines[2:]):
        if (a["speaker"] == c["speaker"] != b["speaker"] and b["end"] - b["start"] < FRAGMENT_SEC
                and a["en"].rstrip()[-1:] not in ".?!" and b["start"] - a["end"] < 0.5):
            b["speaker"] = a["speaker"]


def split_gender(lines):
    """Sure lines (both gender models agree) heard as the other gender than most of their character's speech go
    to "<name> (female?)" / "<name> (male?)", so casting gives them a matching voice and the review shows them
    to fix. Returns the count."""
    sure = [l for l in lines if "p_female" in l and l.get("gender_sure", abs(l["p_female"] - 0.5) >= SPLIT_SURE)]
    secs = defaultdict(lambda: [0.0, 0.0])  # speaker -> [female seconds, male seconds]
    for l in sure:
        secs[l["speaker"]][l["p_female"] < 0.5] += l["end"] - l["start"]
    moved = 0
    for l in sure:
        f, m = secs[l["speaker"]]
        main_female = f > m
        if (l["p_female"] > 0.5) != main_female and min(f, m) < 0.5 * max(f, m):
            l["speaker"] = f"{l['speaker']} ({'female' if l['p_female'] > 0.5 else 'male'}?)"
            moved += 1
    return moved


class Polisher:
    """Second pass: a native-editor rewrite of the draft for meaning, grammar and natural speech. Sets "mn".

    A chunk is sent as soon as its drafts, the drafts just before it and its look-ahead exist, so polishing runs
    alongside the draft. Its context comes from drafts only, so polish chunks never wait on each other."""

    def __init__(self, lines, film, model, chunk_size, effort, stats, lock, workers=POLISH_WORKERS):
        self.lines, self.model, self.effort, self.stats, self.lock = lines, model, effort, stats, lock
        self.system = POLISH.format(title=film.get("title", "untitled"), genre=film.get("genre", "drama"),
                                    style=STYLE, tags=TAGS + ", " + SOUNDS) + _film_notes(film) + style_examples()
        self.client = _client(model)
        self.pending = _chunks(len(lines), chunk_size)
        self.pool = ThreadPoolExecutor(max(1, workers))
        self.futures = []

    def _ready(self, n, size):
        lo, hi = max(0, n - CONTEXT), min(len(self.lines), n + size + LOOKAHEAD)
        return all(self.lines[i] is not None for i in range(lo, hi))

    def schedule(self):
        with self.lock:
            ready = [r for r in self.pending if self._ready(*r)]
            self.pending = [r for r in self.pending if r not in ready]
            self.futures += [self.pool.submit(_solve, n, size, self._one) for n, size in ready]

    def wait(self):
        for f in self.futures:
            self.stats["tokens"] += f.result()

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def _one(self, n, size):
        lines, chunk = self.lines, self.lines[n:n + size]
        payload = {
            "previous_lines": [{"speaker": d["draft_speaker"], "draft": d["draft"]}
                               for d in lines[max(0, n - CONTEXT):n]],
            "lines": [{"id": i, "speaker": l["draft_speaker"], "heard": heard(l), "max_chars": max_chars(l),
                       "en": l["en"], "draft": l["draft"]} | ({"on_screen": l["on_screen"]} if l.get("on_screen") else {})
                      for i, l in enumerate(chunk)],
            "next_lines": [{"speaker": l["draft_speaker"], "en": l["en"], "draft": l["draft"]}
                           for l in lines[n + size:n + size + LOOKAHEAD]],
        }
        result, used = _cached(self.client, self.model, self.system, json.dumps(payload, ensure_ascii=False, indent=1),
                               FixedChunk, len(chunk), effort=self.effort)
        if result is None:
            return False, used
        for r in result:
            chunk[r.id]["mn"] = clean_tags(r.text)
            chunk[r.id]["speaker"] = r.speaker.strip() or chunk[r.id]["draft_speaker"]
        with self.lock:
            self.stats["polished"] += size
            print(f"      polish [{self.stats['polished']}/{len(lines)}] "
                  f"{'cached' if not used else f'{used} tokens'}", flush=True)
        return True, used


def edit_pass(lines, film, model, chunk_size, effort, workers=POLISH_WORKERS):
    """Native editor over the polished lines (see EDIT): wording, grammar and flow only; speakers, meaning and tags
    stay. Chunks run in parallel; each sees the final lines before it and after it as context."""
    system = EDIT.format(title=film.get("title", "untitled"), genre=film.get("genre", "drama"),
                         tags=TAGS + ", " + SOUNDS) + _film_notes(film) + style_examples()
    client, lock, done = _client(model), threading.Lock(), [0]
    ranges = _chunks(len(lines), chunk_size)
    context = [{"speaker": l["speaker"], "mn": l["mn"]} for l in lines]  # fixed before editing: chunks don't wait

    def one(n, size):
        chunk = lines[n:n + size]
        payload = {
            "previous_lines": context[max(0, n - EDIT_CONTEXT):n],
            "lines": [{"id": i, "speaker": l["speaker"], "en": l["en"], "max_chars": max_chars(l), "mn": l["mn"]}
                      | ({"on_screen": l["on_screen"]} if l.get("on_screen") else {}) for i, l in enumerate(chunk)],
            "next_lines": context[n + size:n + size + LOOKAHEAD],
        }
        result, used = _cached(client, model, system, json.dumps(payload, ensure_ascii=False, indent=1),
                               EditedChunk, len(chunk), effort=effort)
        if result is None:
            return False, used
        changed = 0
        for r in result:
            text = clean_tags(r.text)
            changed += text != chunk[r.id]["mn"]
            chunk[r.id]["mn"] = text
        with lock:
            done[0] += size
            print(f"      edit   [{done[0]}/{len(lines)}] {changed} changed, "
                  f"{'cached' if not used else f'{used} tokens'}", flush=True)
        return True, used

    with ThreadPoolExecutor(max(1, min(workers, len(ranges)))) as pool:
        return sum(pool.map(lambda r: _solve(*r, one), ranges))


def gender_warnings(lines, cast):
    """Lines whose heard gender contradicts their character's overall gender (likely misattributed)."""
    out = []
    for l in lines:
        c = cast.get(l["speaker"], {})
        g = gender_word(l["p_female"]) if "p_female" in l else "uncertain"
        if g != "uncertain" and c.get("gender") not in (None, "uncertain", g):
            out.append(f"{l['id']:>5}  {l['start']:8.1f}s  {l['speaker']} ({c['gender']}) but heard {g}: {l['en'][:70]}")
    return out
