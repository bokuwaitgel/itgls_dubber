# Dub Studio

Turns English videos into Mongolian dubs. You upload a video in a web page. The pipeline transcribes it, finds who
speaks when, and translates it into natural spoken Mongolian. Then you review and fix the script next to the video.
Mongolian voices are only generated when you confirm the ElevenLabs credit cost.

```
upload ─▶ audio ─▶ voice/music split ─▶ transcript ─▶ speakers ─▶ translation ─▶ casting ─▶ YOUR REVIEW ─▶ voices ─▶ final video
          ffmpeg    Demucs               Scribe v2      pyannote     GPT (2 passes)  voice pool                ElevenLabs   ffmpeg
```

## Setup

Needs Python 3.11 and an NVIDIA GPU (CPU works, but slowly). ffmpeg comes with `imageio-ffmpeg`.

```bash
conda create -n dub python=3.11 -y
conda activate dub
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
cp .env.example .env          # then fill in the keys
```

Keys in `.env`:

| key | used for |
| --- | --- |
| `ELEVENLABS_API_KEY` | transcription (Scribe) and the Mongolian voices |
| `OPENAI_API_KEY` | translation |
| `GEMINI_API_KEY` | only with a `gemini-*` translation model |
| `HF_TOKEN` | pyannote speaker diarization (accept the terms at hf.co/pyannote/speaker-diarization-3.1) |

The voice pool (`voices/pool.json`) is built from the Mongolian cloned voices in your ElevenLabs account on the
first dub. Edit it to change who can be cast.

## Run the studio

```bash
python -m studio            # http://127.0.0.1:8000
```

Or as a desktop app in its own window: run `powershell -ExecutionPolicy Bypass -File make_shortcut.ps1` once to
put a **Dub Studio** icon on the desktop, then double-click it (same as `pythonw -m studio.desktop`). Closing the
window stops the server; a job that was running then shows as stopped, with Resume. Its log is `data/desktop.log`.

1. **New dub**: drop a video and pick or write a film profile (see below). Try a clip of a few minutes first.
2. Wait while the job runs. You can close the page, because the work runs on the server. Two jobs run at once
   (`DUB_WORKERS` in `.env`); their GPU steps (voice separation, speaker analysis) take turns so an 8 GB card
   doesn't run out of memory, while transcription, translation and voice generation run side by side.
3. **Review**: the video, a timeline of every line coloured by speaker, and the full script. Click a timecode
   (or press Ctrl+Enter in a line) to hear that line in the original. Fix the Mongolian text or the speaker. Edits
   save automatically. The counter under each line turns red when the line is too long for its slot. You can also
   change voices and pitch per character.
4. **Dub the film**: shows how many lines need a voice and the ElevenLabs credits, and asks before spending.
   Lines that were already voiced are cached, so after an edit only the changed lines cost credits.
5. Switch the player to **Mongolian dub** to check it, then download.

Mongolian subtitles: the studio player always shows them (from the current script, so edits show at once). For
the final video pick one, on the upload form or next to the dub button (then rebuild the video):

- **Switchable**: a subtitle track viewers can turn off. Instant, the picture isn't re-encoded.
- **Always visible**: drawn into the picture, for TikTok, Shorts and Reels. English subtitles already burned into
  the video are found automatically and hidden under a dark bar with the Mongolian text on it. Re-encodes the
  video (on the GPU when there is one).
- **None**.

Watermark: open **Watermark** on a job page. Drag your logo anywhere on a frame of the video, or use a preset
(**Over the video's logo** hides the channel's own logo under yours, like a sticker). Set size, opacity, a blur and a dark
box behind it; **Hide the video's own logo** blurs the channel logo even when yours sits somewhere else. Save, then
**Rebuild the video** (no credits). **Change image** replaces the logo for all videos (PNG with transparency); until
then `watermark-nobackground.png` is used. Command line: `--watermark logo.png --watermark-style '{"auto": false,
"x": 0.8, "y": 0.06, "size": 0.3}'`.

On the command line: `--subs soft|burn|none`. `subtitles_mn.srt` in the work folder has the subtitles timed to the
dub.

With `DUB_PASSWORD` in `.env` the studio shows a sign-in page. To use it from other computers, set it and run
`python -m studio --host 0.0.0.0`. The server
refuses to listen on the network without a password, because anyone who can open the page can spend your credits.

Films dubbed earlier with the command line can be added to the studio:

```bash
python -m studio import --video film.mp4 --work work/film_folder --film films/betrayed.json
```

## The film profile

The translator reads this JSON with every chunk of the script. A good profile is the biggest lever on quality:

- `names`: English name to Mongolian spelling, e.g. `"Ethan": "Итан"`.
- `aliases`: who each character is and what they get called ("Dr. Montgomery", "Mommy").
- `address`: who uses та and who uses чи with whom, and family words (ээж, ах, эгч…).
- `glossary`: fixed terms, e.g. `"leukemia": "цусны хорт хавдар"`.
- `premise`, `genre`, `narrator`, `notes`: story context and anything else the translator should know.

See `films/betrayed.json` for a complete example. After changing the profile of a job, use **Retranslate the
script**. Your current script is kept as a backup file first.

## Command line

The studio runs `dub_video.py`, which also works on its own:

```bash
python dub_video.py film.mp4 --film films/betrayed.json --until cast   # everything up to review, no TTS spend
python dub_video.py film.mp4 --film films/betrayed.json                # the rest (asks before spending)
python dub_video.py --help
```

Translation speed: polish requests run in parallel alongside the draft (`--polish-workers`, default 6).
`--translate-workers 3` also translates three parts of the film at once, about 3 times faster, but names can
drift a little between parts. `--effort low` makes the model think less (faster, a bit rougher).
The studio has the same settings under Advanced settings.

## Layout

```
dub_video.py        pipeline entry point (stages, flags, resume)
dubflow/            the stages: media, transcribe, analyze, script (translation), cast, dub (TTS + mix)
studio/             web studio: app.py (API), jobs.py (job queue + runner), static/ (UI), desktop.py (app window)
films/              film profiles
voices/pool.json    voices that can be cast
data/  work/  cache/   job files, CLI work dirs, LLM and TTS caches (not in git)
```

`cache/tts` is shared by all jobs, so a line with the same text and voice is never paid for twice.
