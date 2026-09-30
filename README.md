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

1. **New dub**: drop a video and pick or write a film profile (see below). Try a clip of a few minutes first.
2. Wait while the job runs. You can close the page, because the work runs on the server. Only one job runs at a
   time (one GPU); the others wait in a queue.
3. **Review**: the video, a timeline of every line coloured by speaker, and the full script. Click a timecode
   (or press Ctrl+Enter in a line) to hear that line in the original. Fix the Mongolian text or the speaker. Edits
   save automatically. The counter under each line turns red when the line is too long for its slot. You can also
   change voices and pitch per character.
4. **Dub the film**: shows how many lines need a voice and the ElevenLabs credits, and asks before spending.
   Lines that were already voiced are cached, so after an edit only the changed lines cost credits.
5. Switch the player to **Mongolian dub** to check it, then download.

To use it from other computers, set `DUB_PASSWORD` in `.env` and run `python -m studio --host 0.0.0.0`. The server
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

## Layout

```
dub_video.py        pipeline entry point (stages, flags, resume)
dubflow/            the stages: media, transcribe, analyze, script (translation), cast, dub (TTS + mix)
studio/             web studio: app.py (API), jobs.py (job queue + runner), static/ (UI)
films/              film profiles
voices/pool.json    voices that can be cast
data/  work/  cache/   job files, CLI work dirs, LLM and TTS caches (not in git)
```

`cache/tts` is shared by all jobs, so a line with the same text and voice is never paid for twice.
