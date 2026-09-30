"""English transcript with word timings from ElevenLabs Scribe v2."""
import os

from .common import ffmpeg, load_json, save_json


def transcribe(audio, work, out_json):
    from elevenlabs import ElevenLabs

    upload = work / "scribe_upload.mp3"
    ffmpeg("-i", audio, "-ac", 1, "-b:a", "96k", upload)
    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    with upload.open("rb") as f:
        result = client.speech_to_text.convert(
            file=f, model_id="scribe_v2", diarize=True, tag_audio_events=True,
            timestamps_granularity="word", request_options={"timeout_in_seconds": 3600},
        )
    upload.unlink()
    save_json(out_json, result.model_dump())


def import_transcript(src_json, out_json, start=0.0, end=None):
    """Reuse an existing Scribe JSON, shifted/trimmed to a --start/--end test window."""
    data = load_json(src_json)
    words = []
    for w in data["words"]:
        if w["start"] is None or w["start"] < start or (end is not None and w["start"] >= end):
            continue
        words.append({**w, "start": w["start"] - start, "end": w["end"] - start})
    save_json(out_json, {**data, "words": words})
