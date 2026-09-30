"""Video in, stems out, video back: extract audio, split voice from background (Demucs), mux the dub."""
import numpy as np
import soundfile as sf
import torch

from .common import MIX_RATE, ffmpeg

SEP_CHUNK = 300.0  # seconds of audio per Demucs pass (bounds RAM on a 2-hour film)
SEP_PAD = 5.0  # context on both sides of a chunk, cut off afterwards so chunk edges are seamless


def cut(video, out, start, end):
    """Re-encoded excerpt of the video, for test runs on a few minutes."""
    ffmpeg("-ss", start, "-to", end, "-i", video, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-c:a", "aac", "-b:a", "192k", _part(out))
    _part(out).replace(out)


def _part(path):
    """Temp name next to path; stages write there and rename at the end, so a killed run never looks done."""
    return path.with_name(path.stem + ".part" + path.suffix)


def extract(video, out):
    ffmpeg("-i", video, "-vn", "-ac", 2, "-ar", MIX_RATE, "-c:a", "pcm_s16le", _part(out))
    _part(out).replace(out)


def separate(original, vocals_out, background_out, device):
    """Demucs htdemucs: vocals stem + everything else (music, effects) as background."""
    from demucs.api import Separator

    sep = Separator(model="htdemucs", device=device, progress=False)
    assert sep.samplerate == MIX_RATE
    info = sf.info(str(original))
    chunk, pad = int(SEP_CHUNK * MIX_RATE), int(SEP_PAD * MIX_RATE)
    with sf.SoundFile(str(original)) as src, \
            sf.SoundFile(str(_part(vocals_out)), "w", MIX_RATE, 2, "PCM_16") as voc, \
            sf.SoundFile(str(_part(background_out)), "w", MIX_RATE, 2, "PCM_16") as bg:
        for start in range(0, info.frames, chunk):
            a, b = max(0, start - pad), min(info.frames, start + chunk + pad)
            src.seek(a)
            x = torch.from_numpy(src.read(b - a, dtype="float32", always_2d=True).T.copy())
            _, stems = sep.separate_tensor(x, MIX_RATE)
            keep = slice(start - a, start - a + min(chunk, info.frames - start))
            vocals = stems["vocals"][:, keep]
            rest = torch.stack([s for name, s in stems.items() if name != "vocals"]).sum(0)[:, keep]
            voc.write(np.clip(vocals.T.numpy(), -1, 1))
            bg.write(np.clip(rest.T.numpy(), -1, 1))
            print(f"      separated {min(start + chunk, info.frames) / MIX_RATE / 60:5.1f} / "
                  f"{info.frames / MIX_RATE / 60:.1f} min", flush=True)
    _part(vocals_out).replace(vocals_out)
    _part(background_out).replace(background_out)
    del sep
    if device == "cuda":
        torch.cuda.empty_cache()


def mux(video, audio, out):
    """Original video stream (copied, no re-encode) with the dubbed audio."""
    ffmpeg("-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out)
