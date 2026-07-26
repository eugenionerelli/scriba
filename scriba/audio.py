"""Audio preparation: whatever goes in, a clean 16 kHz mono WAV comes out.

Why it matters: whisper and pyannote both work at 16 kHz mono. Leave the conversion
to the library every time and you pay for decoding twice, with nowhere to normalize
the level. On voice memos recorded from across the room, that level gap is what hurts
diarization the most.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 16_000

AUDIO_EXTS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff", ".aif",
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm",
}


class FFmpegMissing(RuntimeError):
    pass


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FFmpegMissing("ffmpeg not found in PATH (brew install ffmpeg)")
    return exe


def _ffprobe() -> str | None:
    return shutil.which("ffprobe")


@dataclass
class AudioInfo:
    path: Path
    duration: float
    sample_rate: int
    channels: int
    codec: str


def probe(path: Path) -> AudioInfo:
    probe_exe = _ffprobe()
    if not probe_exe:
        return AudioInfo(path, 0.0, 0, 0, "?")
    out = subprocess.run(
        [probe_exe, "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    return AudioInfo(
        path=path,
        duration=float(data.get("format", {}).get("duration", 0.0) or 0.0),
        sample_rate=int(stream.get("sample_rate", 0) or 0),
        channels=int(stream.get("channels", 0) or 0),
        codec=str(stream.get("codec_name", "?")),
    )


def prepare(
    src: Path,
    dest: Path,
    *,
    normalize: bool = True,
    highpass: int = 60,
) -> Path:
    """Convert to 16 kHz mono PCM WAV.

    normalize: single-pass EBU R128 loudnorm. Not the two-pass version, which is more
        accurate and takes twice as long. For ASR that extra accuracy never reaches
        the text. What counts is closing the level gap between whoever sits next to
        the microphone and whoever sits across the room.
    highpass: cuts background noise below 60 Hz (traffic, fans, phone handling). The
        voice starts around 85 Hz, so it comes through untouched.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    filters = []
    if highpass:
        filters.append(f"highpass=f={highpass}")
    if normalize:
        filters.append("loudnorm=I=-18:TP=-2:LRA=11")

    cmd = [
        _ffmpeg(), "-y", "-nostdin", "-loglevel", "error",
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
    ]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-c:a", "pcm_s16le", str(dest)]

    subprocess.run(cmd, check=True)
    return dest


def slice_wav(src: Path, dest: Path, start: float, end: float) -> Path:
    """Extract a snippet. Used for the confirmation playback when you assign a name."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [_ffmpeg(), "-y", "-nostdin", "-loglevel", "error",
         "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dest)],
        check=True,
    )
    return dest


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTS and not path.name.startswith(".")
