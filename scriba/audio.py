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
from datetime import datetime
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
    # Voice Memos writes these into the container. They survive copying, which the
    # file's own timestamps do not: exporting a memo sets its modification time to
    # the moment of the copy, so a conversation from last week gets today's date.
    created: str = ""      # creation_time, ISO 8601
    memo_uuid: str = ""    # identifies the recording across renames and re-encodes
    device: str = ""       # the encoder tag names the phone or watch that recorded it


def _number(value, default: float) -> float:
    """ffprobe writes the string "N/A" where a value is missing.

    Raw ADTS and some streamed captures carry no duration at all, and a bare
    float() on that raised exactly the unreadable cast error the friendly
    messages in this module were written to replace.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def probe(path: Path) -> AudioInfo:
    probe_exe = _ffprobe()
    if not probe_exe:
        return AudioInfo(path, 0.0, 0, 0, "?")
    out = subprocess.run(
        [probe_exe, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        # An empty file, a download that stopped halfway, a .m4a that is really
        # something else. ffprobe knows what is wrong and says so on stderr, and that
        # sentence is more useful than the CalledProcessError that used to surface.
        detail = (out.stderr or "").strip().splitlines()
        raise RuntimeError(
            f"cannot read {path.name} as audio: "
            f"{detail[-1] if detail else 'ffprobe gave no reason'}"
        )
    data = json.loads(out.stdout)
    fmt = data.get("format", {})
    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}
    stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    return AudioInfo(
        path=path,
        duration=_number(fmt.get("duration"), 0.0),
        sample_rate=int(_number(stream.get("sample_rate"), 0)),
        channels=int(stream.get("channels", 0) or 0),
        codec=str(stream.get("codec_name", "?")),
        created=str(tags.get("creation_time", "")),
        memo_uuid=str(tags.get("voice_memo_uuid", "")),
        device=_recorder(str(tags.get("encoder", ""))),
    )


def _recorder(encoder: str) -> str:
    """The encoder tag, when it names a recording device rather than a converter.

    Voice Memos writes something like "com.apple.VoiceMemos (iPhone Version 18.2)",
    which is worth carrying into the document. Anything transcoded through ffmpeg
    carries "Lavf62.3.100" instead, which describes the last tool that touched the
    file and says nothing about where the conversation happened.
    """
    if not encoder or encoder.lower().startswith(("lavf", "lavc", "libav")):
        return ""
    return encoder


def recorded_at(info: AudioInfo) -> tuple[datetime | None, str]:
    """When the recording was made, and how confident we are about it.

    Three sources, worst case last. The container's `creation_time` is written by
    Voice Memos and travels with the file. The filesystem birth time survives a copy
    on the same machine and not much else. The modification time is the moment the
    file was last written, which for an exported memo is the export, and can sit days
    away from the conversation.

    Returning the source alongside the date matters: a document that states a date is
    making a claim, and the reader should be able to tell a recorded timestamp from a
    filesystem guess.
    """
    if info.created:
        try:
            raw = info.created.replace("Z", "+00:00")
            when = datetime.fromisoformat(raw)
            if when.tzinfo is not None:
                when = when.astimezone().replace(tzinfo=None)
            return when, "recorded"
        except ValueError:
            pass
    try:
        st = info.path.stat()
        birth = getattr(st, "st_birthtime", None)
        if birth:
            return datetime.fromtimestamp(birth), "file created"
        return datetime.fromtimestamp(st.st_mtime), "file modified"
    except OSError:
        return None, ""


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

    # Convert to a scratch name and rename once ffmpeg is done. Interrupt the
    # conversion halfway and what is left behind is a shorter WAV that opens fine,
    # decodes fine, and is nine tenths of the recording. Every stage downstream then
    # reuses it without a word, and the transcript is missing the end of the
    # conversation with nothing to show that anything went wrong. The rename is
    # atomic, so a job either has the whole audio or none of it.
    partial = dest.with_suffix(dest.suffix + ".part")
    cmd = [
        _ffmpeg(), "-y", "-nostdin", "-loglevel", "error",
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
    ]
    if filters:
        cmd += ["-af", ",".join(filters)]
    # State the container. ffmpeg picks the output format from the file extension, and
    # the scratch file ends in .part, which means nothing to it.
    cmd += ["-c:a", "pcm_s16le", "-f", "wav", str(partial)]

    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        partial.unlink(missing_ok=True)
        detail = (exc.stderr or "").strip().splitlines()
        raise RuntimeError(
            f"ffmpeg could not read {src.name}: "
            f"{detail[-1] if detail else f'exit code {exc.returncode}'}"
        ) from exc

    partial.replace(dest)
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
