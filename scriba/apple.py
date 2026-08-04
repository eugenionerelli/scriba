"""Transcription through Apple's on-device speech models.

Why this is here at all: ctranslate2, the engine under faster-whisper, has no
Metal backend, so on Apple Silicon whisper runs on the CPU for about as long as
the recording lasts. Apple's transcriber runs on the Neural Engine. Measured on
an M4, same 6m45s of Spanish conversation, same machine, same day:

    faster-whisper large-v3 int8      443 s
    Apple SpeechTranscriber             3 s

What it does NOT do is replace the alignment stage. Apple returns a time range
per word, and those ranges are contiguous: each word starts exactly where the
previous one ended, so on that recording 124 of the 182 seconds of silence
between words were swallowed into the words themselves. Average word duration
came out at 0.44 s against 0.28 s from forced alignment. Speaker attribution
is decided by overlap with the diarization, so those boundaries matter.

So the text comes from here and the boundaries keep coming from wav2vec2, which
is the stage scriba already ran. That combination measured better than either
side alone: word duration 0.223 s and 221 s of silence recognised, against
0.282 s and 182 s for the pipeline it replaces, because Apple finds more real
words and the aligner puts them where they belong.

Requires macOS 26, where the framework exists at all.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path

from .config import DATA_DIR

SOURCE = Path(__file__).resolve().parent.parent / "tools" / "apple-transcribe.swift"
BINARY = DATA_DIR / "bin" / "apple-transcribe"


def available() -> tuple[bool, str]:
    """Whether this backend can run here, and why not when it cannot."""
    if platform.system() != "Darwin":
        return False, "Apple's speech models exist only on macOS"
    major = int(platform.mac_ver()[0].split(".")[0] or 0)
    if major < 26:
        return False, (f"needs macOS 26 for the SpeechAnalyzer framework, this is "
                       f"macOS {platform.mac_ver()[0]}")
    if not BINARY.exists() and shutil.which("swiftc") is None:
        return False, ("needs the Swift compiler to build the helper once: install "
                       "the Xcode command line tools with `xcode-select --install`")
    if not SOURCE.exists() and not BINARY.exists():
        return False, f"the helper source is missing at {SOURCE}"
    return True, ""


def _build() -> Path:
    """Compile the helper once and keep it.

    Built here rather than shipped as a binary: it is 150 lines of Swift, the
    machine that runs scriba already has the compiler if it has Xcode's tools,
    and a binary in a git repository is a thing nobody can read.
    """
    if BINARY.exists() and BINARY.stat().st_mtime >= SOURCE.stat().st_mtime:
        return BINARY

    BINARY.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["swiftc", "-O", "-parse-as-library", str(SOURCE), "-o", str(BINARY)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not BINARY.exists():
        raise RuntimeError(
            "could not build the Apple speech helper.\n"
            + (proc.stderr.strip() or "swiftc said nothing")
        )
    return BINARY


def supported_languages() -> list[str]:
    """The language codes with an on-device model, without the region."""
    binary = _build()
    out = subprocess.run([str(binary), "--locales"], capture_output=True, text=True)
    if out.returncode != 0:
        return []
    codes = {line.split("_")[0].split("-")[0] for line in out.stdout.split() if line}
    return sorted(codes)


def transcribe(wav: Path, language: str) -> list[dict]:
    """Segments with start, end, text and the model's own confidence.

    The shape is whisper's, so the rest of the pipeline does not know or care
    which engine produced it. Confidence is the one thing whisper does not give:
    it travels through to the document, where a line the model was unsure of is
    worth marking.
    """
    ok, why = available()
    if not ok:
        raise RuntimeError(f"the Apple backend cannot run here: {why}")

    binary = _build()
    proc = subprocess.run(
        [str(binary), "--lang", language, str(wav)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "the Apple speech helper failed")

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"the Apple speech helper returned no JSON: {exc}") from exc

    return [
        {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip(),
         "confidence": s.get("confidence")}
        for s in payload.get("segments", [])
        if s.get("text", "").strip()
    ]
