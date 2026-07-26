"""Paths, secrets and default settings for scriba."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, asdict, field
from pathlib import Path

APP_NAME = "scriba"

HOME = Path.home()
DATA_DIR = Path(os.environ.get("SCRIBA_HOME", HOME / ".scriba"))
VOICES_DIR = DATA_DIR / "voices"
JOBS_DIR = DATA_DIR / "jobs"
SETTINGS_PATH = DATA_DIR / "settings.json"

KEYCHAIN_SERVICE = "scriba-hf-token"


# --------------------------------------------------------------------------- #
# Hugging Face token: Keychain > env > legacy file
# --------------------------------------------------------------------------- #

def keychain_get(service: str = KEYCHAIN_SERVICE) -> str | None:
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-w"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or None
    except subprocess.CalledProcessError:
        return None
    except FileNotFoundError:  # non-macOS
        return None


def keychain_set(token: str, service: str = KEYCHAIN_SERVICE) -> None:
    subprocess.run(
        ["security", "add-generic-password", "-U", "-a", os.getlogin(),
         "-s", service, "-w", token],
        check=True, capture_output=True,
    )


def hf_token() -> str | None:
    """The pyannote token. Never hardcoded: Keychain first, then environment variables."""
    return (
        keychain_get()
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or None
    )


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

@dataclass
class Settings:
    # ASR
    backend: str = "whisperx"          # whisperx | mlx
    model: str = "large-v3"
    # "auto" and not "it": a language pinned by default is the most expensive defect
    # this pipeline can have. Whisper does not fail on the wrong language, it
    # translates by ear and produces invented text that looks correct.
    language: str = "auto"
    compute_type: str = "int8"
    batch_size: int = 8
    threads: int = 0                   # 0 = auto (the fast cores)
    align: bool = True                 # forced alignment for per-word timestamps

    # anti-hallucination (see notes in asr.py)
    beam_size: int = 5
    temperature_fallback: bool = True
    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.6
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    vad_onset: float = 0.500
    vad_offset: float = 0.363
    initial_prompt: str | None = None
    hotwords: list[str] = field(default_factory=list)

    # diarization
    diarize: bool = True
    min_speakers: int | None = None
    max_speakers: int | None = None
    # "cpu" or "mps". MPS runs about 12x faster, and pays for it with timestamps
    # pyannote itself does not stand behind on Metal (pyannote-audio#1337, closed
    # with no fix). "cpu" here is a deliberate choice, not an oversight.
    diarize_device: str = "cpu"

    # Voice matching. These thresholds are not gospel, they are a starting point to
    # calibrate on YOUR recordings. The honest way to do it: annotate three or four
    # conversations by hand, look at where the same-person and different-person
    # similarity distributions cross, and put the threshold at that crossing.
    # Twenty minutes of annotated audio tells you more than any published number,
    # and for diarization in Italian there is no published number to argue with.
    voice_match_threshold: float = 0.75      # at or above this, the name assigns itself
    voice_suggest_threshold: float = 0.55    # between the two: it suggests, it does not decide
    voice_match_margin: float = 0.05         # minimum gap from the runner-up candidate
    voice_min_speech_sec: float = 8.0        # minimum speech before trusting the voice print

    # output
    output_formats: list[str] = field(
        default_factory=lambda: ["notebooklm", "md", "json", "srt", "vtt", "txt"]
    )
    timestamp_every: int = 0                 # 0 = timestamp on every turn

    @classmethod
    def load(cls) -> "Settings":
        if SETTINGS_PATH.exists():
            raw = json.loads(SETTINGS_PATH.read_text())
            known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
            return cls(**known)
        return cls()

    def save(self) -> None:
        ensure_dirs()
        SETTINGS_PATH.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))


def ensure_dirs() -> None:
    for d in (DATA_DIR, VOICES_DIR, JOBS_DIR):
        d.mkdir(parents=True, exist_ok=True)
