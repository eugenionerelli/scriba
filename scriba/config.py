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


def write_atomic(path: Path, data: str | bytes) -> None:
    """Write through a scratch file and rename.

    A process killed between opening a file and finishing it leaves a truncated
    file that the next run fails to parse, or worse, parses into something
    plausible. The rename is atomic, so a reader sees the old file or the new
    one and never half of either.
    """
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        if isinstance(data, bytes):
            tmp.write_bytes(data)
        else:
            tmp.write_text(data)
        tmp.replace(path)
    except BaseException:
        # Leave nothing behind. A half-written scratch file that nobody reads is
        # harmless right up to the moment somebody goes looking for why a job
        # folder is bigger than it should be.
        tmp.unlink(missing_ok=True)
        raise


# The output formats export.write_all knows how to write. Anything else in
# output_formats is a typo that would silently produce one file fewer.
KNOWN_FORMATS = {"source", "md", "txt", "srt", "vtt", "json"}


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
    # "auto", "mps" or "cpu". "auto" picks Metal when the machine has it.
    #
    # Metal was off by default here for a while, on the strength of
    # pyannote-audio#1337, which reports wrong timestamps under MPS and was closed
    # with no fix. Measured instead of assumed, on pyannote 4.0.7 with
    # community-1 on an M4: 36s against 226s, and the two outputs are identical.
    # Same 165 turns, same labels, and every start and end matching to the
    # millisecond. The report does not reproduce on this combination.
    #
    # One file on one machine is not proof for every machine, so `cpu` stays
    # reachable, and the engine says which device it used on every run. If a
    # transcript ever comes out with one speaker owning the whole recording, that
    # is the shape #1337 describes: set this to "cpu" and compare.
    diarize_device: str = "auto"

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
        default_factory=lambda: ["source", "md", "json", "srt", "vtt", "txt"]
    )
    timestamp_every: int = 0                 # 0 = timestamp on every turn

    def validate(self) -> None:
        """Reject values that would break something later, quietly and permanently.

        Every check here stands for a way the tool used to accept nonsense and then
        misbehave a long way from the cause. A threshold of 5 is silently stored, and
        from then on the voice registry never matches anyone again, with no error to
        connect the two. `output_formats` given as a string makes write_all iterate
        over its characters and produce no files at all, after the eight minutes of
        transcription have already been spent. A non-numeric speaker count crashes
        inside pyannote, again after the transcription.
        """
        problems: list[str] = []

        def numeric(name: str) -> bool:
            v = getattr(self, name)
            return isinstance(v, (int, float)) and not isinstance(v, bool)

        for field_name in ("voice_match_threshold", "voice_suggest_threshold",
                           "voice_match_margin"):
            v = getattr(self, field_name)
            if not numeric(field_name) or not -1.0 <= float(v) <= 1.0:
                problems.append(f"{field_name}={v!r}: cosine similarity lives in [-1, 1]")
        # Only compare once both sides are known to be numbers. Comparing them
        # anyway raised TypeError from inside a method whose whole job is to
        # raise a readable ValueError, and since nearly every command starts with
        # Settings.load(), one null hand-edited into settings.json took the tool
        # down with a traceback instead of naming the field.
        if (numeric("voice_suggest_threshold") and numeric("voice_match_threshold")
                and self.voice_suggest_threshold > self.voice_match_threshold):
            problems.append(
                f"voice_suggest_threshold ({self.voice_suggest_threshold}) is above "
                f"voice_match_threshold ({self.voice_match_threshold}), so the zone "
                "that suggests a name sits above the zone that applies one")

        for field_name in ("min_speakers", "max_speakers"):
            v = getattr(self, field_name)
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
                problems.append(f"{field_name}={v!r}: a speaker count is a whole number, 1 or more")
        if (isinstance(self.min_speakers, int) and isinstance(self.max_speakers, int)
                and not isinstance(self.min_speakers, bool)
                and self.min_speakers > self.max_speakers):
            problems.append(f"min_speakers ({self.min_speakers}) is above "
                            f"max_speakers ({self.max_speakers})")

        if not isinstance(self.output_formats, list) or not self.output_formats:
            problems.append(f"output_formats={self.output_formats!r}: expected a list of names")
        else:
            unknown = [f for f in self.output_formats if f not in KNOWN_FORMATS]
            if unknown:
                problems.append(f"output_formats: {', '.join(unknown)} "
                                f"(known: {', '.join(sorted(KNOWN_FORMATS))})")

        if not isinstance(self.hotwords, list):
            problems.append(f"hotwords={self.hotwords!r}: expected a list of words")
        if self.diarize_device not in ("auto", "cpu", "mps"):
            problems.append(f"diarize_device={self.diarize_device!r}: expected auto, cpu or mps")
        if self.voice_min_speech_sec < 0:
            problems.append(f"voice_min_speech_sec={self.voice_min_speech_sec}: cannot be negative")

        if problems:
            raise ValueError("settings out of range:\n  " + "\n  ".join(problems))

    @classmethod
    def load(cls) -> "Settings":
        if SETTINGS_PATH.exists():
            raw = json.loads(SETTINGS_PATH.read_text())
            known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
            # The main output format was once named after the tool it was written
            # for. Renaming it would turn every stored settings file into a hard
            # error about an unknown format, so translate it instead.
            formats = known.get("output_formats")
            if isinstance(formats, list):
                known["output_formats"] = ["source" if f == "notebooklm" else f
                                           for f in formats]
            s = cls(**known)
            s.validate()
            return s
        return cls()

    def save(self) -> None:
        self.validate()
        ensure_dirs()
        SETTINGS_PATH.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))


def ensure_dirs() -> None:
    for d in (DATA_DIR, VOICES_DIR, JOBS_DIR):
        d.mkdir(parents=True, exist_ok=True)
