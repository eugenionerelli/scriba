"""Multi-sample language detection.

Why this module exists: get the language wrong and Whisper does not fail. It
translates by ear, and the text comes out fluent, plausible, correctly punctuated
and completely made up. Spanish and Italian share too many near-identical words for
the guess to look like a guess: `salir` lands on `salire`, `éxito` on `esito`, and
every sentence reads as though somebody meant it. Nothing shows up on screen. That
silence is what makes it the most dangerous way this pipeline can go wrong: the
result *looks* right, and then gets read by someone, or something, as fact.

Whisper on its own looks at the first 30 seconds. Thirty seconds of small talk, or
of someone speaking the wrong language for a moment, and the call is made wrong for
the whole file. Here we sample across the whole duration and vote.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
WINDOW_SEC = 30


@dataclass
class LanguageGuess:
    language: str
    confidence: float
    votes: dict[str, float]
    samples: list[tuple[float, str, float]]     # (timestamp, language, probability)
    reliable: bool
    note: str


def detect(
    wav: Path,
    *,
    model_size: str = "small",
    n_samples: int = 5,
    compute_type: str = "int8",
    threads: int = 8,
) -> LanguageGuess:
    """Sample `n_samples` 30 s windows spread across the file and vote.

    We use `small` and not `large-v3`: language identification is an easy job,
    `small` gets there in a few seconds, and loading large-v3 for this alone would
    cost more than the entire transcription.
    """
    import soundfile as sf
    from faster_whisper import WhisperModel

    data, sr = sf.read(str(wav), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz, got {sr}: run audio.prepare() first")

    duration = len(data) / sr
    model = WhisperModel(model_size, device="cpu", compute_type=compute_type, cpu_threads=threads)

    # Stay away from the start and the end: the first and last handful of seconds are
    # almost always hiss, "ok, we're recording", or the phone being put down.
    fracs = np.linspace(0.05, 0.85, num=max(n_samples, 1))
    votes: dict[str, float] = defaultdict(float)
    samples: list[tuple[float, str, float]] = []

    for frac in fracs:
        a = int(frac * duration * sr)
        b = min(a + WINDOW_SEC * sr, len(data))
        if b - a < sr:            # less than a second: the window is useless
            continue
        lang, prob, _ = model.detect_language(audio=data[a:b])
        samples.append((a / sr, lang, float(prob)))
        # Vote weighted by probability: an uncertain window must not count as much
        # as a confident one. Below 0.5 the model is guessing.
        votes[lang] += float(prob) if prob >= 0.5 else float(prob) * 0.25

    del model

    if not votes:
        return LanguageGuess("", 0.0, {}, [], False, "audio too short to sample")

    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    winner, score = ranked[0]
    total = sum(votes.values()) or 1.0
    confidence = score / total
    runner_up = ranked[1][0] if len(ranked) > 1 else None

    reliable = confidence >= 0.6
    if not reliable:
        note = (f"language unclear between {winner} and {runner_up}: "
                "this may be a bilingual conversation. Check it by hand.")
    elif runner_up:
        note = f"{winner} prevails; some windows were classified as {runner_up}"
    else:
        note = f"{winner} across every window"

    return LanguageGuess(winner, confidence, dict(votes), samples, reliable, note)
