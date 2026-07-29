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
from statistics import mean
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
WINDOW_SEC = 30

# Languages this model routinely mistakes for each other, grouped by the family
# the confusion happens inside. A unanimous vote between neighbours is worth less
# than a unanimous vote against the whole world, because the losing option was
# never really in the running: a Spanish conversation classified as Galician in
# every window comes back as Galician-flavoured Spanish, spelled in a way that
# reads as a transcription error rather than as the wrong language.
NEIGHBOURS: list[set[str]] = [
    {"es", "gl", "pt", "ca", "it"},          # romance, western
    {"it", "co", "la"},
    {"da", "no", "nn", "sv"},                # scandinavian
    {"cs", "sk"},
    {"hr", "sr", "bs", "sl"},
    {"ms", "id"},
    {"hi", "ur", "pa"},
    {"nl", "af"},
    {"ru", "uk", "be", "bg"},
]

# Below this average probability, a unanimous vote between neighbours is reported
# as a doubt rather than as a decision. Whisper's own scores in this band come out
# around 0.6 to 0.8 on genuinely ambiguous material, which is exactly where it
# picks the neighbour.
NEIGHBOUR_DOUBT = 0.85


def neighbours_of(language: str) -> list[str]:
    """The languages this one gets confused with, without itself."""
    found: set[str] = set()
    for family in NEIGHBOURS:
        if language in family:
            found |= family
    found.discard(language)
    return sorted(found)


@dataclass
class LanguageGuess:
    language: str
    confidence: float                           # agreement * strength
    votes: dict[str, float]
    samples: list[tuple[float, str, float]]     # (timestamp, language, probability)
    reliable: bool
    note: str
    # The two halves, kept apart because they fail differently and a single
    # number cannot say which one is low. Agreement is how many windows said the
    # same thing; strength is how sure those windows were.
    agreement: float = 0.0
    strength: float = 0.0


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
        try:
            lang, prob, _ = model.detect_language(audio=data[a:b])
        except Exception:
            # This module exists to survive a bad window, so it has to survive a
            # bad window. One that raises used to take the whole detection with
            # it, discarding the windows that had already voted.
            continue
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
    agreement = score / total
    runner_up = ranked[1][0] if len(ranked) > 1 else None

    # How sure the winning windows were, on their own terms. Agreement alone was
    # the whole confidence, and agreement is a share: when every window says the
    # same thing the share is exactly 1.0 however weak those windows were, so a
    # file nobody could identify came out as the most confident case there is,
    # and the header of the document said so. Five windows at 0.30 are five
    # guesses that happen to agree.
    strength = mean([p for _, lang, p in samples if lang == winner] or [0.0])
    confidence = agreement * strength

    near = neighbours_of(winner)
    unsure_neighbour = bool(near) and strength < NEIGHBOUR_DOUBT

    reliable = agreement >= 0.6 and strength >= 0.5 and not unsure_neighbour
    if agreement < 0.6:
        note = (f"language unclear between {winner} and {runner_up}: "
                "this may be a bilingual conversation. Check it by hand.")
    elif strength < 0.5:
        # "in every window" only when it was every window. The runner-up sits in
        # `samples` next to this sentence, and a note that talks past it is the
        # same overstatement this module exists to avoid.
        where = "in most windows" if runner_up else "in every window"
        note = (f"{winner} {where}, but the model was unsure in each of them "
                f"(average {strength:.0%}). Say the language yourself if you know it.")
    elif unsure_neighbour:
        note = (f"{winner} at {strength:.0%} average, which is not high enough to "
                f"separate it from {', '.join(near)}. These get confused with each "
                f"other, and the transcript then comes out in a mixture. Pass "
                f"--lang if you know which it is.")
    elif runner_up:
        note = f"{winner} prevails; some windows were classified as {runner_up}"
    else:
        note = f"{winner} across every window"

    return LanguageGuess(winner, confidence, dict(votes), samples, reliable, note,
                         agreement=agreement, strength=strength)
