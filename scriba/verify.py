"""A second engine, asked the same question, to see where the first one is wrong.

A transcript gives no sign of being wrong. It reads the same whether the words are
right and in the right place or whether a whole sentence has been filed thirteen
seconds from where it was said. Nothing in the file distinguishes the two, and the
person reading it a month later has no way to tell either.

So: run the other engine over the same audio, put both texts through the *same*
forced aligner, and compare where each says the words are. Agreement means little
on its own. Disagreement is the useful half, because it points at a specific
second of audio and says: one of these two is wrong here, go and listen.

What this cost and what it caught, measured on a 6m45s conversation:

    whisperX large-v3, the transcript being checked        443 s
    Apple on-device model, the second opinion                6 s
    forced alignment of that second opinion                 10 s

Sixteen seconds against four hundred and forty-three, so the check is under 4% of
what the transcript already cost. With both sides through the same aligner the two
engines agreed to a median of 0.01 s, 97% of words within a second. What was left
was three disagreements, one of them real and worth the whole exercise: whisperX
had filed a whole sentence, with a figure in it, eleven seconds before it was
said. The audio where it put that sentence is somebody saying "no". Cutting those
seconds out of the file and reading them with the other engine settled it.

Why timing and not spelling: which words were said is a matter for a human ear,
and two engines disagreeing about a name proves nothing. But *when* is what decides
who is credited with saying it, because attribution is decided by overlap with the
diarization. A sentence in the wrong place can be handed to the wrong person, and
that is the one failure this tool cannot be allowed to have quietly.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
from typing import Any

# How far apart two engines must place the same word before it is worth a person's
# attention. Below a second is the aligner's own noise on either side. Two seconds
# is roughly a short sentence: far enough that the words could land in somebody
# else's turn.
DISAGREEMENT = 2.0

# Words closer together than this belong to the same disagreement. Without it a
# thirteen-second slip reads as eleven separate findings.
JOIN_WITHIN = 3.0


@dataclass
class Zone:
    """One stretch where the two engines disagree about when something was said."""
    start: float          # where the second engine puts it
    end: float
    other_start: float    # where the transcript being checked puts it
    words: int
    text: str
    changes_speaker: bool  # whether the two positions fall in different speakers' turns


@dataclass
class Silence:
    """A stretch where the second engine heard speech and the transcript has none.

    This is the failure that timing comparison misses entirely, because there is
    nothing to compare: the words are not late, they are absent. Measured shape,
    on a public benchmark clip of 21 seconds and 51 words, whisper large-v3 with
    this project's settings produced two words and stopped. Not a threshold and
    not a gate: float32 on the CPU and float16 on the GPU both produce the same
    two words, and only one particular numeric configuration escapes it. Nothing
    in the transcript says anything is wrong, and 21 seconds of a conversation
    are simply not in the document.
    """
    start: float
    end: float
    words_there: int   # how many words the second engine put in that stretch
    words_here: int    # how many the transcript has, usually zero
    text: str          # what the second engine heard, for somebody to judge


@dataclass
class Report:
    engine: str
    compared: int          # words both engines produced, in the same order
    only_here: int         # words the transcript has and the second engine does not
    only_there: int        # and the other way round
    median_offset: float
    within_one_second: float
    zones: list[Zone]
    silences: list[Silence] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.silences is None:
            self.silences = []

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)


def _tokens(text: str) -> list[str]:
    """Words stripped to what two engines can be expected to agree on.

    Case, accents and punctuation are where they differ for reasons nobody cares
    about: one writes "40 €", the other "40 euros", one capitalises after a full
    stop the other did not put there.
    """
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]", " ", text).split()


def _timed_words(segments: list[dict]) -> list[tuple[str, float]]:
    """Every word with a time, flattened. Words the aligner could not place are dropped.

    The Spanish alignment vocabulary has no digits, so "40" comes back with no
    start at all. Keeping it would mean comparing a word against a time that does
    not exist.
    """
    out: list[tuple[str, float]] = []
    for seg in segments:
        for w in (seg.get("words") or []):
            if w.get("start") is None:
                continue
            for token in _tokens(str(w.get("word", ""))):
                out.append((token, float(w["start"])))
    return out


def _speaker_at(turns: list[dict], t: float) -> str | None:
    for turn in turns:
        if float(turn["start"]) <= t <= float(turn["end"]):
            return turn.get("speaker")
    return None


# What counts as a stretch worth judging, and how far short the transcript has to
# fall inside it.
#
# The thresholds come from the two real dropouts rather than from taste. A 21-second
# Spanish passage of 51 words came back as 2, and a 15-second Italian one of 34 came
# back as 10. The second is why the bar is half and not a quarter: at a quarter the
# Italian case passed the check silently, which is the entire failure this is for.
#
# Against a false alarm: on a 6m45s conversation where the two engines genuinely
# agree, they produced 743 and 682 words, so ordinary disagreement runs around a
# tenth. Half is a long way outside that.
SILENT_SECONDS = 5.0
SILENT_WORDS = 8
SILENT_SHARE = 0.5

# Two utterances closer together than this are one passage. Without it a fifteen
# second dropout split into three utterances is three findings that each look too
# small to matter.
STRETCH_GAP = 1.5


def _silences(here: list[tuple[str, float]], there: list[dict]) -> list[Silence]:
    """Where the second engine hears a passage and the transcript has almost none of it.

    Judged over stretches of the second engine's own speech rather than a sliding
    window: an utterance is already the unit somebody would go and listen to, and a
    window straddling two of them reports half a sentence.
    """
    mine = sorted(t for _, t in here)

    def count_between(a: float, b: float) -> int:
        return sum(1 for t in mine if a <= t <= b)

    stretches: list[list[dict]] = []
    for seg in there:
        words = [w for w in (seg.get("words") or []) if w.get("start") is not None]
        if not words:
            continue
        if stretches and float(words[0]["start"]) - float(stretches[-1][-1]["end"]) <= STRETCH_GAP:
            stretches[-1].extend(words)
        else:
            stretches.append(list(words))

    found: list[Silence] = []
    for words in stretches:
        start, end = float(words[0]["start"]), float(words[-1]["end"])
        if len(words) < SILENT_WORDS or end - start < SILENT_SECONDS:
            continue
        # Half a second of slack at each end: a word starting just before the
        # stretch still covers it, and this is looking for absence.
        here_count = count_between(start - 0.5, end + 0.5)
        if here_count <= SILENT_SHARE * len(words):
            found.append(Silence(
                start=round(start, 1), end=round(end, 1),
                words_there=len(words), words_here=here_count,
                text=" ".join(str(w.get("word", "")).strip() for w in words)[:200]))
    return found


def compare(transcript: list[dict], second: list[dict], *,
            turns: list[dict] | None = None, engine: str = "",
            threshold: float = DISAGREEMENT) -> Report:
    """Where the two engines put the same words, and where they do not.

    Both sides must already have per-word times from the same aligner. Comparing
    an aligned transcript against raw engine output measures the aligner, not the
    engines: done that way the same recording appeared to disagree on 13% of its
    words, and with both sides aligned it was 0.8%.
    """
    a, b = _timed_words(second), _timed_words(transcript)
    matcher = SequenceMatcher(None, [t for t, _ in a], [t for t, _ in b], autojunk=False)
    blocks = matcher.get_matching_blocks()

    pairs: list[tuple[str, float, float]] = []
    for i, j, n in blocks:
        for k in range(n):
            pairs.append((a[i + k][0], a[i + k][1], b[j + k][1]))

    if not pairs:
        # No words in common is itself the loudest possible finding: it means one
        # of the two produced almost nothing. The silences below say where.
        return Report(engine=engine, compared=0, only_here=len(b), only_there=len(a),
                      median_offset=0.0, within_one_second=0.0, zones=[],
                      silences=_silences(b, second))

    offsets = [abs(x - y) for _, x, y in pairs]
    apart = [p for p in pairs if abs(p[1] - p[2]) > threshold]

    zones: list[Zone] = []
    for token, here, there in sorted(apart, key=lambda p: p[1]):
        if zones and here - zones[-1].end <= JOIN_WITHIN:
            z = zones[-1]
            z.end = here
            z.words += 1
            z.text = f"{z.text} {token}".strip()
        else:
            zones.append(Zone(start=here, end=here, other_start=there, words=1,
                              text=token, changes_speaker=False))

    if turns:
        for z in zones:
            z.changes_speaker = (_speaker_at(turns, z.start)
                                 != _speaker_at(turns, z.other_start))

    return Report(
        engine=engine,
        compared=len(pairs),
        only_here=len(b) - len(pairs),
        only_there=len(a) - len(pairs),
        median_offset=round(median(offsets), 3),
        within_one_second=round(sum(o <= 1.0 for o in offsets) / len(offsets), 3),
        zones=zones,
        # A stretch that looks empty because its words were filed eleven seconds
        # earlier is not missing, it is displaced, and the zone above already says
        # so. Reporting both makes one defect look like two, and the second one is
        # the more alarming of the pair.
        silences=[s for s in _silences(b, second)
                  if not any(z.start <= s.end and z.end >= s.start for z in zones)],
    )


def run(job_dir: Path, settings: Any, *, report=None) -> Report:
    """Transcribe again with the other engine and compare. Nothing is overwritten.

    The second opinion is cached beside the transcript: it costs seconds to make
    but the comparison is worth rerunning after a name is corrected or a threshold
    is moved, and there is no reason to pay for it twice.
    """
    from . import asr

    say = report or (lambda m: None)
    tpath = job_dir / "transcript.json"
    if not tpath.exists():
        raise FileNotFoundError(
            f"{job_dir.name} has no transcript yet: there is nothing to check.")

    cached = json.loads(tpath.read_text())
    transcript = cached["segments"] if isinstance(cached, dict) else cached
    if not any(s.get("words") for s in transcript):
        raise ValueError(
            "this transcript has no per-word times, so there is nothing to compare "
            "against. Run it again with alignment on.")

    other = "apple" if settings.backend != "apple" else "whisperx"
    second_path = job_dir / f"second-opinion-{other}.json"
    if second_path.exists():
        say(f"second opinion: reusing the {other} pass already on disk")
        second = json.loads(second_path.read_text())
    else:
        say(f"second opinion: transcribing again with {other}")
        s2 = type(settings)(**{**settings.__dict__, "backend": other, "align": True})
        s2.language = cached.get("language", settings.language) if isinstance(cached, dict) \
            else settings.language
        result = asr.transcribe(job_dir / "audio16k.wav", s2, progress=None)
        second = result.segments
        second_path.write_text(json.dumps(second, ensure_ascii=False, default=str))

    dpath = job_dir / "diarization.json"
    turns = json.loads(dpath.read_text()) if dpath.exists() else None
    if isinstance(turns, dict):
        turns = turns.get("turns")

    rep = compare(transcript, second, turns=turns, engine=other)
    (job_dir / "verification.json").write_text(rep.to_json())
    return rep
