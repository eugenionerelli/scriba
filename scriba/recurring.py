"""Find the voice that keeps coming back, and calibrate the thresholds on it.

Most of somebody's own recordings have their own voice in them. That fact is worth
more than it looks, because it solves two problems at once and neither needs anything
to be labelled by hand.

The first is enrollment. Naming yourself once per recording is the tax the voice
registry was built to remove, and the registry still has to be told who you are to
begin with. It does not: across a pile of recordings, the voice present in the most
of them is the person who owns the microphone.

The second is the calibration nobody ever does. Every threshold in voices.py was
picked from a briefing, and the honest way to set them is to compare two recordings
of the same person made on different days. That comparison is exactly what a scan of
an existing corpus produces, for free, in the hundreds.

The expensive stage is skipped entirely. Working out who is in a recording needs
diarization and embeddings, not a transcript. On Metal that is about a tenth of
realtime, so hours of audio can be scanned in minutes.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import is_audio
from .config import Settings
from .pipeline import Job
from .voices import cosine, l2norm


@dataclass
class Sample:
    """One speaker in one recording."""
    file: Path
    label: str
    embedding: np.ndarray
    speech_seconds: float
    longest_start: float
    longest_end: float


@dataclass
class Cluster:
    samples: list[Sample] = field(default_factory=list)

    @property
    def files(self) -> set[Path]:
        return {s.file for s in self.samples}

    @property
    def speech_seconds(self) -> float:
        return sum(s.speech_seconds for s in self.samples)

    def centroid(self) -> np.ndarray:
        return l2norm(l2norm(np.vstack([s.embedding for s in self.samples])).mean(axis=0))


# Voice recordings, not video. A folder of work material holds screen recordings and
# exported footage alongside the memos, and the first run of this on a real folder
# picked up 12 GB of video. Video can hold a conversation worth scanning, so it is
# reachable, and it is not what someone means by "scan my voice memos".
VOICE_EXTS = {".m4a", ".wav", ".mp3", ".aac", ".flac", ".aiff", ".aif", ".opus", ".ogg", ".wma"}


def scan(
    folder: Path,
    settings: Settings | None = None,
    *,
    min_speech: float = 20.0,
    include_video: bool = False,
    report=print,
) -> list[Sample]:
    """Diarize every recording in a folder and keep one embedding per speaker.

    Uses the ordinary job cache, so a second scan costs nothing and a recording that
    was already transcribed is not touched again.

    `min_speech` drops speakers who barely say anything. Those embeddings are built
    from too little audio to mean much, and letting them into the clustering pulls
    centroids around.
    """
    s = settings or Settings.load()
    s.diarize = True
    everything = [p for p in folder.rglob("*") if p.is_file() and is_audio(p)]
    files = sorted(p for p in everything
                   if include_video or p.suffix.lower() in VOICE_EXTS)
    skipped = len(everything) - len(files)
    if not files:
        raise RuntimeError(
            f"no voice recordings under {folder}"
            + (f" ({skipped} video files were skipped, pass --include-video for those)"
               if skipped else "")
        )

    report(f"{len(files)} recordings to scan. Only diarization runs, no transcription."
           + (f" {skipped} video files skipped." if skipped else ""))
    samples: list[Sample] = []

    for n, path in enumerate(files, 1):
        try:
            job = Job(path, s, report=lambda m: None)
            job._drop_stale_cache()
            job.prepare_audio()
            dia = job.diarize()
        except Exception as exc:
            report(f"  [{n}/{len(files)}] {path.name}: skipped ({exc})")
            continue

        speech = dia.speech_time()
        kept = 0
        for label, vec in dia.embeddings.items():
            if speech.get(label, 0.0) < min_speech:
                continue
            longest = dia.longest_turns(label, n=1)
            samples.append(Sample(
                file=path, label=label, embedding=np.asarray(vec, dtype=np.float32),
                speech_seconds=speech.get(label, 0.0),
                longest_start=longest[0].start if longest else 0.0,
                longest_end=longest[0].end if longest else 0.0,
            ))
            kept += 1
        report(f"  [{n}/{len(files)}] {path.name}: {len(dia.speakers())} voices, {kept} usable")

    return samples


def cross_file_similarities(samples: list[Sample]) -> np.ndarray:
    """Similarities between speakers recorded in different files.

    Pairs from inside one recording are excluded on purpose. Two speakers in the same
    room were separated by the diarizer precisely because they sound different, so
    those pairs say nothing about whether one person sounds like themselves on another
    day, which is the question the thresholds have to answer.
    """
    out = []
    for a, b in itertools.combinations(samples, 2):
        if a.file == b.file:
            continue
        out.append(float(cosine(a.embedding, b.embedding[None, :]).reshape(-1)[0]))
    return np.array(sorted(out))


def suggest_threshold(sims: np.ndarray) -> tuple[float, str]:
    """Where the same-person and different-person similarities separate.

    The question is not "split these in two", it is "which of these are unusual". Most
    pairs in a corpus are two strangers, and those pile up in a narrow band. The pairs
    that matter are the few sitting far above that band. So the cut is an outlier
    boundary: the middle of the stranger band, plus several deviations measured in a
    way that outliers cannot inflate.

    Two earlier attempts are worth recording, because both looked reasonable.

    Widest gap in the upper tail. Fails on real material: one person recorded across
    many days spreads from 0.4 to 0.9 rather than clustering, and the widest gap in
    that spread falls inside their own voice and cuts it in half.

    Otsu's method. Fails for a documented reason. Stranger pairs form a dense mass
    around zero, and Otsu's weighting term is largest when the two classes are equal
    in size, so it splits that mass down the middle and returns something near 0.08,
    which calls half the strangers a match.

    Median and MAD rather than mean and standard deviation: the recurring voice is
    itself in the sample, and it would drag both of those upward.
    """
    if len(sims) < 30:
        return 0.0, f"only {len(sims)} cross-file pairs, too few to say anything"

    median = float(np.median(sims))
    mad = float(np.median(np.abs(sims - median)))
    # 1.4826 rescales the median absolute deviation to match a standard deviation on
    # normal data, which is what the stranger band looks like.
    spread = mad * 1.4826
    if spread < 1e-6:
        # More than half the pairs sitting on the same value flattens the MAD to
        # zero, which happens with duplicate copies of one recording in a folder.
        # The old message said every pair was identical, which is false as soon as
        # anything stands above them, and the refusal that follows had the reader
        # looking for the wrong problem.
        # Anything off the flat value, above or below it. Counting only what sits
        # above assumed the background is the lower group, which it need not be.
        outliers = int((np.abs(sims - median) > 1e-6).sum())
        if outliers:
            return 0.0, (f"{outliers} of {len(sims)} pairs stand apart from a background "
                         f"that is otherwise a single value ({median:.2f}), so the "
                         "spread cannot be measured. Duplicate copies of one "
                         "recording do this: remove them and run it again")
        return 0.0, "every pair is identical, so there is nothing to separate"

    cut = median + 5.0 * spread
    above = sims[sims > cut]

    if len(above) == 0:
        return 0.0, (f"no pair stands out from the background (everything sits within "
                     f"5 deviations of {median:.2f}), so no voice repeats here")
    distance = (float(above.min()) - median) / spread
    return cut, (f"{len(above)} of {len(sims)} pairs stand out, the closest at "
                 f"{distance:.0f} deviations above the {median:.2f} background")


MIN_USABLE_THRESHOLD = 0.30


def cluster(samples: list[Sample], threshold: float) -> list[Cluster]:
    """Group speakers who sound like the same person across recordings.

    Single linkage over the graph of pairs above `threshold`. Single linkage chains,
    which is usually a flaw and is right here: one person recorded in a quiet room and
    again in a corridor may not look similar directly, while both look similar to a
    third recording in between.
    """
    if threshold < MIN_USABLE_THRESHOLD:
        # Two ways to arrive here, and both mean the same thing. Either the split
        # search gave up and returned its sentinel, or it found a cut that separates
        # this particular corpus while sitting far too low to mean "same person".
        # Different speakers routinely reach 0.2 against each other, so anything
        # under 0.3 groups strangers together. Clustering at zero is worse still: it
        # joins everyone to everyone and returns one group holding the whole corpus,
        # which on screen reads exactly like a confident answer.
        raise ValueError(
            f"the data suggests a cut at {threshold:.2f}, which is below the {MIN_USABLE_THRESHOLD:.2f} "
            "floor for calling two voices the same person. Either no voice recurs "
            "here, or the recordings are too unalike to link"
        )

    n = len(samples)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in itertools.combinations(range(n), 2):
        if samples[i].file == samples[j].file:
            continue
        sim = float(cosine(samples[i].embedding, samples[j].embedding[None, :]).reshape(-1)[0])
        if sim >= threshold:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

    groups: dict[int, Cluster] = {}
    for i, sample in enumerate(samples):
        groups.setdefault(find(i), Cluster()).samples.append(sample)

    # Most recordings first, then most speech. A voice in eight files beats a voice
    # that talks for an hour in one.
    return sorted(groups.values(),
                  key=lambda c: (len(c.files), c.speech_seconds), reverse=True)
