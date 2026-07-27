"""Diarization with pyannote, called directly instead of through whisperx.

Why the bypass: `whisperx.DiarizationPipeline` throws the embeddings away. pyannote 3.3
computes them anyway (they are how it decides who is who) and with
`return_embeddings=True` hands them back for free, one centroid per speaker.
Those are the same vectors that feed the voice registry in voices.py: without them,
name attribution would have to start from scratch on every recording.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


def _default_model() -> str:
    """community-1 under pyannote 4, 3.1 under pyannote 3.

    Measured on 6:45 of Spanish, two speakers, both models inside pyannote 4.0.7.
    They agree almost entirely: diarization error rate between the two segmentations
    is 0.047, and both land on the same two speakers when told there are two.

    The reason to prefer community-1 is what happens when nobody tells it. Asked to
    work the speaker count out on its own it finds two, which is right. 3.1 finds
    three, and the third is an artifact: 32 fragments, most under half a second,
    scattered across a stretch where one person is leafing through paperwork. That
    invented speaker is the defect this project keeps running into, and it matters
    because the whole point is that you should not have to know how many people were
    in the room before you can transcribe them.

    Speed is not the reason. Inside pyannote 4, 3.1 takes 211s and community-1 takes
    218s on this file. The jump from 372s came from pyannote 4 itself, running the
    same 3.1 model, not from the new one.
    """
    try:
        import pyannote.audio
        major = int(str(pyannote.audio.__version__).split(".")[0])
    except Exception:
        major = 3
    return ("pyannote/speaker-diarization-community-1" if major >= 4
            else "pyannote/speaker-diarization-3.1")


DIARIZATION_MODEL = os.environ.get("SCRIBA_DIARIZATION_MODEL") or _default_model()

# The telemetry opt-out lives in scriba/__init__.py, which runs before any import
# path can reach pyannote. Leaving it here as well would read as though this were
# the place that mattered, and it is not: pyannote checks the variable once, at
# import time.


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Diarization:
    turns: list[SpeakerTurn]
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)

    # pyannote 4 returns a second segmentation with the overlaps removed, built for
    # lining diarization up against a transcript. Empty on pyannote 3.
    #
    # It is kept and not used. Switching `assign` over to it moves 33 words out of
    # 680 on the reference file. Reading all 33: about fourteen get better, eight get
    # worse, the rest are arguable. It rescues the cases where the regular
    # segmentation drops one speaker's word into the middle of someone else's
    # sentence, and it invents new versions of the same mistake elsewhere.
    #
    # A 2% shuffle with no ground truth to score it against is not an improvement,
    # it is a coin flip. The data is here for whoever annotates a file by hand and
    # can settle it.
    exclusive_turns: list[SpeakerTurn] = field(default_factory=list)

    def speakers(self) -> list[str]:
        return sorted({t.speaker for t in self.turns})

    def speech_time(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for t in self.turns:
            out[t.speaker] = out.get(t.speaker, 0.0) + t.duration
        return out

    def longest_turns(self, speaker: str, n: int = 3) -> list[SpeakerTurn]:
        turns = [t for t in self.turns if t.speaker == speaker]
        return sorted(turns, key=lambda t: t.duration, reverse=True)[:n]


def run(
    wav: Path,
    *,
    hf_token: str | None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    device: str = "auto",
) -> Diarization:
    import torch
    import torchaudio
    from pyannote.audio import Pipeline

    device = _pick_device(device)

    # The keyword changed name between pyannote 3 and 4. Try the current one, fall
    # back to the old one, so the same code runs against either installed version.
    try:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=hf_token)

    if pipeline is None:
        raise RuntimeError(
            f"pyannote did not return the '{DIARIZATION_MODEL}' pipeline. "
            "It is almost always the gating: accept the terms at "
            f"https://hf.co/{DIARIZATION_MODEL} and at "
            "https://hf.co/pyannote/segmentation-3.0 with the same account as the token."
        )
    pipeline.to(torch.device(device))

    # Loading into memory instead of passing the path keeps pyannote from decoding the
    # file a second time, and guarantees that ASR and diarization see exactly the same
    # audio. Once the two decodes diverge, the timestamps stop lining up.
    waveform, sr = torchaudio.load(str(wav))
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio = {"waveform": waveform, "sample_rate": sr}
    kwargs = {"min_speakers": min_speakers, "max_speakers": max_speakers}

    if _returns_dataclass(pipeline):
        result = pipeline(audio, **kwargs)
    else:
        # pyannote 3 hands the embeddings back only when asked, and hands them back
        # as the second half of a tuple.
        result = pipeline(audio, return_embeddings=True, **kwargs)

    annotation, centroids, exclusive = _unpack(result)

    turns = _to_turns(annotation)
    exclusive_turns = _to_turns(exclusive) if exclusive is not None else []

    embeddings: dict[str, np.ndarray] = {}
    if centroids is not None:
        for label, vec in zip(annotation.labels(), np.asarray(centroids)):
            # pyannote fills in a zero vector for any speaker it has no real centroid
            # for. That happens when the speaker count asked for is above the number
            # of clusters it found. A zero vector is not a voice print: drop it, or it
            # lands in the voice registry and from there poisons every future match.
            if np.all(np.isfinite(vec)) and float(np.linalg.norm(vec)) > 1e-6:
                embeddings[str(label)] = np.asarray(vec, dtype=np.float32)

    return Diarization(turns=turns, embeddings=embeddings,
                       exclusive_turns=exclusive_turns)


def _pick_device(requested: str) -> str:
    """Resolve "auto" into a real device, and refuse to fail quietly.

    Asking for Metal on a machine without it used to be an error thrown from deep
    inside torch. Here it falls back to the CPU and says so, because a diarization
    that runs slowly is a nuisance and one that does not run at all costs the whole
    job.
    """
    if requested not in ("auto", "mps"):
        return requested
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    if requested == "mps":
        print("[scriba] Metal was asked for and this machine has none. Using the CPU.")
    return "cpu"


def _returns_dataclass(pipeline) -> bool:
    """Does this pyannote return a DiarizeOutput, or the old Annotation and tuple?

    Asked of the object rather than of a version string. A version check would go
    stale at the next release; this goes stale only if pyannote renames the field,
    at which point the unpacking below has to change anyway.
    """
    import pyannote.audio
    return int(str(pyannote.audio.__version__).split(".")[0]) >= 4


def _unpack(result):
    """Flatten every shape pyannote has returned into (annotation, centroids, exclusive).

    Three shapes exist in the wild. pyannote 4 returns a DiarizeOutput dataclass.
    pyannote 3 with return_embeddings returns a two-tuple. pyannote 3 without it
    returns a bare Annotation. Handling all three keeps one code path working against
    whichever version is installed, which matters because the two live in separate
    conda environments here.
    """
    ann = getattr(result, "speaker_diarization", None)
    if ann is not None:
        return (ann,
                getattr(result, "speaker_embeddings", None),
                getattr(result, "exclusive_speaker_diarization", None))
    if isinstance(result, tuple):
        annotation, centroids = result
        return annotation, centroids, None
    return result, None, None


def _to_turns(annotation) -> list[SpeakerTurn]:
    turns = [
        SpeakerTurn(start=float(seg.start), end=float(seg.end), speaker=str(label))
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t.start)
    return turns


# --------------------------------------------------------------------------- #
# Assigning speakers to words
# --------------------------------------------------------------------------- #

def assign(segments: list[dict], dia: Diarization, *, word_level: bool = True) -> list[dict]:
    """Attach a speaker to every word and to every segment.

    Compared to `whisperx.assign_word_speakers`: here a word that overlaps no turn is
    neither left orphaned nor handed to the "nearest" turn in absolute terms (which
    can be very far away). It inherits from the previous word if that word ended
    within half a second, otherwise it stays without one. Orphan words are the main
    reason diarized transcripts end up with one-word turns attributed to the wrong
    person.
    """
    turns = dia.turns
    if not turns:
        return segments

    starts = np.array([t.start for t in turns])
    ends = np.array([t.end for t in turns])
    labels = [t.speaker for t in turns]

    def best_label(a: float, b: float) -> tuple[str | None, float]:
        inter = np.minimum(ends, b) - np.maximum(starts, a)
        hit = inter > 0
        if not hit.any():
            return None, 0.0
        totals: dict[str, float] = {}
        for i in np.nonzero(hit)[0]:
            totals[labels[i]] = totals.get(labels[i], 0.0) + float(inter[i])
        label = max(totals, key=totals.get)
        span = max(b - a, 1e-6)
        return label, totals[label] / span

    for seg in segments:
        s_start = float(seg.get("start", 0.0))
        s_end = float(seg.get("end", s_start))
        label, conf = best_label(s_start, s_end)
        seg["speaker"] = label
        seg["speaker_confidence"] = round(conf, 3)

        if word_level and seg.get("words"):
            last_label: str | None = None
            last_end: float | None = None
            for w in seg["words"]:
                if "start" not in w or "end" not in w:
                    # Words with no timestamp: happens with numbers and symbols the
                    # aligner cannot map. They inherit from the context.
                    w["speaker"] = last_label or label
                    continue
                wl, _ = best_label(float(w["start"]), float(w["end"]))
                if wl is None:
                    gap = float(w["start"]) - last_end if last_end is not None else 99.0
                    wl = last_label if (last_label and gap <= 0.5) else label
                w["speaker"] = wl
                last_label, last_end = wl, float(w["end"])

    return segments


def to_turns(segments: list[dict], *, max_gap: float = 2.0) -> list[dict]:
    """Recompose the segments into actual conversation turns.

    whisper cuts every ~30 s regardless of who is talking, so its "segments" are not
    turns. Here consecutive segments from the same speaker separated by less than
    `max_gap` seconds are merged: that is what makes the transcript readable by a
    human (and by an LLM) instead of a list of fragments.
    """
    turns: list[dict] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        spk = seg.get("speaker")
        conf = float(seg.get("speaker_confidence", 1.0))
        if turns and turns[-1]["speaker"] == spk and \
                float(seg.get("start", 0)) - turns[-1]["end"] <= max_gap:
            turns[-1]["text"] += " " + text
            turns[-1]["end"] = float(seg.get("end", turns[-1]["end"]))
            # The weakest segment sets the confidence for the merged turn. Averaging
            # would let one solid minute bury the doubtful sentence inside it, and the
            # doubtful sentence is the whole reason for carrying the number.
            turns[-1]["confidence"] = min(turns[-1]["confidence"], conf)
        else:
            turns.append({
                "speaker": spk,
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": text,
                "confidence": conf,
            })
    return turns
