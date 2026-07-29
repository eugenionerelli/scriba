"""The middle of the pipeline: a speaker label turning into a name in a document.

Every other file in this suite aims at an edge. `test_voices` proves the matching
policy decides correctly, `test_export` proves the writer formats correctly, and
between the two sits `Job.run`, which carries a verdict from one to the other:
match -> profile -> names -> meta["unresolved"] -> the written file. Nothing tested
that carry. A change that promoted a borderline candidate to a confirmed name would
have left both edges green while putting somebody's name on words they may not have
said, which is the single worst thing this tool can do.

So these tests run the real `Job.run()` with only the expensive parts faked, and
then assert on the finished document rather than on the dictionaries that produced
it. A `Voice 1` in the transcript is not evidence of anything if the header
announced a name; only the file as a whole is.

Nothing here loads whisper or pyannote, decodes audio or reaches the network:
ffmpeg, the ASR and the diarizer are replaced, and the registry verdicts are
handed in by the test rather than computed from vectors. Every path lives under
tmp_path. Every personal name below is invented.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scriba import asr, audio, config, diarize as diarize_mod, export, lang, pipeline
from scriba import voices as V
from scriba.config import Settings

# Small enough to read in a failure message, and the value carries the label so a
# stubbed match can tell which voice it was handed.
DIM = 8
DURATION = 180.0
RECORDED = "2026-03-04T10:15:00Z"

# Invented people. None of these is anybody.
CERTAIN_NAME = "Tobias Venn"
CANDIDATE_NAME = "Marisol Adeyemi"
THIRD_NAME = "Ada Nkemelu"
HAND_NAME = "Ines Kovac"


# --------------------------------------------------------------------------- #
# isolation
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """Point every path the pipeline and the registry know about at tmp_path.

    Two import-time bindings have to be rebound by hand: `pipeline` did
    `from .config import JOBS_DIR`, and `voices` computed REGISTRY_PATH and
    EMB_PATH from config.VOICES_DIR. Setting SCRIBA_HOME alone would leave both
    writing into the user's real home, and these tests do enroll voices.
    """
    root = tmp_path / "scriba-home"
    jobs = root / "jobs"
    voices_dir = root / "voices"
    jobs.mkdir(parents=True)
    voices_dir.mkdir(parents=True)

    monkeypatch.setenv("SCRIBA_HOME", str(root))
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(config, "VOICES_DIR", voices_dir)
    monkeypatch.setattr(config, "JOBS_DIR", jobs)
    monkeypatch.setattr(config, "SETTINGS_PATH", root / "settings.json")
    monkeypatch.setattr(pipeline, "JOBS_DIR", jobs)
    monkeypatch.setattr(V, "VOICES_DIR", voices_dir)
    monkeypatch.setattr(V, "REGISTRY_PATH", voices_dir / "registry.json")
    monkeypatch.setattr(V, "EMB_PATH", voices_dir / "embeddings.npy")

    # Checked rather than assumed: a patch that missed would write voice prints
    # into the registry of whoever is running the suite.
    for p in (config.DATA_DIR, config.VOICES_DIR, config.JOBS_DIR,
              config.SETTINGS_PATH, pipeline.JOBS_DIR,
              V.VOICES_DIR, V.REGISTRY_PATH, V.EMB_PATH):
        assert Path(p).is_relative_to(tmp_path)
    return root


@pytest.fixture(autouse=True)
def _no_models(monkeypatch):
    """Any test here that reaches a model is a broken test, not a slow one."""
    def forbidden(*a, **kw):
        raise AssertionError("a test tried to run a real model")

    monkeypatch.setattr(asr, "transcribe", forbidden)
    monkeypatch.setattr(diarize_mod, "run", forbidden)
    monkeypatch.setattr(lang, "detect", forbidden)
    monkeypatch.setattr(pipeline, "hf_token", lambda: None)


@pytest.fixture(autouse=True)
def _no_ffmpeg(monkeypatch):
    """Probe and conversion, without a subprocess and without real audio."""
    def _probe(path):
        return audio.AudioInfo(path=Path(path), duration=DURATION, sample_rate=16_000,
                               channels=1, codec="aac", created=RECORDED)

    def _prepare(src, dest, **kw):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"RIFF" + b"\0" * 40)
        return dest

    monkeypatch.setattr(audio, "probe", _probe)
    monkeypatch.setattr(audio, "prepare", _prepare)


# --------------------------------------------------------------------------- #
# the conversations
# --------------------------------------------------------------------------- #
#
# (label, start, end, text). One diarization turn and one ASR segment per entry,
# with identical boundaries, so `assign` attributes every segment at confidence
# 1.0 and no "(uncertain)" marker appears to muddle the assertions.

TALK = [
    ("SPEAKER_00", 0.0, 40.0,
     "Allora, riprendiamo dal punto rimasto aperto la settimana scorsa."),
    ("SPEAKER_01", 40.0, 80.0,
     "Va bene, pero prima vorrei capire chi segue la raccolta dei dati."),
    ("SPEAKER_00", 80.0, 110.0,
     "Ce ne occupiamo noi, poi passiamo il file a chi deve rivederlo."),
]

TALK_THREE = TALK + [
    ("SPEAKER_02", 110.0, 150.0,
     "Intanto preparo la scaletta per il prossimo incontro e la giro a tutti."),
]

# Eight seconds exactly against five: the boundary of voice_min_speech_sec.
GATE_TALK = [
    ("SPEAKER_00", 0.0, 8.0, "Otto secondi tondi, quanto basta per confrontare la voce."),
    ("SPEAKER_01", 8.0, 13.0, "Cinque secondi soltanto, troppo poco."),
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def marker(label: str) -> float:
    """SPEAKER_00 -> 1.0. Its voice print is that number repeated."""
    return float(int(label.removeprefix("SPEAKER_")) + 1)


def label_of(embedding) -> str:
    """The inverse, so a stubbed `match` knows which voice it was handed."""
    value = int(round(float(np.asarray(embedding).reshape(-1)[0])))
    return f"SPEAKER_{value - 1:02d}"


def certain(name: str, score: float = 0.90) -> V.Match:
    """What VoiceRegistry.match returns when it is sure. `person` is set."""
    return V.Match(person=V.Person(id="p" + name[:2].lower(), name=name),
                   score=score, runner_up=None, runner_up_score=0.0,
                   reason="accepted")


def borderline(name: str, score: float = 0.62) -> V.Match:
    """Above the suggestion threshold, below the certainty one. `person` is None.

    The reason names the candidate, exactly as the real one does. That is on
    purpose: it gives the document a second way to leak the name, and the test
    below says it must not take it.
    """
    return V.Match(person=None, score=score, runner_up=None, runner_up_score=0.0,
                   reason=(f"could be {name} ({score:.3f}). That is under the certainty "
                           "threshold (0.75), so confirm it yourself"),
                   candidate=V.Person(id="c" + name[:2].lower(), name=name))


def stranger() -> V.Match:
    return V.Match(None, 0.31, None, 0.0,
                   "no useful similarity: most likely a new voice")


def stub_registry(monkeypatch, verdicts: dict[str, V.Match],
                  probed: list[str] | None = None) -> None:
    """Decide the registry verdict per speaker label, and record who was asked."""
    def _match(self, embedding, **kw):
        label = label_of(embedding)
        if probed is not None:
            probed.append(label)
        return verdicts.get(label, stranger())

    monkeypatch.setattr(V.VoiceRegistry, "match", _match)


def build_job(tmp_path, monkeypatch, spans, *, settings: Settings | None = None,
              log: list[str] | None = None,
              name: str = "riunione di progetto.m4a") -> pipeline.Job:
    """A job whose diarization is already on disk and whose ASR is a stub.

    Everything else in `run()` is the real thing.
    """
    src = tmp_path / "memos" / name
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"no decoder ever looks at these bytes")

    s = settings if settings is not None else Settings(language="it")
    job = pipeline.Job(src, s,
                       report=(log.append if log is not None else (lambda m: None)))

    turns = [{"start": float(a), "end": float(b), "speaker": label}
             for label, a, b, _ in spans]
    (job.dir / "diarization.json").write_text(json.dumps(turns))
    labels = sorted({label for label, *_ in spans})
    np.savez(job.dir / "embeddings.npz",
             **{label: np.full(DIM, marker(label), dtype=np.float32) for label in labels})
    job.state["diar_fingerprint"] = job._diar_fingerprint()

    def _transcribe(wav, s_, **kw):
        return asr.Transcript(
            segments=[{"start": float(a), "end": float(b), "text": text}
                      for _, a, b, text in spans],
            language=s_.language, word_level=False,
        )

    monkeypatch.setattr(asr, "transcribe", _transcribe)
    return job


def source_doc_of(result: pipeline.JobResult) -> str:
    """The document that matters, read back from disk."""
    hits = [p for p in result.outputs if p.name.endswith("(source).md")]
    assert len(hits) == 1, f"expected one source document, got {hits}"
    return hits[0].read_text()


def payload_of(result: pipeline.JobResult) -> dict:
    hits = [p for p in result.outputs if p.suffix == ".json"]
    assert len(hits) == 1, f"expected one json payload, got {hits}"
    return json.loads(hits[0].read_text())


def match_entry(payload: dict, label: str) -> dict:
    return next(m for m in payload["speaker_matches"] if m["speaker"] == label)


def header_line(doc: str, prefix: str) -> str:
    return next(line for line in doc.splitlines() if line.startswith(prefix))


def labels_in(result: pipeline.JobResult) -> set[str]:
    """The speaker labels the run actually produced, read from turns.json."""
    turns = json.loads((result.job_dir / "turns.json").read_text())
    return {t["speaker"] for t in turns if t.get("speaker")}


# --------------------------------------------------------------------------- #
# 1. the dangerous path: a candidate is not a name
# --------------------------------------------------------------------------- #

def test_a_borderline_candidate_never_reaches_the_document_as_a_name(
        tmp_path, monkeypatch):
    """0.62 is a guess. A guess in a source document is read as a fact.

    The registry says "could be Marisol Adeyemi, confirm it yourself". That
    sentence belongs in the briefing a human reads before deciding. It must not
    appear anywhere in the file that gets handed to a colleague or to a model,
    because whatever reads that file cannot tell a suggestion from an attribution.
    """
    job = build_job(tmp_path, monkeypatch, TALK)
    stub_registry(monkeypatch, {"SPEAKER_00": borderline(CANDIDATE_NAME, 0.62)})

    result = job.run()
    doc = source_doc_of(result)

    # The name is nowhere in the finished document, in any spelling.
    assert CANDIDATE_NAME not in doc
    assert "Marisol" not in doc
    assert "Adeyemi" not in doc

    # And the voice reads as unidentified on its own lines, not only in the header.
    assert "**Voice 1 (unidentified)** [" in doc
    assert "**Voice 1**" not in doc
    assert doc.count("**Voice 1 (unidentified)** [") == 2   # both of its turns

    assert "Do not guess who they are" in doc
    assert "Voice 1" in result.unresolved
    assert CANDIDATE_NAME not in result.names.values()
    assert "SPEAKER_00" not in result.names

    # Not a vacuous pass: the candidate did arrive, it was kept out on purpose.
    entry = match_entry(payload_of(result), "SPEAKER_00")
    assert entry["candidate"] == CANDIDATE_NAME
    assert entry["name"] is None
    assert entry["score"] == pytest.approx(0.62)


# --------------------------------------------------------------------------- #
# 2. the mirror: a certain match does reach the document
# --------------------------------------------------------------------------- #

def test_a_certain_match_reaches_the_document_as_a_name(tmp_path, monkeypatch):
    """The other half. Refusing every name would pass the test above and be useless."""
    job = build_job(tmp_path, monkeypatch, TALK)
    stub_registry(monkeypatch, {"SPEAKER_00": certain(CERTAIN_NAME, 0.90)})

    result = job.run()
    doc = source_doc_of(result)

    assert result.names["SPEAKER_00"] == CERTAIN_NAME
    assert doc.count(f"**{CERTAIN_NAME}** [") == 2          # both of its turns
    assert f"{CERTAIN_NAME} (unidentified)" not in doc
    assert "**Voice 1" not in doc                            # it is no longer a voice

    assert CERTAIN_NAME not in result.unresolved
    assert "Voice 1" not in result.unresolved
    assert result.unresolved == ["Voice 2"]                  # the other one still is
    assert "**Voice 2 (unidentified)** [" in doc

    entry = match_entry(payload_of(result), "SPEAKER_00")
    assert entry["name"] == CERTAIN_NAME
    assert entry["score"] == pytest.approx(0.90)


# --------------------------------------------------------------------------- #
# 3. the seam: matches -> profiles -> names -> meta -> the written files
# --------------------------------------------------------------------------- #

def test_the_finished_document_agrees_with_the_job_result(tmp_path, monkeypatch):
    """One recognised voice, one borderline, one stranger, and everything lines up.

    Each assertion here is a joint. They pass individually in every other test
    file and were never checked against each other.
    """
    settings = Settings(language="it")
    job = build_job(tmp_path, monkeypatch, TALK_THREE, settings=settings)
    stub_registry(monkeypatch, {
        "SPEAKER_00": certain(CERTAIN_NAME, 0.90),
        "SPEAKER_01": borderline(CANDIDATE_NAME, 0.62),
    })

    result = job.run()
    doc = source_doc_of(result)
    payload = payload_of(result)
    labels = labels_in(result)

    assert labels == {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02"}
    assert set(result.speakers) == labels

    # One file per configured format, all of them on disk, named as export says.
    stem = pipeline.slugify(job.source.stem)
    assert len(result.outputs) == len(settings.output_formats)
    assert {p.name for p in result.outputs} == {
        export.FILENAMES[f].format(stem=stem) for f in settings.output_formats}
    assert all(p.is_file() for p in result.outputs)

    # Named and unresolved are complementary. Neither invents a speaker.
    named = set(result.names)
    assert named == {"SPEAKER_00"}
    assert set(result.unresolved) == {export.display(l, {}) for l in labels - named}

    # Nobody is both named and listed as unidentified.
    for name in result.names.values():
        assert name not in result.unresolved
        assert f"**{name} (unidentified)**" not in doc

    # Every speaker in the turns is in the participants line, under the name the
    # document uses for it everywhere else.
    participants = header_line(doc, "- **Participants**:")
    for label in labels:
        assert export.display(label, result.names) in participants

    # The header's caveat lists exactly the unresolved voices and nobody else.
    unidentified = header_line(doc, "- **Unidentified voices**:")
    for who in result.unresolved:
        assert who in unidentified
    assert CERTAIN_NAME not in unidentified
    assert CANDIDATE_NAME not in unidentified

    # The machine-readable half says the same thing as the JobResult.
    assert payload["names"] == result.names
    assert payload["meta"]["unresolved"] == result.unresolved
    assert {m["speaker"] for m in payload["speaker_matches"]} == labels
    assert payload["meta"]["speaker_stats"] == {
        "SPEAKER_00": 70.0, "SPEAKER_01": 40.0, "SPEAKER_02": 40.0}

    # And so do the other formats, for the one speaker that has a name.
    plain = next(p for p in result.outputs if p.suffix == ".txt").read_text()
    subtitles = next(p for p in result.outputs if p.suffix == ".srt").read_text()
    assert f"{CERTAIN_NAME}:" in plain
    assert f"[{CERTAIN_NAME}]" in subtitles
    assert CANDIDATE_NAME not in plain
    assert CANDIDATE_NAME not in subtitles


# --------------------------------------------------------------------------- #
# 4. the minimum-speech gate in identify()
# --------------------------------------------------------------------------- #

def test_a_voice_with_too_little_speech_is_never_compared_at_all(tmp_path, monkeypatch):
    """Five seconds of audio is not a voice print, whatever the score says.

    The registry here would name both speakers with high confidence. The gate has
    to stop the thin one before the comparison happens, not filter the answer
    afterwards, so the test watches which vectors reach `match`.

    The other speaker has exactly `voice_min_speech_sec` of speech and is matched,
    which pins the comparison at its boundary.
    """
    settings = Settings(language="it", voice_min_speech_sec=8.0)
    job = build_job(tmp_path, monkeypatch, GATE_TALK, settings=settings)
    probed: list[str] = []
    stub_registry(monkeypatch, {
        "SPEAKER_00": certain(CERTAIN_NAME, 0.93),
        "SPEAKER_01": certain(THIRD_NAME, 0.97),
    }, probed=probed)

    result = job.run()
    doc = source_doc_of(result)

    assert probed == ["SPEAKER_00"], \
        "the thin voice print was compared against the registry anyway"

    # 5 seconds: no name, anywhere.
    assert THIRD_NAME not in doc
    assert "Nkemelu" not in doc
    assert THIRD_NAME not in result.names.values()
    assert "**Voice 2 (unidentified)** [" in doc
    assert "Voice 2" in result.unresolved

    # 8 seconds exactly: matched and named. `<` here, not `<=`.
    assert result.names["SPEAKER_00"] == CERTAIN_NAME
    assert f"**{CERTAIN_NAME}** [" in doc
    assert "Voice 1" not in result.unresolved

    # The refusal is recorded, with its reason, instead of looking like a miss.
    entry = match_entry(payload_of(result), "SPEAKER_01")
    assert entry["name"] is None
    assert entry["candidate"] is None
    assert entry["score"] == 0.0
    assert "only 5s of speech" in entry["reason"]
    assert "too thin to trust" in entry["reason"]


# --------------------------------------------------------------------------- #
# 5. set_names: what it refuses, and what it makes permanent
# --------------------------------------------------------------------------- #

def test_set_names_refuses_to_enroll_under_an_empty_name(tmp_path, monkeypatch):
    """An empty name enrolled is a voice print filed under nobody.

    It would then match the next recording and attach that emptiness to a real
    speaker, and there is no way back from it: the registry cannot tell an
    unnamed print from a wrong one.
    """
    job = build_job(tmp_path, monkeypatch, TALK)
    stub_registry(monkeypatch, {})
    job.run()

    result = job.set_names({"SPEAKER_01": ""})
    doc = source_doc_of(result)

    reg = V.VoiceRegistry()
    assert reg.people == {}
    assert reg.summary() == []
    assert len(reg.emb) == 0

    # And the empty name never becomes a name in the document either.
    assert "SPEAKER_01" not in result.names
    assert "" not in result.names.values()
    assert "**Voice 2 (unidentified)** [" in doc
    assert "Voice 2" in result.unresolved


def test_a_hand_assigned_name_survives_into_the_document_and_the_registry(
        tmp_path, monkeypatch):
    """The point of the whole tool: name a voice once, and it stays named.

    Once here, in the rewritten document, and once for next time, in the registry.
    """
    job = build_job(tmp_path, monkeypatch, TALK)
    stub_registry(monkeypatch, {"SPEAKER_01": borderline(CANDIDATE_NAME, 0.62)})

    first = job.run()
    assert "Voice 2" in first.unresolved
    assert CANDIDATE_NAME not in source_doc_of(first)

    result = job.set_names({"SPEAKER_01": HAND_NAME})
    doc = source_doc_of(result)

    assert result.names["SPEAKER_01"] == HAND_NAME
    assert f"**{HAND_NAME}** [" in doc
    assert f"{HAND_NAME} (unidentified)" not in doc
    assert "**Voice 2" not in doc
    assert HAND_NAME not in result.unresolved
    assert "Voice 2" not in result.unresolved

    # The name outlives the job: this is what makes the next recording easier.
    reg = V.VoiceRegistry()
    person = reg.by_name(HAND_NAME)
    assert person is not None
    assert len(person.rows) == 1
    assert reg.vectors_of(person).tolist() == [[marker("SPEAKER_01")] * DIM]
    assert person.sources == [job.source.name]

    # It reached disk, not just this instance.
    stored = json.loads(V.REGISTRY_PATH.read_text())
    assert [p["name"] for p in stored["people"]] == [HAND_NAME]
