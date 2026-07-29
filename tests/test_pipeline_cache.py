"""Cache invalidation and job identity in `scriba.pipeline`.

Nothing here loads a model, decodes audio or reaches the network. Every test
builds the job folder by hand and checks the decisions the pipeline makes
*around* the expensive parts: which folder a source file is filed under, whether
a cached artefact still describes the file in front of it, and whether a
half-written file can ever be read.

The three failures these tests stand for all produced a finished document that
looked right, which is the only reason the fingerprints exist at all.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scriba import asr, config, diarize as diarize_mod, naming, pipeline
from scriba.config import Settings

REAL_JOBS = Path.home() / ".scriba" / "jobs"

# The files a job derives from its source, in the order pipeline lists them.
DERIVED = ("audio16k.wav", "transcript.json", "diarization.json",
           "embeddings.npz", "turns.json")


# --------------------------------------------------------------------------- #
# isolation
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module", autouse=True)
def _real_home_untouched():
    """A patch that misses would write into the user's actual job folder."""
    def listing():
        return sorted(p.name for p in REAL_JOBS.iterdir()) if REAL_JOBS.exists() else None

    before = listing()
    yield
    assert listing() == before, "these tests wrote into the real ~/.scriba/jobs"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """Point every path the pipeline knows about at a throwaway directory."""
    root = tmp_path / "scriba-home"
    jobs = root / "jobs"
    jobs.mkdir(parents=True)
    monkeypatch.setenv("SCRIBA_HOME", str(root))
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(config, "VOICES_DIR", root / "voices")
    monkeypatch.setattr(config, "JOBS_DIR", jobs)
    monkeypatch.setattr(config, "SETTINGS_PATH", root / "settings.json")
    # `from .config import JOBS_DIR` bound the value into the pipeline module,
    # so patching config alone would leave Job writing to the real home.
    monkeypatch.setattr(pipeline, "JOBS_DIR", jobs)
    return root


@pytest.fixture(autouse=True)
def _no_models(monkeypatch):
    """Any test that reaches a model is a broken test, not a slow one."""
    def forbidden(*a, **kw):
        raise AssertionError("a test tried to run a real model")

    monkeypatch.setattr(asr, "transcribe", forbidden)
    monkeypatch.setattr(diarize_mod, "run", forbidden)
    monkeypatch.setattr(pipeline, "hf_token", lambda: None)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def audio_file(folder: Path, name: str = "riunione.m4a",
               data: bytes = b"first-take-bytes") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(data)
    return path


def make_job(source: Path, settings: Settings | None = None,
             log: list[str] | None = None) -> pipeline.Job:
    return pipeline.Job(
        source,
        settings if settings is not None else Settings(),
        report=(log.append if log is not None else (lambda m: None)),
    )


def tweak(base: Settings, **changes) -> Settings:
    return dataclasses.replace(base, **changes)


def fill_derived(job: pipeline.Job) -> None:
    """Plausible leftovers from a completed run of the previous audio."""
    (job.dir / "audio16k.wav").write_bytes(b"RIFF-not-really")
    (job.dir / "transcript.json").write_text(
        json.dumps({"segments": [{"text": "prima registrazione"}],
                    "language": "it", "word_level": True}))
    (job.dir / "diarization.json").write_text(
        json.dumps([{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"}]))
    np.savez(job.dir / "embeddings.npz", SPEAKER_00=np.zeros(4, dtype=np.float32))
    (job.dir / "turns.json").write_text(json.dumps([{"speaker": "SPEAKER_00"}]))


def write_diar_cache(job: pipeline.Job, labels=("SPEAKER_00", "SPEAKER_01")) -> None:
    turns = [{"start": float(i * 3), "end": float(i * 3 + 2), "speaker": label}
             for i, label in enumerate(labels)]
    (job.dir / "diarization.json").write_text(json.dumps(turns))
    np.savez(job.dir / "embeddings.npz",
             **{label: np.full(4, i + 1, dtype=np.float32)
                for i, label in enumerate(labels)})


# --------------------------------------------------------------------------- #
# job identity: job_slug
# --------------------------------------------------------------------------- #

def test_job_dir_lives_under_the_patched_home(tmp_path):
    """Proof the isolation above actually holds before anything else runs."""
    job = make_job(audio_file(tmp_path / "memos"))
    assert str(job.dir).startswith(str(tmp_path))
    assert job.dir.is_dir()


def test_same_name_in_two_folders_gets_two_slugs(tmp_path):
    desktop = audio_file(tmp_path / "Desktop", "Recording 38.m4a", b"AAAA")
    cloud = audio_file(tmp_path / "CloudStorage", "Recording 38.m4a", b"BBBBBB")
    assert pipeline.job_slug(desktop) != pipeline.job_slug(cloud)


def test_same_name_in_two_folders_gets_two_job_dirs(tmp_path):
    """The failure this exists for: the second file reusing the first transcript."""
    a = make_job(audio_file(tmp_path / "Desktop", "Recording 38.m4a", b"AAAA"))
    b = make_job(audio_file(tmp_path / "CloudStorage", "Recording 38.m4a", b"BBBBBB"))
    assert a.dir != b.dir
    assert a.dir.parent == b.dir.parent


def test_slug_is_stable_across_calls(tmp_path):
    src = audio_file(tmp_path / "memos")
    assert pipeline.job_slug(src) == pipeline.job_slug(src)


def test_slug_is_stable_across_job_instances(tmp_path):
    """Rerunning the same file must land in the same folder, cache and all."""
    src = audio_file(tmp_path / "memos")
    assert make_job(src).dir == make_job(src).dir


def test_slug_ignores_how_the_path_was_spelled(tmp_path, monkeypatch):
    """A relative path, a `..` or a symlink must not fork the job folder."""
    src = audio_file(tmp_path / "memos")
    monkeypatch.chdir(tmp_path / "memos")
    indirect = Path("..") / "memos" / src.name
    assert make_job(indirect).dir == make_job(src).dir


def test_slug_keeps_the_name_readable_and_adds_a_short_digest(tmp_path):
    src = audio_file(tmp_path / "memos", "Riunione di Lavoro.m4a")
    slug = pipeline.job_slug(src)
    stem, _, digest = slug.rpartition("-")
    assert stem == "riunione-di-lavoro"
    assert len(digest) == 6 and all(c in "0123456789abcdef" for c in digest)


def test_the_digest_comes_from_the_folder_not_the_file(tmp_path):
    """Two different names in one folder share the suffix, nothing more."""
    folder = tmp_path / "memos"
    one = pipeline.job_slug(audio_file(folder, "alfa.m4a"))
    two = pipeline.job_slug(audio_file(folder, "beta.m4a"))
    assert one.rpartition("-")[2] == two.rpartition("-")[2]
    assert one != two


def test_moving_the_file_changes_the_slug(tmp_path):
    """The honest trade the docstring names: the old job is orphaned, not reused."""
    src = audio_file(tmp_path / "memos")
    before = pipeline.job_slug(src)
    moved = tmp_path / "archivio" / src.name
    moved.parent.mkdir()
    src.rename(moved)
    assert pipeline.job_slug(moved) != before


def test_a_name_with_no_word_characters_still_gets_a_folder(tmp_path):
    src = audio_file(tmp_path / "memos", "!!!.m4a")
    assert pipeline.job_slug(src).startswith("recording-")


def test_accents_survive_as_plain_letters(tmp_path):
    src = audio_file(tmp_path / "memos", "Perché però.m4a")
    assert pipeline.job_slug(src).startswith("perche-pero-")


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN, not fixed. slugify collapses spaces, underscores and hyphens to one "
    "character, so 'verbale 1.m4a' and 'verbale_1.m4a' in the same folder share a "
    "job directory and overwrite each other's output/. The source fingerprint "
    "keeps the transcript from being wrong, at the cost of a full re-transcription "
    "on every alternation. Fixing it means putting the file name into the digest, "
    "which re-keys every job folder already on disk and re-transcribes all of them, "
    "so it waits for a moment when that is acceptable."))
def test_space_and_underscore_variants_do_not_collide(tmp_path):
    folder = tmp_path / "memos"
    spaced = audio_file(folder, "verbale 1.m4a", b"AAAA")
    scored = audio_file(folder, "verbale_1.m4a", b"BBBBBB")
    assert pipeline.job_slug(spaced) != pipeline.job_slug(scored)


# --------------------------------------------------------------------------- #
# _source_fingerprint
# --------------------------------------------------------------------------- #

def test_source_fingerprint_is_stable_when_nothing_changes(tmp_path):
    job = make_job(audio_file(tmp_path / "memos"))
    first = job._source_fingerprint()
    job.source.read_bytes()  # reading is not changing
    assert job._source_fingerprint() == first
    assert make_job(job.source)._source_fingerprint() == first


def test_source_fingerprint_carries_the_size(tmp_path):
    src = audio_file(tmp_path / "memos", data=b"0123456789")
    size, _, mtime = make_job(src)._source_fingerprint().partition(":")
    assert size == str(src.stat().st_size) == "10"
    assert mtime.isdigit()


def test_source_fingerprint_changes_when_the_file_grows(tmp_path):
    job = make_job(audio_file(tmp_path / "memos", data=b"short"))
    first = job._source_fingerprint()
    job.source.write_bytes(b"a much longer second take")
    assert job._source_fingerprint() != first


def test_source_fingerprint_notices_a_recording_made_over_the_old_one(tmp_path):
    """Same name, same length, different audio. The case the cache got wrong."""
    src = audio_file(tmp_path / "memos", data=b"AAAAAAAAAAAA")
    job = make_job(src)
    first = job._source_fingerprint()
    st = src.stat()
    src.write_bytes(b"BBBBBBBBBBBB")
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
    assert src.stat().st_size == st.st_size
    assert job._source_fingerprint() != first


def test_two_files_with_the_same_name_have_different_fingerprints(tmp_path):
    a = make_job(audio_file(tmp_path / "Desktop", "Recording 38.m4a", b"AAAA"))
    b = make_job(audio_file(tmp_path / "Cloud", "Recording 38.m4a", b"BBBBBB"))
    assert a._source_fingerprint() != b._source_fingerprint()


def test_source_fingerprint_misses_an_edit_that_preserves_size_and_mtime(tmp_path):
    """Documented trade, not a defect: hashing 200 MB per run buys too little."""
    src = audio_file(tmp_path / "memos", data=b"AAAAAAAAAAAA")
    job = make_job(src)
    first = job._source_fingerprint()
    st = src.stat()
    src.write_bytes(b"BBBBBBBBBBBB")
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert job._source_fingerprint() == first


# --------------------------------------------------------------------------- #
# _asr_fingerprint
# --------------------------------------------------------------------------- #

ASR_MATTERS = [
    ("model", "small"),
    ("compute_type", "float32"),
    ("beam_size", 1),
    ("align", False),
    ("vad_onset", 0.6),
    ("vad_offset", 0.42),
    ("initial_prompt", "Una riunione di lavoro, con punteggiatura."),
    ("hotwords", ["Bergamotto", "Quadrifoglio"]),
    ("temperature_fallback", False),
    ("condition_on_previous_text", True),
    ("language", "es"),
]

ASR_DOES_NOT_MATTER = [
    ("threads", 4),                     # speed only
    ("timestamp_every", 5),             # formatting
    ("output_formats", ["md"]),         # what gets written, not what is said
    ("voice_match_threshold", 0.9),     # name attribution
    ("voice_min_speech_sec", 30.0),
    ("min_speakers", 2),                # diarization
    ("max_speakers", 4),
    ("diarize", False),
    ("diarize_device", "cpu"),
]


def test_asr_fingerprint_is_stable_for_the_same_settings(tmp_path):
    src = audio_file(tmp_path / "memos")
    s = Settings()
    first = make_job(src, s)._asr_fingerprint()
    assert make_job(src, s)._asr_fingerprint() == first
    assert make_job(src, tweak(s))._asr_fingerprint() == first


@pytest.mark.parametrize("field,value", ASR_MATTERS, ids=[f for f, _ in ASR_MATTERS])
def test_asr_fingerprint_changes_with_a_setting_that_changes_the_words(
        tmp_path, field, value):
    src = audio_file(tmp_path / "memos")
    base = Settings()
    assert (make_job(src, tweak(base, **{field: value}))._asr_fingerprint()
            != make_job(src, base)._asr_fingerprint())


@pytest.mark.parametrize("field,value", ASR_DOES_NOT_MATTER,
                         ids=[f for f, _ in ASR_DOES_NOT_MATTER])
def test_asr_fingerprint_ignores_settings_the_transcript_does_not_depend_on(
        tmp_path, field, value):
    src = audio_file(tmp_path / "memos")
    base = Settings()
    assert (make_job(src, tweak(base, **{field: value}))._asr_fingerprint()
            == make_job(src, base)._asr_fingerprint())


def test_asr_fingerprint_follows_the_language_recorded_in_the_state(tmp_path):
    """Redo the job in another language and the transcript cannot stand."""
    job = make_job(audio_file(tmp_path / "memos"))
    job.state["language"] = "it"
    italian = job._asr_fingerprint()
    job.state["language"] = "es"
    assert job._asr_fingerprint() != italian


def test_a_detected_language_beats_the_auto_default(tmp_path):
    job = make_job(audio_file(tmp_path / "memos"))
    auto = job._asr_fingerprint()          # settings.language == "auto"
    job.state["language"] = "it"
    assert job._asr_fingerprint() != auto


def test_hotword_order_is_not_a_change(tmp_path):
    src = audio_file(tmp_path / "memos")
    one = make_job(src, tweak(Settings(), hotwords=["Bergamotto", "Quadrifoglio"]))
    two = make_job(src, tweak(Settings(), hotwords=["Quadrifoglio", "Bergamotto"]))
    assert one._asr_fingerprint() == two._asr_fingerprint()


def test_asr_fingerprint_is_a_short_hex_digest(tmp_path):
    fp = make_job(audio_file(tmp_path / "memos"))._asr_fingerprint()
    assert len(fp) == 12 and all(c in "0123456789abcdef" for c in fp)


@pytest.mark.parametrize("field,value", [
    ("no_speech_threshold", 0.2),
    ("log_prob_threshold", -0.4),
    ("compression_ratio_threshold", 1.8),
], ids=["no_speech_threshold", "log_prob_threshold", "compression_ratio_threshold"])
# Regression: this used to fail.
def test_asr_fingerprint_covers_the_decoding_thresholds(tmp_path, field, value):
    src = audio_file(tmp_path / "memos")
    base = Settings()
    assert (make_job(src, tweak(base, **{field: value}))._asr_fingerprint()
            != make_job(src, base)._asr_fingerprint())


# --------------------------------------------------------------------------- #
# _diar_fingerprint
# --------------------------------------------------------------------------- #

def test_diar_fingerprint_is_stable_for_the_same_settings(tmp_path):
    src = audio_file(tmp_path / "memos")
    s = Settings()
    assert make_job(src, s)._diar_fingerprint() == make_job(src, s)._diar_fingerprint()


@pytest.mark.parametrize("field,value", [("min_speakers", 2), ("max_speakers", 4)])
def test_diar_fingerprint_changes_with_the_speaker_counts(tmp_path, field, value):
    src = audio_file(tmp_path / "memos")
    base = Settings()
    assert (make_job(src, tweak(base, **{field: value}))._diar_fingerprint()
            != make_job(src, base)._diar_fingerprint())


def test_diar_fingerprint_changes_with_the_model(tmp_path, monkeypatch):
    """Which pyannote model is in use depends on the installed version."""
    job = make_job(audio_file(tmp_path / "memos"))
    before = job._diar_fingerprint()
    monkeypatch.setattr(diarize_mod, "DIARIZATION_MODEL",
                        "pyannote/speaker-diarization-3.1-invented-for-this-test")
    assert job._diar_fingerprint() != before


@pytest.mark.parametrize("field,value", [
    ("model", "small"), ("language", "es"), ("hotwords", ["Bergamotto"]),
    ("beam_size", 1), ("align", False), ("compute_type", "float32"),
], ids=["model", "language", "hotwords", "beam_size", "align", "compute_type"])
def test_diar_fingerprint_ignores_the_asr_settings(tmp_path, field, value):
    """Changing the transcription must not throw away 200s of diarization."""
    src = audio_file(tmp_path / "memos")
    base = Settings()
    assert (make_job(src, tweak(base, **{field: value}))._diar_fingerprint()
            == make_job(src, base)._diar_fingerprint())


def test_diar_fingerprint_is_a_short_hex_digest(tmp_path):
    fp = make_job(audio_file(tmp_path / "memos"))._diar_fingerprint()
    assert len(fp) == 12 and all(c in "0123456789abcdef" for c in fp)


# Regression: this used to fail.
def test_diar_fingerprint_changes_with_the_device(tmp_path):
    src = audio_file(tmp_path / "memos")
    base = Settings()
    assert (make_job(src, tweak(base, diarize_device="cpu"))._diar_fingerprint()
            != make_job(src, tweak(base, diarize_device="mps"))._diar_fingerprint())


# --------------------------------------------------------------------------- #
# _drop_stale_cache
# --------------------------------------------------------------------------- #

def test_first_run_adopts_the_fingerprint_and_keeps_the_artefacts(tmp_path):
    """No recorded fingerprint means nothing to compare, not 'throw it away'."""
    job = make_job(audio_file(tmp_path / "memos"))
    fill_derived(job)
    job.state["asr_fingerprint"] = "abc123abc123"
    job._drop_stale_cache()
    assert job.state["source_fingerprint"] == job._source_fingerprint()
    assert job.state["asr_fingerprint"] == "abc123abc123"
    assert all((job.dir / name).exists() for name in DERIVED)


def test_an_unchanged_source_keeps_every_cached_stage(tmp_path):
    log: list[str] = []
    job = make_job(audio_file(tmp_path / "memos"), log=log)
    fill_derived(job)
    job.state.update({"source_fingerprint": job._source_fingerprint(),
                      "asr_fingerprint": "aaaaaaaaaaaa",
                      "diar_fingerprint": "bbbbbbbbbbbb",
                      "language": "it", "duration": 612.0, "word_level": True})
    job._drop_stale_cache()
    assert all((job.dir / name).exists() for name in DERIVED)
    assert job.state["asr_fingerprint"] == "aaaaaaaaaaaa"
    assert job.state["diar_fingerprint"] == "bbbbbbbbbbbb"
    assert job.state["language"] == "it"
    assert log == []


def test_a_changed_source_deletes_everything_derived_from_it(tmp_path):
    src = audio_file(tmp_path / "memos", data=b"first take")
    job = make_job(src)
    fill_derived(job)
    job.state["source_fingerprint"] = job._source_fingerprint()

    src.write_bytes(b"a completely different second recording")
    job._drop_stale_cache()

    assert [name for name in DERIVED if (job.dir / name).exists()] == []


def test_a_changed_source_forgets_the_fingerprints_and_the_language(tmp_path):
    src = audio_file(tmp_path / "memos", data=b"first take")
    job = make_job(src)
    fill_derived(job)
    job.state.update({
        "source_fingerprint": job._source_fingerprint(),
        "asr_fingerprint": "aaaaaaaaaaaa", "diar_fingerprint": "bbbbbbbbbbbb",
        "language": "it", "language_note": "set by hand", "language_confidence": 0.98,
        "language_samples": [], "word_level": True, "duration": 612.0,
    })

    src.write_bytes(b"a completely different second recording")
    job._drop_stale_cache()

    for key in ("asr_fingerprint", "diar_fingerprint", "language", "language_note",
                "language_confidence", "language_samples", "word_level", "duration"):
        assert key not in job.state, f"{key} survived a source change"
    assert job.state["source_fingerprint"] == job._source_fingerprint()


def test_a_changed_source_is_reported_not_swallowed(tmp_path):
    log: list[str] = []
    src = audio_file(tmp_path / "memos", data=b"first take")
    job = make_job(src, log=log)
    job.state["source_fingerprint"] = job._source_fingerprint()
    src.write_bytes(b"a completely different second recording")
    job._drop_stale_cache()
    assert any("source file changed" in m for m in log)


def test_the_new_fingerprint_reaches_disk_immediately(tmp_path):
    """A crash before the end of the run must not resurrect the old cache."""
    src = audio_file(tmp_path / "memos", data=b"first take")
    job = make_job(src)
    job.state["source_fingerprint"] = job._source_fingerprint()
    src.write_bytes(b"a completely different second recording")
    job._drop_stale_cache()

    on_disk = json.loads((job.dir / "state.json").read_text())
    assert on_disk["source_fingerprint"] == job._source_fingerprint()

    # A fresh Job reading that state agrees the source is current.
    again = make_job(src)
    fill_derived(again)
    again._drop_stale_cache()
    assert all((again.dir / name).exists() for name in DERIVED)


def test_dropping_twice_is_not_a_second_deletion(tmp_path):
    src = audio_file(tmp_path / "memos", data=b"first take")
    job = make_job(src)
    job.state["source_fingerprint"] = job._source_fingerprint()
    src.write_bytes(b"a completely different second recording")
    job._drop_stale_cache()
    fill_derived(job)          # the rerun rebuilds them
    job._drop_stale_cache()
    assert all((job.dir / name).exists() for name in DERIVED)


def test_a_changed_source_keeps_the_untouched_files(tmp_path):
    """Only what is derived from the audio goes. The state file stays readable."""
    src = audio_file(tmp_path / "memos", data=b"first take")
    job = make_job(src)
    fill_derived(job)
    (job.dir / "who-is-who.md").write_text("# chi e chi\n")
    job.state["source_fingerprint"] = job._source_fingerprint()
    src.write_bytes(b"a completely different second recording")
    job._drop_stale_cache()
    assert (job.dir / "who-is-who.md").exists()
    assert (job.dir / "state.json").exists()


# Regression: this used to fail.
def test_a_changed_source_forgets_the_speaker_names(tmp_path):
    src = audio_file(tmp_path / "memos", data=b"first take")
    job = make_job(src)
    job.state.update({"source_fingerprint": job._source_fingerprint(),
                      "names": {"SPEAKER_00": "Ospite Uno"}})
    src.write_bytes(b"a completely different second recording")
    job._drop_stale_cache()
    assert not job.state.get("names")


def test_the_registry_decides_once_the_stale_names_are_gone(tmp_path):
    """Why the leftover mattered.

    A stored mapping wins over the registry in apply_names, so a name left over
    from the previous recording was reapplied to whoever came out as SPEAKER_00
    in the new one, and the finished document said so. With the mapping dropped,
    the person the new audio actually contains is the one who gets named.
    """
    src = audio_file(tmp_path / "memos", data=b"first take")
    job = make_job(src)
    job.state.update({"source_fingerprint": job._source_fingerprint(),
                      "names": {"SPEAKER_00": "Ospite Uno"}})
    src.write_bytes(b"a completely different second recording")
    job._drop_stale_cache()

    # The speaker the *new* audio produced, already recognised as someone else.
    fresh = naming.SpeakerProfile(label="SPEAKER_00", speech_seconds=90.0,
                                  turn_count=12, first_seen=0.0,
                                  registry_name="Ospite Due")
    applied = naming.apply_names(job.state.get("names", {}), [fresh])
    assert applied["SPEAKER_00"] == "Ospite Due"


# --------------------------------------------------------------------------- #
# transcribe(): the cached transcript against a changed fingerprint
# --------------------------------------------------------------------------- #

def fake_asr(monkeypatch, calls: list, text: str = "seconda trascrizione"):
    def _transcribe(wav, s, **kw):
        calls.append(s)
        return asr.Transcript(segments=[{"text": text}], language=s.language,
                              word_level=False)
    monkeypatch.setattr(asr, "transcribe", _transcribe)


def test_a_matching_asr_fingerprint_reuses_the_transcript(tmp_path, monkeypatch):
    log: list[str] = []
    job = make_job(audio_file(tmp_path / "memos"), log=log)
    (job.dir / "transcript.json").write_text(json.dumps(
        {"segments": [{"text": "prima trascrizione"}], "language": "it",
         "word_level": True}))
    job.state["asr_fingerprint"] = job._asr_fingerprint()

    calls: list = []
    fake_asr(monkeypatch, calls)
    segments = job.transcribe()

    assert calls == []                                   # no model was loaded
    assert segments == [{"text": "prima trascrizione"}]
    assert job.state["word_level"] is True
    assert any("reusing the cache" in m for m in log)


def test_a_changed_asr_setting_does_not_keep_the_old_transcript(tmp_path, monkeypatch):
    """The whole point of the ASR fingerprint, through the real state file."""
    src = audio_file(tmp_path / "memos")
    calls: list = []
    fake_asr(monkeypatch, calls)

    first = make_job(src, Settings())
    first.transcribe()
    assert len(calls) == 1

    log: list[str] = []
    second = make_job(src, tweak(Settings(), model="small"), log=log)
    assert second.dir == first.dir
    assert second.state["asr_fingerprint"] == first.state["asr_fingerprint"]

    segments = second.transcribe()
    assert len(calls) == 2, "a changed model silently reused the cached transcript"
    assert calls[1].model == "small"
    assert segments == [{"text": "seconda trascrizione"}]
    assert second.state["asr_fingerprint"] == second._asr_fingerprint()
    assert json.loads((second.dir / "transcript.json").read_text())["segments"] \
        == [{"text": "seconda trascrizione"}]
    assert any("settings changed" in m for m in log)


def test_a_transcript_with_no_recorded_fingerprint_is_redone(tmp_path, monkeypatch):
    """A job folder from before the fingerprint existed cannot be vouched for."""
    job = make_job(audio_file(tmp_path / "memos"))
    (job.dir / "transcript.json").write_text(json.dumps(
        {"segments": [{"text": "prima trascrizione"}], "language": "it",
         "word_level": True}))
    calls: list = []
    fake_asr(monkeypatch, calls)
    assert job.transcribe() == [{"text": "seconda trascrizione"}]
    assert len(calls) == 1


def test_a_changed_language_in_the_state_redoes_the_transcript(tmp_path, monkeypatch):
    """'The state flipped, the text did not', from the comment block."""
    job = make_job(audio_file(tmp_path / "memos"))
    job.state["language"] = "it"
    (job.dir / "transcript.json").write_text(json.dumps(
        {"segments": [{"text": "prima trascrizione"}], "language": "it",
         "word_level": True}))
    job.state["asr_fingerprint"] = job._asr_fingerprint()

    job.state["language"] = "es"
    calls: list = []
    fake_asr(monkeypatch, calls)
    job.transcribe()
    assert len(calls) == 1
    assert calls[0].language == "es"


def test_force_ignores_a_matching_fingerprint(tmp_path, monkeypatch):
    job = make_job(audio_file(tmp_path / "memos"))
    (job.dir / "transcript.json").write_text(json.dumps(
        {"segments": [{"text": "prima trascrizione"}], "language": "it",
         "word_level": True}))
    job.state["asr_fingerprint"] = job._asr_fingerprint()
    calls: list = []
    fake_asr(monkeypatch, calls)
    job.transcribe(force=True)
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# diarize(): the cached turns against a changed fingerprint
# --------------------------------------------------------------------------- #

def test_a_matching_diar_fingerprint_reuses_the_turns_and_the_voice_prints(tmp_path):
    log: list[str] = []
    job = make_job(audio_file(tmp_path / "memos"), log=log)
    write_diar_cache(job)
    job.state["diar_fingerprint"] = job._diar_fingerprint()

    dia = job.diarize()   # hf_token() returns None: reaching pyannote would raise

    assert [t.speaker for t in dia.turns] == ["SPEAKER_00", "SPEAKER_01"]
    assert set(dia.embeddings) == {"SPEAKER_00", "SPEAKER_01"}
    assert dia.embeddings["SPEAKER_01"].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert any("reusing the cache" in m for m in log)


def test_a_changed_speaker_count_does_not_reuse_the_diarization(tmp_path):
    log: list[str] = []
    job = make_job(audio_file(tmp_path / "memos"), tweak(Settings(), max_speakers=2),
                   log=log)
    write_diar_cache(job)
    job.state["diar_fingerprint"] = job._diar_fingerprint()
    job._save_state()

    # The next run, with the speaker count raised, reads that state back.
    changed = make_job(job.source, tweak(Settings(), max_speakers=4), log=log)
    assert changed.dir == job.dir
    assert changed.state["diar_fingerprint"] != changed._diar_fingerprint()
    # No token, so a run that decides to re-diarize stops at the token check.
    with pytest.raises(RuntimeError) as err:
        changed.diarize()
    assert pipeline.ERR_NO_TOKEN in str(err.value)
    assert any("speaker count changed" in m for m in log)


def test_diarization_with_no_recorded_fingerprint_is_redone(tmp_path):
    job = make_job(audio_file(tmp_path / "memos"))
    write_diar_cache(job)
    with pytest.raises(RuntimeError) as err:
        job.diarize()
    assert pipeline.ERR_NO_TOKEN in str(err.value)


# --------------------------------------------------------------------------- #
# write_atomic
# --------------------------------------------------------------------------- #

OLD = json.dumps({"names": {"SPEAKER_00": "Ospite Uno"}}, indent=2)
NEW = json.dumps({"names": {"SPEAKER_00": "Ospite Due"}}, indent=2)


def test_write_atomic_creates_the_file_and_leaves_nothing_behind(tmp_path):
    target = tmp_path / "state.json"
    pipeline.write_atomic(target, NEW)
    assert target.read_text() == NEW
    assert list(tmp_path.glob("*.part")) == []


def test_write_atomic_replaces_by_rename_not_in_place(tmp_path):
    """A different inode is the proof: an in-place rewrite would keep the old one."""
    target = tmp_path / "state.json"
    target.write_text(OLD)
    before = target.stat().st_ino
    pipeline.write_atomic(target, NEW)
    assert target.read_text() == NEW
    assert target.stat().st_ino != before


def test_the_target_is_never_half_written(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    target.write_text(OLD)
    seen: list[str] = []
    scratch: list[Path] = []
    real_write_text = Path.write_text

    def spy(self, data, *args, **kwargs):
        result = real_write_text(self, data, *args, **kwargs)
        if self.name.endswith(".part"):
            scratch.append(self)
            seen.append(target.read_text())   # a reader looking mid-write
        return result

    monkeypatch.setattr(Path, "write_text", spy)
    pipeline.write_atomic(target, NEW)

    assert seen == [OLD], "the new content was visible under the real name too early"
    assert scratch and scratch[0].name == "state.json.part"
    assert target.read_text() == NEW


def test_a_failed_write_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    target.write_text(OLD)
    real_write_text = Path.write_text

    def out_of_space(self, data, *args, **kwargs):
        real_write_text(self, data[: len(data) // 2], *args, **kwargs)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_text", out_of_space)
    with pytest.raises(OSError):
        pipeline.write_atomic(target, NEW)

    monkeypatch.setattr(Path, "write_text", real_write_text)
    assert target.read_text() == OLD
    assert json.loads(target.read_text())["names"] == {"SPEAKER_00": "Ospite Uno"}


# Regression: this used to fail.
def test_a_failed_write_does_not_leave_the_scratch_file(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    target.write_text(OLD)
    real_write_text = Path.write_text

    def out_of_space(self, data, *args, **kwargs):
        real_write_text(self, data[: len(data) // 2], *args, **kwargs)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_text", out_of_space)
    with pytest.raises(OSError):
        pipeline.write_atomic(target, NEW)
    monkeypatch.setattr(Path, "write_text", real_write_text)

    assert not (tmp_path / "state.json.part").exists()


def test_write_atomic_round_trips_accents_and_symbols(tmp_path):
    """State and transcripts are dumped with ensure_ascii=False."""
    target = tmp_path / "turns.json"
    text = json.dumps({"text": "Perché però, andiamo — sì. 🎙"}, ensure_ascii=False)
    pipeline.write_atomic(target, text)
    assert json.loads(target.read_text())["text"] == "Perché però, andiamo — sì. 🎙"


def test_write_atomic_handles_a_name_without_a_suffix(tmp_path):
    target = tmp_path / "lockfile"
    pipeline.write_atomic(target, "ok")
    assert target.read_text() == "ok"
    assert not (tmp_path / "lockfile.part").exists()


def test_saving_the_state_goes_through_write_atomic(tmp_path):
    job = make_job(audio_file(tmp_path / "memos"))
    job.state["names"] = {"SPEAKER_00": "Ospite Uno"}
    job._save_state()
    assert json.loads((job.dir / "state.json").read_text())["names"] \
        == {"SPEAKER_00": "Ospite Uno"}
    assert list(job.dir.glob("*.part")) == []


# --------------------------------------------------------------------------- #
# speaker_labels
# --------------------------------------------------------------------------- #

def test_no_embeddings_means_no_labels(tmp_path):
    job = make_job(audio_file(tmp_path / "memos"))
    assert job.speaker_labels() == set()


def test_the_labels_come_from_the_voice_prints(tmp_path):
    job = make_job(audio_file(tmp_path / "memos"))
    write_diar_cache(job, labels=("SPEAKER_00", "SPEAKER_01", "SPEAKER_02"))
    assert job.speaker_labels() == {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02"}


def test_a_typo_in_a_label_is_not_a_known_speaker(tmp_path):
    """What the method exists for: rejecting a name for a speaker that is not there."""
    job = make_job(audio_file(tmp_path / "memos"))
    write_diar_cache(job, labels=("SPEAKER_00", "SPEAKER_01"))
    assert "SPEAKER_1" not in job.speaker_labels()


def test_speaker_labels_releases_the_file(tmp_path):
    job = make_job(audio_file(tmp_path / "memos"))
    write_diar_cache(job)
    job.speaker_labels()
    (job.dir / "embeddings.npz").unlink()          # no handle left open
    assert job.speaker_labels() == set()


# Regression: this used to fail.
def test_a_truncated_npz_is_not_a_crash(tmp_path):
    job = make_job(audio_file(tmp_path / "memos"))
    (job.dir / "embeddings.npz").write_bytes(b"PK\x03\x04 truncated by a ^C")
    assert job.speaker_labels() == set()


# --------------------------------------------------------------------------- #
# ERR_NO_TOKEN: the one string that crosses into Swift
# --------------------------------------------------------------------------- #

def test_err_no_token_is_exactly_this_string():
    """Engine.swift matches on it. A rename here breaks the app, not the tests."""
    assert pipeline.ERR_NO_TOKEN == "[scriba:error:hf-token]"


def test_the_swift_side_spells_it_the_same_way():
    engine = Path(__file__).resolve().parents[1] / "macapp/Sources/Scriba/Engine.swift"
    if not engine.exists():                     # pragma: no cover
        pytest.skip("the macOS app is not in this checkout")
    assert f'"{pipeline.ERR_NO_TOKEN}"' in engine.read_text(encoding="utf-8")


def test_the_missing_token_error_carries_the_marker(tmp_path):
    job = make_job(audio_file(tmp_path / "memos"))
    with pytest.raises(RuntimeError) as err:
        job.diarize()
    message = str(err.value)
    assert pipeline.ERR_NO_TOKEN in message
    # The app strips the marker and shows the rest, so the rest has to be useful.
    human = message.replace(pipeline.ERR_NO_TOKEN, "").strip()
    assert "scriba token" in human and len(human) > 20
