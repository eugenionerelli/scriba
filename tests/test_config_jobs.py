"""Unit tests for scriba.config and scriba.jobs.

Nothing here touches the network, an audio file, a model or the real ~/.scriba.
Every path lives under pytest's tmp_path.

Isolation note: config.py computes DATA_DIR/JOBS_DIR/VOICES_DIR/SETTINGS_PATH from
SCRIBA_HOME *at import time*, and jobs.py copies JOBS_DIR into its own namespace with
`from .config import JOBS_DIR`. So setting the environment variable inside a test is
not enough on its own: the module attributes have to be redirected too, in both
modules. The `scriba_home` fixture does all of that, and `never_touch_real_home`
verifies afterwards that the real directory was left alone.
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest

from scriba import config as config_mod
from scriba import jobs as jobs_mod

Settings = config_mod.Settings


# --------------------------------------------------------------------------- #
# isolation
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def never_touch_real_home():
    """Fail loudly if any test writes into the user's actual ~/.scriba."""
    real = Path.home() / ".scriba"
    before = sorted(p.name for p in real.iterdir()) if real.is_dir() else None
    yield
    after = sorted(p.name for p in real.iterdir()) if real.is_dir() else None
    assert after == before, "a test wrote into the real ~/.scriba"


@pytest.fixture
def scriba_home(tmp_path, monkeypatch):
    """A throwaway SCRIBA_HOME, wired into both modules that cached its pieces."""
    home = tmp_path / "scriba_home"
    jobs_dir = home / "jobs"
    voices_dir = home / "voices"
    jobs_dir.mkdir(parents=True)
    voices_dir.mkdir(parents=True)

    monkeypatch.setenv("SCRIBA_HOME", str(home))
    monkeypatch.setattr(config_mod, "DATA_DIR", home)
    monkeypatch.setattr(config_mod, "VOICES_DIR", voices_dir)
    monkeypatch.setattr(config_mod, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(config_mod, "SETTINGS_PATH", home / "settings.json")
    # jobs.py did `from .config import JOBS_DIR`, so it holds its own binding.
    monkeypatch.setattr(jobs_mod, "JOBS_DIR", jobs_dir)
    return home


@pytest.fixture
def jobs_dir(scriba_home):
    return scriba_home / "jobs"


def test_fixture_really_isolates_the_data_dir(scriba_home, tmp_path):
    for attr in ("DATA_DIR", "VOICES_DIR", "JOBS_DIR", "SETTINGS_PATH"):
        value = getattr(config_mod, attr)
        assert tmp_path in value.parents, f"{attr} escaped tmp_path: {value}"
    assert jobs_mod.JOBS_DIR == config_mod.JOBS_DIR
    assert Path.home() / ".scriba" not in config_mod.SETTINGS_PATH.parents


def test_scriba_home_is_read_at_import_time(tmp_path, monkeypatch):
    """Reloading with SCRIBA_HOME set moves every derived path, and creates nothing."""
    home = tmp_path / "alt_home"
    monkeypatch.setenv("SCRIBA_HOME", str(home))
    try:
        reloaded = importlib.reload(config_mod)
        assert reloaded.DATA_DIR == home
        assert reloaded.JOBS_DIR == home / "jobs"
        assert reloaded.VOICES_DIR == home / "voices"
        assert reloaded.SETTINGS_PATH == home / "settings.json"
        # importing must not create anything on disk
        assert not home.exists()
    finally:
        monkeypatch.undo()
        importlib.reload(config_mod)
        importlib.reload(jobs_mod)
    assert config_mod.DATA_DIR == Path.home() / ".scriba"


def test_ensure_dirs_creates_the_three_directories(tmp_path, monkeypatch):
    home = tmp_path / "fresh"
    monkeypatch.setattr(config_mod, "DATA_DIR", home)
    monkeypatch.setattr(config_mod, "VOICES_DIR", home / "voices")
    monkeypatch.setattr(config_mod, "JOBS_DIR", home / "jobs")
    assert not home.exists()
    config_mod.ensure_dirs()
    assert home.is_dir() and (home / "voices").is_dir() and (home / "jobs").is_dir()
    config_mod.ensure_dirs()  # idempotent


# --------------------------------------------------------------------------- #
# Settings.validate: the values it must refuse
# --------------------------------------------------------------------------- #

def _settings(**overrides) -> Settings:
    s = Settings()
    for key, value in overrides.items():
        assert key in Settings.__dataclass_fields__, f"no such field: {key}"
        setattr(s, key, value)
    return s


def test_default_settings_are_valid():
    Settings().validate()  # must not raise


def test_a_realistic_hand_tuned_configuration_is_valid():
    _settings(
        backend="mlx",
        model="medium",
        language="it",
        min_speakers=2,
        max_speakers=4,
        diarize_device="cpu",
        voice_match_threshold=0.68,
        voice_suggest_threshold=0.52,
        voice_match_margin=0.08,
        voice_min_speech_sec=12.0,
        hotwords=["Nerelli", "pyannote"],
        output_formats=["md", "srt"],
    ).validate()


@pytest.mark.parametrize("field_name, value", [
    ("voice_match_threshold", 5),
    ("voice_match_threshold", 5.0),
    ("voice_match_threshold", -1.5),
    ("voice_suggest_threshold", -2.0),
    ("voice_suggest_threshold", 100.0),
    ("voice_match_margin", 1.01),
    ("voice_match_margin", -3.0),
])
def test_thresholds_outside_minus_one_to_one_are_rejected(field_name, value):
    with pytest.raises(ValueError) as exc:
        _settings(**{field_name: value}).validate()
    message = str(exc.value)
    assert field_name in message
    assert "[-1, 1]" in message


@pytest.mark.parametrize("overrides", [
    {"voice_match_threshold": 1.0},
    {"voice_match_threshold": 0.0, "voice_suggest_threshold": 0.0},
    {"voice_match_threshold": -1.0, "voice_suggest_threshold": -1.0},
    {"voice_match_margin": 1.0},
    {"voice_match_margin": -1.0},
    {"voice_suggest_threshold": -1.0},
])
def test_thresholds_on_the_boundary_are_accepted(overrides):
    _settings(**overrides).validate()


def test_suggest_threshold_above_match_threshold_is_rejected():
    with pytest.raises(ValueError) as exc:
        _settings(voice_match_threshold=0.60, voice_suggest_threshold=0.80).validate()
    message = str(exc.value)
    assert "voice_suggest_threshold" in message
    assert "voice_match_threshold" in message
    assert "0.8" in message and "0.6" in message


def test_suggest_threshold_equal_to_match_threshold_is_accepted():
    _settings(voice_match_threshold=0.7, voice_suggest_threshold=0.7).validate()


@pytest.mark.parametrize("field_name", ["min_speakers", "max_speakers"])
@pytest.mark.parametrize("value", [0, -1, 2.5, "2", [2]])
def test_non_integer_or_non_positive_speaker_counts_are_rejected(field_name, value):
    with pytest.raises(ValueError) as exc:
        _settings(**{field_name: value}).validate()
    message = str(exc.value)
    assert field_name in message
    assert "whole number" in message


def test_speaker_counts_may_be_none_or_positive_integers():
    _settings(min_speakers=None, max_speakers=None).validate()
    _settings(min_speakers=1, max_speakers=1).validate()
    _settings(min_speakers=2, max_speakers=9).validate()


def test_min_speakers_above_max_speakers_is_rejected():
    with pytest.raises(ValueError) as exc:
        _settings(min_speakers=5, max_speakers=2).validate()
    message = str(exc.value)
    assert "min_speakers (5) is above" in message
    assert "max_speakers (2)" in message


def test_min_speakers_equal_to_max_speakers_is_accepted():
    _settings(min_speakers=3, max_speakers=3).validate()


@pytest.mark.parametrize("value", ["md", "source", "md,srt", ""])
def test_output_formats_as_a_bare_string_is_rejected(value):
    """A string here makes write_all iterate over characters and write nothing."""
    with pytest.raises(ValueError) as exc:
        _settings(output_formats=value).validate()
    assert "expected a list of names" in str(exc.value)


@pytest.mark.parametrize("value", [[], None, ("md", "srt"), {"md": 1}])
def test_output_formats_must_be_a_non_empty_list(value):
    with pytest.raises(ValueError) as exc:
        _settings(output_formats=value).validate()
    assert "expected a list of names" in str(exc.value)


@pytest.mark.parametrize("bad", ["docx", "pdf", "MD", "notebooklm", "txt "])
def test_unknown_output_format_names_are_rejected(bad):
    with pytest.raises(ValueError) as exc:
        _settings(output_formats=["md", bad]).validate()
    message = str(exc.value)
    assert bad in message
    assert "known:" in message


def test_every_known_output_format_is_accepted():
    _settings(output_formats=sorted(config_mod.KNOWN_FORMATS)).validate()


@pytest.mark.parametrize("value", ["ciao", None, ("a", "b"), 3])
def test_hotwords_must_be_a_list(value):
    with pytest.raises(ValueError) as exc:
        _settings(hotwords=value).validate()
    assert "expected a list of words" in str(exc.value)


def test_hotwords_empty_or_populated_list_is_accepted():
    _settings(hotwords=[]).validate()
    _settings(hotwords=["Buttussi", "EvalGuard"]).validate()


@pytest.mark.parametrize("value", ["cuda", "gpu", "CPU", "Auto", "", None, "metal"])
def test_unknown_diarize_device_is_rejected(value):
    with pytest.raises(ValueError) as exc:
        _settings(diarize_device=value).validate()
    assert "expected auto, cpu or mps" in str(exc.value)


@pytest.mark.parametrize("value", ["auto", "cpu", "mps"])
def test_supported_diarize_devices_are_accepted(value):
    _settings(diarize_device=value).validate()


@pytest.mark.parametrize("value", [-0.1, -1, -900.0])
def test_negative_voice_min_speech_sec_is_rejected(value):
    with pytest.raises(ValueError) as exc:
        _settings(voice_min_speech_sec=value).validate()
    assert "cannot be negative" in str(exc.value)


def test_zero_voice_min_speech_sec_is_accepted():
    _settings(voice_min_speech_sec=0).validate()


def test_validate_reports_every_problem_in_one_message():
    """The point of collecting into `problems` is not stopping at the first one."""
    with pytest.raises(ValueError) as exc:
        _settings(
            voice_match_threshold=4.0,
            min_speakers=0,
            output_formats=["docx"],
            hotwords="ciao",
            diarize_device="cuda",
            voice_min_speech_sec=-2,
        ).validate()
    message = str(exc.value)
    assert message.startswith("settings out of range:")
    for expected in ("voice_match_threshold", "min_speakers", "docx",
                     "hotwords", "diarize_device", "voice_min_speech_sec"):
        assert expected in message, f"{expected} missing from {message!r}"
    assert len(message.strip().splitlines()) >= 7


# --- bugs found while writing the above ------------------------------------- #

# Regression: this used to fail.
@pytest.mark.parametrize("value", [None, "high"])
def test_non_numeric_threshold_should_raise_valueerror(value):
    with pytest.raises(ValueError):
        _settings(voice_match_threshold=value).validate()


# Regression: this used to fail.
def test_non_integer_min_speakers_should_raise_valueerror():
    with pytest.raises(ValueError):
        _settings(min_speakers="2", max_speakers=5).validate()


# --------------------------------------------------------------------------- #
# Settings.load / Settings.save
# --------------------------------------------------------------------------- #

def _write_settings_file(home: Path, payload: dict) -> Path:
    path = home / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def test_load_returns_defaults_when_no_file_exists(scriba_home):
    assert not config_mod.SETTINGS_PATH.exists()
    assert asdict(Settings.load()) == asdict(Settings())


def test_save_writes_under_the_isolated_home_and_creates_dirs(tmp_path, monkeypatch):
    home = tmp_path / "unmade"
    monkeypatch.setattr(config_mod, "DATA_DIR", home)
    monkeypatch.setattr(config_mod, "VOICES_DIR", home / "voices")
    monkeypatch.setattr(config_mod, "JOBS_DIR", home / "jobs")
    monkeypatch.setattr(config_mod, "SETTINGS_PATH", home / "settings.json")

    Settings().save()

    assert (home / "settings.json").is_file()
    assert (home / "voices").is_dir() and (home / "jobs").is_dir()
    assert json.loads((home / "settings.json").read_text())["model"] == "large-v3"


def test_save_load_round_trip(scriba_home):
    original = _settings(
        backend="mlx",
        model="small",
        language="es",
        batch_size=16,
        align=False,
        initial_prompt="una conversazione tra due ricercatori",
        hotwords=["Olivera", "mechint"],
        min_speakers=2,
        max_speakers=6,
        diarize_device="mps",
        voice_match_threshold=0.81,
        voice_suggest_threshold=0.49,
        voice_match_margin=0.11,
        voice_min_speech_sec=15.5,
        output_formats=["source", "txt"],
        timestamp_every=30,
    )
    original.save()
    assert asdict(Settings.load()) == asdict(original)


def test_round_trip_survives_two_passes(scriba_home):
    Settings.load().save()
    first = config_mod.SETTINGS_PATH.read_text()
    Settings.load().save()
    assert config_mod.SETTINGS_PATH.read_text() == first


def test_load_ignores_keys_that_are_not_settings(scriba_home):
    _write_settings_file(scriba_home, {
        "model": "medium",
        "obsolete_option": "whatever",
        "notebooklm_folder": "/nowhere",
    })
    loaded = Settings.load()
    assert loaded.model == "medium"
    assert not hasattr(loaded, "obsolete_option")
    assert loaded.output_formats == Settings().output_formats


def test_load_migrates_the_legacy_notebooklm_format_name(scriba_home):
    """A settings file written before the rename must still load, not hard-error."""
    _write_settings_file(scriba_home, {
        "output_formats": ["notebooklm", "md", "srt"],
    })
    loaded = Settings.load()  # must not raise
    assert loaded.output_formats == ["source", "md", "srt"]


def test_migration_handles_a_file_whose_only_format_is_the_old_name(scriba_home):
    _write_settings_file(scriba_home, {"output_formats": ["notebooklm"]})
    assert Settings.load().output_formats == ["source"]


def test_migration_leaves_current_format_names_alone(scriba_home):
    formats = ["source", "md", "json", "srt", "vtt", "txt"]
    _write_settings_file(scriba_home, {"output_formats": formats})
    assert Settings.load().output_formats == formats


def test_a_migrated_file_can_be_saved_back_without_the_legacy_name(scriba_home):
    _write_settings_file(scriba_home, {"output_formats": ["notebooklm", "md"]})
    Settings.load().save()
    stored = json.loads(config_mod.SETTINGS_PATH.read_text())
    assert stored["output_formats"] == ["source", "md"]
    assert "notebooklm" not in config_mod.SETTINGS_PATH.read_text()


def test_load_rejects_a_stored_value_that_is_out_of_range(scriba_home):
    _write_settings_file(scriba_home, {"voice_match_threshold": 5})
    with pytest.raises(ValueError) as exc:
        Settings.load()
    assert "voice_match_threshold" in str(exc.value)


def test_load_rejects_a_stored_unknown_format(scriba_home):
    _write_settings_file(scriba_home, {"output_formats": ["md", "docx"]})
    with pytest.raises(ValueError) as exc:
        Settings.load()
    assert "docx" in str(exc.value)


def test_save_validates_before_writing_anything(scriba_home):
    bad = _settings(output_formats="md")
    with pytest.raises(ValueError):
        bad.save()
    assert not config_mod.SETTINGS_PATH.exists(), "an invalid settings file was written"


def test_save_does_not_clobber_a_good_file_with_a_bad_one(scriba_home):
    _settings(model="medium").save()
    before = config_mod.SETTINGS_PATH.read_text()
    with pytest.raises(ValueError):
        _settings(diarize_device="cuda").save()
    assert config_mod.SETTINGS_PATH.read_text() == before


def test_saved_file_is_readable_json_with_every_field(scriba_home):
    Settings().save()
    stored = json.loads(config_mod.SETTINGS_PATH.read_text())
    assert set(stored) == set(Settings.__dataclass_fields__)


def test_non_ascii_values_survive_the_round_trip(scriba_home):
    _settings(initial_prompt="perché sì, però", hotwords=["Nerelli", "città"]).save()
    loaded = Settings.load()
    assert loaded.initial_prompt == "perché sì, però"
    assert loaded.hotwords == ["Nerelli", "città"]


# --------------------------------------------------------------------------- #
# hf_token / keychain (no subprocess is ever actually spawned)
# --------------------------------------------------------------------------- #

class _FakeCompleted:
    def __init__(self, stdout=""):
        self.stdout = stdout
        self.returncode = 0


def test_keychain_get_returns_the_stored_token(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _FakeCompleted("hf_finto_token\n")

    monkeypatch.setattr(config_mod.subprocess, "run", fake_run)
    assert config_mod.keychain_get() == "hf_finto_token"
    assert calls == [["security", "find-generic-password", "-s",
                      config_mod.KEYCHAIN_SERVICE, "-w"]]


def test_keychain_get_treats_an_empty_entry_as_absent(monkeypatch):
    monkeypatch.setattr(config_mod.subprocess, "run",
                        lambda cmd, **kw: _FakeCompleted("  \n"))
    assert config_mod.keychain_get() is None


def test_keychain_set_writes_through_the_security_command(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _FakeCompleted()

    monkeypatch.setattr(config_mod.os, "getlogin", lambda: "tester")
    monkeypatch.setattr(config_mod.subprocess, "run", fake_run)

    config_mod.keychain_set("hf_finto_token")

    (cmd, kwargs), = calls
    assert cmd[:2] == ["security", "add-generic-password"]
    assert "-U" in cmd                                   # update in place
    assert cmd[cmd.index("-s") + 1] == config_mod.KEYCHAIN_SERVICE
    assert cmd[cmd.index("-w") + 1] == "hf_finto_token"
    assert kwargs.get("check") is True


def test_keychain_get_returns_none_when_the_entry_is_missing(monkeypatch):
    import subprocess

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(44, args[0] if args else "security")

    monkeypatch.setattr(config_mod.subprocess, "run", fake_run)
    assert config_mod.keychain_get() is None


def test_keychain_get_returns_none_off_macos(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("security")

    monkeypatch.setattr(config_mod.subprocess, "run", fake_run)
    assert config_mod.keychain_get() is None


def test_hf_token_prefers_the_keychain_then_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setattr(config_mod, "keychain_get", lambda: "from-keychain")
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert config_mod.hf_token() == "from-keychain"

    monkeypatch.setattr(config_mod, "keychain_get", lambda: None)
    assert config_mod.hf_token() == "from-env"

    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "from-other-env")
    assert config_mod.hf_token() == "from-other-env"

    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN")
    assert config_mod.hf_token() is None


# --------------------------------------------------------------------------- #
# jobs.inventory
# --------------------------------------------------------------------------- #

NO_OUTPUT = object()


def make_job(jobs_root: Path, name: str, *, state: dict | None = None,
             raw_state: str | bytes | None = None, transcript: bool = False,
             diarization: bool = False, output=NO_OUTPUT,
             wav_bytes: int = 0, extra: dict[str, str] | None = None) -> Path:
    """A job folder with only the files jobs.py actually looks at."""
    job = jobs_root / name
    job.mkdir(parents=True)
    if raw_state is not None:
        path = job / "state.json"
        path.write_bytes(raw_state) if isinstance(raw_state, bytes) else path.write_text(raw_state)
    elif state is not None:
        (job / "state.json").write_text(json.dumps(state))
    if transcript:
        (job / "transcript.json").write_text(json.dumps({"segments": [{"text": "ciao"}]}))
    if diarization:
        (job / "diarization.json").write_text(json.dumps([{"speaker": "SPEAKER_00"}]))
    if output is not NO_OUTPUT:
        out = job / "output"
        out.mkdir()
        for filename in output:
            (out / filename).write_text("contenuto")
    if wav_bytes:
        (job / "audio16k.wav").write_bytes(b"\0" * wav_bytes)
    for filename, content in (extra or {}).items():
        (job / filename).write_text(content)
    return job


def by_name(rows) -> dict[str, object]:
    return {row.path.name: row for row in rows}


def test_inventory_is_empty_when_the_jobs_directory_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_mod, "JOBS_DIR", tmp_path / "never-created")
    assert jobs_mod.inventory() == []


def test_inventory_is_empty_when_there_are_no_jobs(jobs_dir):
    assert jobs_mod.inventory() == []


def test_inventory_skips_loose_files_in_the_jobs_directory(jobs_dir):
    (jobs_dir / ".DS_Store").write_text("junk")
    (jobs_dir / "notes.txt").write_text("junk")
    make_job(jobs_dir, "colloquio-a1b2", state={"source": "/audio/colloquio.m4a"})
    rows = jobs_mod.inventory()
    assert [row.path.name for row in rows] == ["colloquio-a1b2"]


def test_inventory_derives_every_state(jobs_dir):
    make_job(jobs_dir, "done-0001", state={"source": "/audio/uno.m4a"},
             transcript=True, diarization=True, output=["uno.md"])
    make_job(jobs_dir, "transcribed-0002", state={"source": "/audio/due.m4a"},
             transcript=True, diarization=True)
    make_job(jobs_dir, "voices-0003", state={"source": "/audio/tre.m4a"},
             diarization=True)
    make_job(jobs_dir, "text-0004", state={"source": "/audio/quattro.m4a"},
             transcript=True)
    make_job(jobs_dir, "nothing-0005", state={"source": "/audio/cinque.m4a"})

    rows = by_name(jobs_mod.inventory())
    assert rows["done-0001"].state == "done"
    assert rows["transcribed-0002"].state == "transcribed"
    assert rows["voices-0003"].state == "voices only"
    assert rows["text-0004"].state == "text only"
    assert rows["nothing-0005"].state == "nothing"
    assert rows["done-0001"].has_output is True
    assert rows["transcribed-0002"].has_output is False


def test_a_job_with_no_files_at_all_is_nothing(jobs_dir):
    make_job(jobs_dir, "vuoto-0006")
    row = jobs_mod.inventory()[0]
    assert row.state == "nothing"
    assert row.source_name == "vuoto-0006"
    assert row.recorded == ""
    assert row.duration == 0.0
    assert row.speakers == 0
    assert row.names == {}


def test_an_empty_output_directory_does_not_count_as_done(jobs_dir):
    make_job(jobs_dir, "svuotato-0007", state={"source": "/audio/sei.m4a"},
             transcript=True, diarization=True, output=[])
    make_job(jobs_dir, "svuotato-0008", state={"source": "/audio/sette.m4a"}, output=[])
    rows = by_name(jobs_mod.inventory())
    assert rows["svuotato-0007"].state == "transcribed"
    assert rows["svuotato-0007"].has_output is False
    assert rows["svuotato-0008"].state == "nothing"


def test_output_present_wins_over_the_missing_intermediates(jobs_dir):
    make_job(jobs_dir, "esportato-0009", state={"source": "/audio/otto.m4a"},
             output=["otto.md", "otto.srt"])
    assert jobs_mod.inventory()[0].state == "done"


def test_inventory_reads_the_fields_it_reports(jobs_dir):
    make_job(jobs_dir, "riunione-9f3c", state={
        "source": "/Volumes/Registrazioni/riunione con Otello.m4a",
        "recorded": "2026-03-14T09:15:00+01:00",
        "duration": 3671.5,
        "matches": {"SPEAKER_00": "vittoria", "SPEAKER_01": "otello"},
        "names": {"SPEAKER_00": "Vittoria", "SPEAKER_01": "Otello"},
    }, transcript=True, diarization=True, output=["riunione.md"],
        wav_bytes=1_048_576)

    row = jobs_mod.inventory()[0]
    assert row.source_name == "riunione con Otello.m4a"
    assert row.source_path == "/Volumes/Registrazioni/riunione con Otello.m4a"
    assert row.recorded == "2026-03-14"          # truncated to the date
    assert row.duration == pytest.approx(3671.5)
    assert row.speakers == 2
    assert row.names == {"SPEAKER_00": "Vittoria", "SPEAKER_01": "Otello"}
    assert row.audio_mb == pytest.approx(1.0)
    assert row.size_mb >= row.audio_mb
    assert row.state == "done"


def test_audio_mb_is_zero_when_the_prepared_wav_is_gone(jobs_dir):
    make_job(jobs_dir, "senza-audio-0010", state={"source": "/audio/nove.m4a"},
             transcript=True)
    assert jobs_mod.inventory()[0].audio_mb == 0.0


def test_missing_state_keys_fall_back_instead_of_raising(jobs_dir):
    make_job(jobs_dir, "parziale-0011", state={"recorded": None, "duration": None,
                                               "matches": None, "names": None})
    row = jobs_mod.inventory()[0]
    assert row.recorded == ""
    assert row.duration == 0.0
    assert row.speakers == 0
    assert row.names == {}
    assert row.source_name == "parziale-0011"


def test_inventory_sorts_newest_recording_first(jobs_dir):
    make_job(jobs_dir, "b-vecchio", state={"source": "/a/b.m4a", "recorded": "2025-12-31T10:00:00"})
    make_job(jobs_dir, "c-recente", state={"source": "/a/c.m4a", "recorded": "2026-07-20T10:00:00"})
    make_job(jobs_dir, "a-medio", state={"source": "/a/a.m4a", "recorded": "2026-01-05T10:00:00"})
    assert [row.recorded for row in jobs_mod.inventory()] == [
        "2026-07-20", "2026-01-05", "2025-12-31"]


def test_jobs_without_a_recording_date_sort_last(jobs_dir):
    make_job(jobs_dir, "datato", state={"source": "/a/datato.m4a", "recorded": "2020-01-01"})
    make_job(jobs_dir, "senza-data", state={"source": "/a/senza.m4a"})
    assert [row.source_name for row in jobs_mod.inventory()] == ["datato.m4a", "senza.m4a"]


def test_same_day_jobs_are_ordered_by_source_name(jobs_dir):
    day = "2026-05-05T08:00:00"
    make_job(jobs_dir, "uno", state={"source": "/a/aaa.m4a", "recorded": day})
    make_job(jobs_dir, "due", state={"source": "/a/zzz.m4a", "recorded": day})
    make_job(jobs_dir, "tre", state={"source": "/a/mmm.m4a", "recorded": day})
    # reverse=True applies to the whole sort field, so names run Z to A within a day
    assert [row.source_name for row in jobs_mod.inventory()] == [
        "zzz.m4a", "mmm.m4a", "aaa.m4a"]


def test_a_job_with_corrupt_state_json_is_still_listed(jobs_dir):
    """A folder that failed halfway is exactly what somebody runs this to find."""
    make_job(jobs_dir, "rotto-0012", raw_state='{"source": "/a/rotto.m4a", trunc',
             transcript=True, diarization=True, wav_bytes=2048)
    make_job(jobs_dir, "sano-0013", state={"source": "/a/sano.m4a",
                                           "recorded": "2026-02-02T00:00:00"})

    rows = by_name(jobs_mod.inventory())
    assert set(rows) == {"rotto-0012", "sano-0013"}
    broken = rows["rotto-0012"]
    assert broken.state == "transcribed"     # derived from the files, not the JSON
    assert broken.recorded == ""
    assert broken.source_path == ""
    assert broken.source_name == "rotto-0012"
    assert broken.audio_mb > 0


def test_an_empty_state_json_is_still_listed(jobs_dir):
    make_job(jobs_dir, "vuoto-0014", raw_state="", diarization=True)
    row = jobs_mod.inventory()[0]
    assert row.state == "voices only"
    assert row.recorded == ""


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root can read a mode-000 file")
def test_an_unreadable_state_json_is_still_listed(jobs_dir):
    job = make_job(jobs_dir, "protetto-0015", state={"source": "/a/protetto.m4a"},
                   transcript=True)
    state_path = job / "state.json"
    state_path.chmod(0o000)
    try:
        rows = jobs_mod.inventory()
    finally:
        state_path.chmod(0o644)
    assert [row.path.name for row in rows] == ["protetto-0015"]
    assert rows[0].state == "text only"
    assert rows[0].source_path == ""


# Regression: this used to fail.
def test_a_state_json_with_invalid_utf8_is_still_listed(jobs_dir):
    make_job(jobs_dir, "bytes-0016", raw_state=b'\xff\xfe{"source": "/a/x.m4a"}',
             transcript=True, diarization=True)
    make_job(jobs_dir, "sano-0017", state={"source": "/a/sano.m4a"})
    rows = by_name(jobs_mod.inventory())
    assert set(rows) == {"bytes-0016", "sano-0017"}
    assert rows["bytes-0016"].state == "transcribed"


# Regression: this used to fail.
@pytest.mark.parametrize("payload", ["null", "[]", '"stato"'])
def test_a_state_json_that_is_not_an_object_is_still_listed(jobs_dir, payload):
    make_job(jobs_dir, "nonoggetto-0018", raw_state=payload,
             transcript=True, diarization=True)
    rows = jobs_mod.inventory()
    assert [row.path.name for row in rows] == ["nonoggetto-0018"]
    assert rows[0].state == "transcribed"


# --------------------------------------------------------------------------- #
# jobs.prune
# --------------------------------------------------------------------------- #

def build_all_states(jobs_dir: Path) -> dict[str, Path]:
    """One job of every state, each holding a prepared wav."""
    return {
        "done": make_job(jobs_dir, "done-1", state={"source": "/a/done.m4a",
                                                    "recorded": "2026-01-01T00:00:00"},
                         transcript=True, diarization=True, output=["done.md"],
                         wav_bytes=4096),
        "transcribed": make_job(jobs_dir, "transcribed-2",
                                state={"source": "/a/tr.m4a",
                                       "recorded": "2026-01-02T00:00:00"},
                                transcript=True, diarization=True, wav_bytes=4096),
        "voices": make_job(jobs_dir, "voices-3", state={"source": "/a/vo.m4a",
                                                        "recorded": "2026-01-03T00:00:00"},
                           diarization=True, wav_bytes=4096,
                           extra={"embeddings.npz": "finti byte"}),
        "text": make_job(jobs_dir, "text-4", state={"source": "/a/te.m4a",
                                                     "recorded": "2026-01-04T00:00:00"},
                         transcript=True, wav_bytes=4096),
        "nothing": make_job(jobs_dir, "nothing-5", state={"source": "/a/no.m4a",
                                                          "recorded": "2026-01-05T00:00:00"},
                            wav_bytes=4096),
    }


def snapshot(paths: dict[str, Path]) -> dict[Path, bool]:
    files = {}
    for job in paths.values():
        for name in ("transcript.json", "diarization.json", "state.json",
                     "audio16k.wav", "embeddings.npz"):
            files[job / name] = (job / name).exists()
        out = job / "output"
        for f in (out.rglob("*") if out.exists() else []):
            files[f] = True
    return files


def test_prune_dry_run_removes_nothing_but_reports_what_it_would(jobs_dir):
    jobs = build_all_states(jobs_dir)
    before = snapshot(jobs)

    targets = jobs_mod.prune(jobs_mod.inventory(), audio=True, empty=True, dry_run=True)

    assert targets, "a dry run still has to say what it would remove"
    assert all(path.exists() for path, _ in targets)
    assert snapshot(jobs) == before
    assert all(job.is_dir() for job in jobs.values())


def test_dry_run_is_the_default(jobs_dir):
    jobs = build_all_states(jobs_dir)
    before = snapshot(jobs)
    jobs_mod.prune(jobs_mod.inventory(), audio=True, empty=True)
    assert snapshot(jobs) == before


def test_prune_with_both_flags_off_selects_nothing(jobs_dir):
    jobs = build_all_states(jobs_dir)
    before = snapshot(jobs)
    assert jobs_mod.prune(jobs_mod.inventory(), audio=False, empty=False,
                          dry_run=False) == []
    assert snapshot(jobs) == before


def test_prune_audio_removes_only_the_rebuildable_wav(jobs_dir):
    jobs = build_all_states(jobs_dir)

    targets = jobs_mod.prune(jobs_mod.inventory(), audio=True, empty=False,
                             dry_run=False)

    assert {path.name for path, _ in targets} == {"audio16k.wav"}
    for name, job in jobs.items():
        assert job.is_dir(), f"{name}: the job folder was removed"
        assert not (job / "audio16k.wav").exists()
        assert (job / "state.json").exists()
    assert (jobs["done"] / "transcript.json").exists()
    assert (jobs["done"] / "diarization.json").exists()
    assert (jobs["done"] / "output" / "done.md").exists()
    assert (jobs["voices"] / "embeddings.npz").exists()


def test_prune_empty_removes_only_the_jobs_that_produced_nothing(jobs_dir):
    jobs = build_all_states(jobs_dir)

    targets = jobs_mod.prune(jobs_mod.inventory(), audio=False, empty=True,
                             dry_run=False)

    assert [path for path, _ in targets] == [jobs["nothing"]]
    assert not jobs["nothing"].exists()
    for name in ("done", "transcribed", "voices", "text"):
        assert jobs[name].is_dir()
        assert (jobs[name] / "audio16k.wav").exists(), "audio went with empty=False"


def test_prune_never_removes_a_transcript_or_a_diarization(jobs_dir):
    """The one guarantee of this function: only rebuildable things go."""
    jobs = build_all_states(jobs_dir)
    protected = [
        jobs["done"] / "transcript.json", jobs["done"] / "diarization.json",
        jobs["done"] / "output" / "done.md",
        jobs["transcribed"] / "transcript.json", jobs["transcribed"] / "diarization.json",
        jobs["voices"] / "diarization.json", jobs["voices"] / "embeddings.npz",
        jobs["text"] / "transcript.json",
    ]
    assert all(p.exists() for p in protected)

    targets = jobs_mod.prune(jobs_mod.inventory(), audio=True, empty=True,
                             dry_run=False)

    for path in protected:
        assert path.exists(), f"prune removed work that costs minutes of CPU: {path}"
    # and nothing outside the two allowed kinds of target was even selected
    for path, _ in targets:
        assert path.name == "audio16k.wav" or path == jobs["nothing"], path
    assert not (jobs["done"] / "audio16k.wav").exists()


def test_prune_does_not_delete_a_job_whose_state_json_is_corrupt(jobs_dir):
    """State comes from the files on disk, so unreadable JSON never looks empty."""
    job = make_job(jobs_dir, "rotto-9", raw_state="{ non json",
                   transcript=True, diarization=True, wav_bytes=4096)
    jobs_mod.prune(jobs_mod.inventory(), audio=True, empty=True, dry_run=False)
    assert job.is_dir()
    assert (job / "transcript.json").exists()
    assert (job / "diarization.json").exists()
    assert not (job / "audio16k.wav").exists()


def test_an_empty_job_is_removed_whole_and_not_targeted_twice(jobs_dir):
    job = make_job(jobs_dir, "fallito-10", state={"source": "/a/f.m4a"}, wav_bytes=4096)
    targets = jobs_mod.prune(jobs_mod.inventory(), audio=True, empty=True, dry_run=True)
    assert [path for path, _ in targets] == [job]


def test_prune_skips_jobs_that_have_no_prepared_audio(jobs_dir):
    make_job(jobs_dir, "senza-wav-11", state={"source": "/a/s.m4a"}, transcript=True)
    assert jobs_mod.prune(jobs_mod.inventory(), audio=True, empty=True,
                          dry_run=False) == []


def test_prune_reports_the_size_it_would_free(jobs_dir):
    make_job(jobs_dir, "grosso-12", state={"source": "/a/g.m4a"},
             transcript=True, diarization=True, wav_bytes=2 * 1_048_576)
    empty_job = make_job(jobs_dir, "vuoto-13", state={"source": "/a/v.m4a"},
                         wav_bytes=1_048_576)

    sizes = dict(jobs_mod.prune(jobs_mod.inventory(), audio=True, empty=True,
                                dry_run=True))

    assert sizes[jobs_dir / "grosso-12" / "audio16k.wav"] == pytest.approx(2.0)
    assert sizes[empty_job] == pytest.approx(1.0, abs=0.01)  # whole folder


def test_prune_on_an_empty_inventory_is_a_no_op(jobs_dir):
    assert jobs_mod.prune([], audio=True, empty=True, dry_run=False) == []


def test_prune_run_twice_does_not_raise(jobs_dir):
    jobs = build_all_states(jobs_dir)
    rows = jobs_mod.inventory()
    jobs_mod.prune(rows, audio=True, empty=True, dry_run=False)
    jobs_mod.prune(rows, audio=True, empty=True, dry_run=False)  # stale rows, same call
    assert jobs["done"].is_dir()
    assert (jobs["done"] / "transcript.json").exists()


def test_inventory_after_pruning_audio_shows_the_same_jobs(jobs_dir):
    build_all_states(jobs_dir)
    before = {row.path.name: row.state for row in jobs_mod.inventory()}
    jobs_mod.prune(jobs_mod.inventory(), audio=True, empty=False, dry_run=False)
    after = {row.path.name: row.state for row in jobs_mod.inventory()}
    assert after == before
    assert all(row.audio_mb == 0.0 for row in jobs_mod.inventory())
