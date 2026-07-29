"""Tests for `scriba.lang` (multi-window language vote) and `scriba.audio`
(timestamp provenance, atomic conversion).

Nothing here loads a model, opens an audio file or touches the network:
`soundfile` and `faster_whisper` are replaced in `sys.modules` before
`lang.detect()` imports them, `ffprobe`/`ffmpeg` are replaced with fakes, and
every file lives under `tmp_path`.

The two modules are tested together because they answer the same question from
two sides: *what is this recording, really*. Getting either one wrong produces a
document that is fluent, plausible and false, with nothing on screen to say so.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from scriba import audio, lang

SR = lang.SAMPLE_RATE          # 16 000
DURATION = 120.0               # long enough that all five windows are usable


# --------------------------------------------------------------------------
# fake backends for lang.detect()
# --------------------------------------------------------------------------

@dataclass
class Backend:
    """Records what `lang.detect()` asked the (fake) libraries to do."""
    model_args: list = field(default_factory=list)
    model_kwargs: list = field(default_factory=list)
    windows: list = field(default_factory=list)      # audio handed to detect_language
    read_paths: list = field(default_factory=list)
    deleted: list = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return len(self.windows)


def install_backends(monkeypatch, *, results, duration=DURATION, sr=SR, channels=1):
    """Replace soundfile + faster_whisper.

    `results` is the per-window outcome, in order: either a `(language,
    probability)` pair or an exception instance, which the fake raises.
    """
    back = Backend()
    n = int(duration * sr)
    if channels == 1:
        data = np.zeros(n, dtype=np.float32)
    else:
        data = np.zeros((n, channels), dtype=np.float32)

    sf_mod = types.ModuleType("soundfile")

    def _read(path, dtype="float32"):
        back.read_paths.append(path)
        return data, sr

    sf_mod.read = _read

    pending = list(results)

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            back.model_args.append(args)
            back.model_kwargs.append(kwargs)

        def detect_language(self, audio=None):
            back.windows.append(audio)
            if not pending:
                raise AssertionError(
                    f"detect_language called {back.n_windows} times, "
                    f"only {len(results)} results were provided"
                )
            item = pending.pop(0)
            if isinstance(item, BaseException):
                raise item
            language, prob = item
            # faster-whisper returns (language, probability, all_probs)
            return language, prob, {language: prob}

        def __del__(self):
            back.deleted.append(True)

    fw_mod = types.ModuleType("faster_whisper")
    fw_mod.WhisperModel = FakeWhisperModel

    monkeypatch.setitem(sys.modules, "soundfile", sf_mod)
    monkeypatch.setitem(sys.modules, "faster_whisper", fw_mod)
    return back


@pytest.fixture
def wav(tmp_path):
    """A path that is never actually read: soundfile is faked."""
    p = tmp_path / "audio16k.wav"
    p.write_bytes(b"RIFF")
    return p


# ==========================================================================
# lang.detect(): the vote
# ==========================================================================

def test_clear_majority_wins(monkeypatch, wav):
    back = install_backends(monkeypatch, results=[
        ("es", 0.94), ("es", 0.91), ("es", 0.88), ("it", 0.83), ("es", 0.95),
    ])
    guess = lang.detect(wav)

    assert guess.language == "es"
    assert guess.reliable is True
    assert guess.confidence > 0.6
    assert back.n_windows == 5
    assert set(guess.votes) == {"es", "it"}
    assert guess.votes["es"] > guess.votes["it"]


def test_small_talk_in_the_wrong_language_does_not_decide_the_file(monkeypatch, wav):
    """The failure this module exists for.

    Whisper looks at the first thirty seconds. Here those thirty seconds are
    Italian small talk and the conversation that follows is Spanish. Sampling
    the first window alone gets it wrong; the vote gets it right.
    """
    back = install_backends(monkeypatch, results=[
        ("it", 0.87),                                   # "ciao, come stai" at the top
        ("es", 0.96), ("es", 0.97), ("es", 0.95), ("es", 0.96),
    ])
    guess = lang.detect(wav)

    assert guess.samples[0][1] == "it"     # the first window really did say Italian
    assert guess.language == "es"          # and it did not decide the file
    assert guess.reliable is True
    assert back.n_windows == 5


def test_windows_are_spread_across_the_whole_file(monkeypatch, wav):
    install_backends(monkeypatch, results=[("es", 0.9)] * 5)
    guess = lang.detect(wav)

    offsets = [t for t, _lang, _p in guess.samples]
    assert offsets == sorted(offsets)
    # Not five looks at the same opening minute.
    assert offsets[-1] - offsets[0] > 0.6 * DURATION
    # And nothing sampled from the very start or the very end.
    assert offsets[0] > 0.0
    assert offsets[-1] < DURATION


def test_close_vote_is_reported_as_uncertain_not_decided(monkeypatch, wav):
    """Two languages neck and neck: say so, do not pick one quietly."""
    install_backends(monkeypatch, results=[
        ("es", 0.90), ("it", 0.90), ("es", 0.90), ("it", 0.90),
    ] + [("it", 0.90)])
    guess = lang.detect(wav)

    assert guess.reliable is False
    assert guess.confidence < 0.6
    assert "unclear" in guess.note
    # Both candidates are named, so a human can go and check.
    assert "es" in guess.note and "it" in guess.note
    assert "hand" in guess.note


def test_exact_tie_is_never_reported_as_reliable(monkeypatch, wav):
    install_backends(monkeypatch, results=[
        ("es", 0.9), ("it", 0.9), ("es", 0.9), ("it", 0.9),
    ])
    # one window skipped on purpose: 4 usable results, 5 windows requested
    install_backends(monkeypatch, results=[
        ("es", 0.9), ("it", 0.9), ("es", 0.9), ("it", 0.9), ("es", 0.9),
    ])
    guess = lang.detect(wav)
    # 3 x 0.9 vs 2 x 0.9 -> 0.6 exactly at the threshold, still the closest call
    assert guess.confidence == pytest.approx(0.6, abs=1e-9)
    assert guess.language == "es"


def test_one_confident_window_is_not_outvoted_by_several_weak_ones(monkeypatch, wav):
    """A single sure window beats four hesitant ones.

    Below 0.5 the model is guessing, and the code weighs those windows at a
    quarter. Without that discount the four weak windows would win 1.80 to 0.97.
    """
    install_backends(monkeypatch, results=[
        ("it", 0.45), ("it", 0.45), ("es", 0.97), ("it", 0.45), ("it", 0.45),
    ])
    guess = lang.detect(wav)

    assert guess.language == "es"
    assert guess.votes["es"] == pytest.approx(0.97)
    assert guess.votes["it"] == pytest.approx(4 * 0.45 * 0.25)
    # sanity: unweighted, Italian would have won
    assert 4 * 0.45 > 0.97
    assert guess.reliable is True


def test_weak_windows_are_weighted_at_a_quarter(monkeypatch, wav):
    install_backends(monkeypatch, results=[
        ("es", 0.50),   # exactly at the threshold: full weight
        ("it", 0.49),   # just under: a quarter
        ("fr", 0.80), ("fr", 0.80), ("fr", 0.80),
    ])
    guess = lang.detect(wav)

    assert guess.votes["es"] == pytest.approx(0.50)
    assert guess.votes["it"] == pytest.approx(0.49 * 0.25)
    assert guess.votes["fr"] == pytest.approx(2.40)


def test_the_discount_has_a_cliff_at_exactly_one_half(monkeypatch, wav):
    """Documented, not endorsed.

    0.50 counts four times as much as 0.49. Four windows that shift by a
    hundredth of a point swing the total from 0.49 to 2.00, which is enough to
    flip the winner. Any future smoothing of the weight should break this test.
    """
    install_backends(monkeypatch, results=[("it", 0.49)] * 4 + [("es", 1.0)])
    just_under = lang.detect(wav)
    install_backends(monkeypatch, results=[("it", 0.50)] * 4 + [("es", 1.0)])
    just_over = lang.detect(wav)

    assert just_under.language == "es"
    assert just_over.language == "it"
    assert just_over.votes["it"] / just_under.votes["it"] == pytest.approx(4 * 0.5 / 0.49)


def test_unanimous_file_says_so(monkeypatch, wav):
    install_backends(monkeypatch, results=[("it", 0.98)] * 5)
    guess = lang.detect(wav)

    assert guess.language == "it"
    assert guess.confidence == pytest.approx(1.0)
    assert guess.reliable is True
    assert guess.note == "it across every window"
    assert list(guess.votes) == ["it"]


def test_note_names_the_runner_up_when_the_winner_is_clear(monkeypatch, wav):
    install_backends(monkeypatch, results=[
        ("it", 0.97), ("it", 0.96), ("it", 0.95), ("es", 0.92), ("it", 0.94),
    ])
    guess = lang.detect(wav)

    assert guess.reliable is True
    assert "prevails" in guess.note
    assert "es" in guess.note          # the reader is told the file was not unanimous


def test_samples_carry_timestamp_language_and_probability(monkeypatch, wav):
    install_backends(monkeypatch, results=[
        ("es", 0.94), ("es", 0.91), ("it", 0.60), ("es", 0.88), ("es", 0.95),
    ])
    guess = lang.detect(wav)

    assert len(guess.samples) == 5
    for offset, language, prob in guess.samples:
        assert isinstance(offset, float)
        assert isinstance(language, str)
        assert isinstance(prob, float)
        assert 0.0 <= offset < DURATION
    assert [s[1] for s in guess.samples] == ["es", "es", "it", "es", "es"]
    assert guess.samples[0][0] == pytest.approx(0.05 * DURATION, abs=0.1)
    assert guess.samples[-1][0] == pytest.approx(0.85 * DURATION, abs=0.1)


def test_probabilities_are_plain_floats_not_numpy(monkeypatch, wav):
    """The guess is serialised into state.json; numpy scalars do not survive that."""
    install_backends(monkeypatch, results=[("es", np.float32(0.93))] * 5)
    guess = lang.detect(wav)

    assert type(guess.samples[0][2]) is float
    assert type(guess.votes["es"]) is float
    json.dumps({"votes": guess.votes, "samples": guess.samples})


# ---------------------------------------------------------------- degenerate

def test_audio_too_short_to_sample_returns_an_empty_guess(monkeypatch, wav):
    """One second of audio: every window is useless, and the vote says nothing."""
    back = install_backends(monkeypatch, results=[], duration=1.0)
    guess = lang.detect(wav)

    assert back.n_windows == 0          # the model was never asked
    assert guess.language == ""
    assert guess.confidence == 0.0
    assert guess.votes == {}
    assert guess.samples == []
    assert guess.reliable is False
    assert guess.note == "audio too short to sample"


def test_short_file_votes_on_the_windows_that_survive(monkeypatch, wav):
    """Two seconds: the late windows are under a second and get dropped, the
    early ones still vote. A truncated sample set must not crash the vote."""
    back = install_backends(monkeypatch, results=[
        ("es", 0.9), ("es", 0.9), ("es", 0.9),
    ], duration=2.0)
    guess = lang.detect(wav)

    assert 0 < back.n_windows < 5
    assert guess.language == "es"
    assert len(guess.samples) == back.n_windows
    assert guess.reliable is True


def test_stereo_is_mixed_down_before_detection(monkeypatch, wav):
    back = install_backends(monkeypatch, results=[("es", 0.9)] * 5, channels=2)
    guess = lang.detect(wav)

    assert guess.language == "es"
    for window in back.windows:
        assert window.ndim == 1


def test_wrong_sample_rate_is_refused_with_a_useful_message(monkeypatch, wav):
    install_backends(monkeypatch, results=[], sr=44_100)
    with pytest.raises(ValueError) as excinfo:
        lang.detect(wav)

    message = str(excinfo.value)
    assert "16000" in message and "44100" in message
    assert "prepare" in message          # says what to do about it


def test_n_samples_controls_how_many_windows_are_taken(monkeypatch, wav):
    back = install_backends(monkeypatch, results=[("es", 0.9)] * 3)
    lang.detect(wav, n_samples=3)
    assert back.n_windows == 3


def test_n_samples_zero_still_takes_one_window(monkeypatch, wav):
    back = install_backends(monkeypatch, results=[("es", 0.9)])
    guess = lang.detect(wav, n_samples=0)

    assert back.n_windows == 1
    assert guess.language == "es"


def test_model_is_loaded_small_and_on_cpu(monkeypatch, wav):
    back = install_backends(monkeypatch, results=[("es", 0.9)] * 5)
    lang.detect(wav)

    assert back.model_args[0][0] == "small"      # not large-v3: this is the cheap job
    assert back.model_kwargs[0]["device"] == "cpu"
    assert back.model_kwargs[0]["compute_type"] == "int8"
    assert back.model_kwargs[0]["cpu_threads"] == 8


def test_model_settings_are_forwarded(monkeypatch, wav):
    back = install_backends(monkeypatch, results=[("es", 0.9)] * 5)
    lang.detect(wav, model_size="medium", compute_type="float32", threads=2)

    assert back.model_args[0][0] == "medium"
    assert back.model_kwargs[0]["compute_type"] == "float32"
    assert back.model_kwargs[0]["cpu_threads"] == 2


def test_windows_are_thirty_seconds_of_audio(monkeypatch, wav):
    back = install_backends(monkeypatch, results=[("es", 0.9)] * 5)
    lang.detect(wav)

    # every window is at most 30 s, and never shorter than the 1 s floor
    for window in back.windows:
        assert SR <= len(window) <= lang.WINDOW_SEC * SR


# ------------------------------------------------------------------- BUG #1

@pytest.mark.xfail(
    raises=RuntimeError, strict=True,
    reason="BUG: lang.py:75 has no per-window error handling. One window that "
           "raises aborts the whole detection, throwing away the windows that "
           "already voted, in the one module whose job is to survive a bad "
           "window.",
)
def test_a_window_that_fails_does_not_crash_the_vote(monkeypatch, wav):
    install_backends(monkeypatch, results=[
        ("es", 0.94),
        ("es", 0.92),
        RuntimeError("failed to decode window"),
        ("es", 0.90),
        ("es", 0.93),
    ])
    guess = lang.detect(wav)

    assert guess.language == "es"
    assert len(guess.samples) == 4       # the broken window is simply missing


# ------------------------------------------------------------------- BUG #2

@pytest.mark.xfail(
    strict=True,
    reason="BUG: lang.py:88-92 computes confidence as a share of the vote, not "
           "as certainty. Five windows that were all guessing (p=0.30, which "
           "the code itself calls guessing at line 78) agree with each other, "
           "so confidence is 1.0 and reliable is True. The pipeline then prints "
           "'100%' and no warning for a file nobody actually identified.",
)
def test_a_file_where_every_window_was_guessing_is_not_reliable(monkeypatch, wav):
    install_backends(monkeypatch, results=[("it", 0.30)] * 5)
    guess = lang.detect(wav)

    assert guess.reliable is False
    assert guess.confidence < 0.6


def test_unanimous_guessing_currently_reports_full_confidence(monkeypatch, wav):
    """Pins the behaviour BUG #2 describes, so the damage is visible and any
    fix has to come through the xfail above rather than by accident."""
    install_backends(monkeypatch, results=[("it", 0.30)] * 5)
    guess = lang.detect(wav)

    assert guess.confidence == pytest.approx(1.0)
    assert guess.reliable is True
    assert max(p for _t, _l, p in guess.samples) == pytest.approx(0.30)


# ==========================================================================
# the explicit-language short circuit
# ==========================================================================
#
# `lang.detect()` takes no language argument on purpose: it is the detector, and
# a detector that can be told the answer is not one. The short circuit lives one
# level up, in Job.detect_language(), and it is tested here because it is the
# other half of the same promise: ask for Spanish and nothing gets sampled.

def test_detect_has_no_language_override():
    import inspect
    assert "language" not in inspect.signature(lang.detect).parameters


def _bare_job(tmp_path, language):
    """A Job without __init__: no ~/.scriba, no job folder, no source file."""
    from scriba.config import Settings
    from scriba.pipeline import Job

    job = Job.__new__(Job)
    job.s = Settings(language=language)
    job.state = {}
    job.dir = tmp_path
    job.state_path = tmp_path / "state.json"
    job.wav = tmp_path / "audio16k.wav"
    job.report = lambda _m: None
    return job


def test_an_explicitly_requested_language_short_circuits_detection(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("detection ran even though the language was given")

    monkeypatch.setattr("scriba.lang.detect", boom)
    job = _bare_job(tmp_path, "es")

    assert job.detect_language() == "es"
    assert job.state["language"] == "es"
    assert job.state["language_note"] == "set by hand"
    # and the choice is recorded, so the document can say where the date came from
    assert json.loads(job.state_path.read_text())["language"] == "es"


def test_auto_still_runs_the_vote(tmp_path, monkeypatch):
    calls = []

    def fake_detect(path, **kw):
        calls.append(path)
        return lang.LanguageGuess("es", 0.82, {"es": 3.7, "it": 0.8}, [], True, "es prevails")

    monkeypatch.setattr("scriba.lang.detect", fake_detect)
    job = _bare_job(tmp_path, "auto")

    assert job.detect_language() == "es"
    assert calls == [job.wav]
    assert job.state["language_confidence"] == pytest.approx(0.82)
    assert job.state["language_note"] == "es prevails"


# ==========================================================================
# audio._recorder(): who recorded this, if anyone can tell
# ==========================================================================

@pytest.mark.parametrize("encoder", [
    "Lavf62.3.100",
    "Lavf58.29.100",
    "LAVF62.3.100",
    "lavf62.3.100",
    "Lavc60.31.102",
    "libavformat 62.3.100",
])
def test_generic_encoder_tags_are_not_a_recording_device(encoder):
    """A re-encoded file must not claim ffmpeg recorded it."""
    assert audio._recorder(encoder) == ""


@pytest.mark.parametrize("encoder", [
    "com.apple.VoiceMemos (iPhone Version 18.2)",
    "com.apple.VoiceMemos (Apple Watch Version 11.1)",
    "Zoom H1n",
])
def test_a_real_recorder_is_kept(encoder):
    assert audio._recorder(encoder) == encoder


def test_no_encoder_tag_is_no_device():
    assert audio._recorder("") == ""


def test_a_padded_generic_tag_slips_through(monkeypatch):
    """Documented gap, low severity: the filter matches on a prefix and the tag
    is never stripped, so a leading space is enough to get 'Lavf' into the
    document as a recording device."""
    assert audio._recorder(" Lavf62.3.100") == " Lavf62.3.100"


# ==========================================================================
# audio.probe()
# ==========================================================================

def _ffprobe_json(**overrides):
    fmt = {
        "duration": "742.5",
        "tags": {
            "major_brand": "M4A ",
            "creation_time": "2026-03-11T09:14:07.000000Z",
            "encoder": "com.apple.VoiceMemos (iPhone Version 18.2)",
            "voice_memo_uuid": "5B1D6C10-INVENTED-UUID",
        },
    }
    fmt.update(overrides.pop("format", {}))
    fmt.setdefault("tags", {}).update(overrides.pop("tags", {}))
    streams = overrides.pop("streams", [
        {"codec_type": "audio", "codec_name": "aac", "sample_rate": "44100", "channels": 2},
    ])
    return json.dumps({"format": fmt, "streams": streams})


def _fake_ffprobe(monkeypatch, *, stdout="", stderr="", returncode=0):
    calls = []
    monkeypatch.setattr(audio, "_ffprobe", lambda: "/usr/local/bin/ffprobe")

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    return calls


def test_probe_reads_the_container_metadata(monkeypatch, tmp_path):
    src = tmp_path / "conversation.m4a"
    src.write_bytes(b"\0")
    _fake_ffprobe(monkeypatch, stdout=_ffprobe_json())

    info = audio.probe(src)

    assert info.path == src
    assert info.duration == pytest.approx(742.5)
    assert info.sample_rate == 44100
    assert info.channels == 2
    assert info.codec == "aac"
    assert info.created == "2026-03-11T09:14:07.000000Z"
    assert info.memo_uuid == "5B1D6C10-INVENTED-UUID"
    assert info.device == "com.apple.VoiceMemos (iPhone Version 18.2)"


def test_probe_drops_a_generic_encoder(monkeypatch, tmp_path):
    src = tmp_path / "reencoded.wav"
    src.write_bytes(b"\0")
    _fake_ffprobe(monkeypatch, stdout=_ffprobe_json(tags={"encoder": "Lavf62.3.100"}))

    assert audio.probe(src).device == ""


def test_probe_matches_tags_case_insensitively(monkeypatch, tmp_path):
    src = tmp_path / "odd-case.mov"
    src.write_bytes(b"\0")
    payload = json.dumps({
        "format": {"duration": "12.0",
                   "tags": {"Creation_Time": "2026-03-11T09:14:07Z", "ENCODER": "Zoom H1n"}},
        "streams": [{"codec_type": "audio", "codec_name": "pcm_s16le",
                     "sample_rate": "48000", "channels": 1}],
    })
    _fake_ffprobe(monkeypatch, stdout=payload)

    info = audio.probe(src)
    assert info.created == "2026-03-11T09:14:07Z"
    assert info.device == "Zoom H1n"


def test_probe_skips_the_video_stream(monkeypatch, tmp_path):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\0")
    payload = _ffprobe_json(streams=[
        {"codec_type": "video", "codec_name": "h264", "width": 1920},
        {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 1},
    ])
    _fake_ffprobe(monkeypatch, stdout=payload)

    info = audio.probe(src)
    assert info.codec == "aac"
    assert info.sample_rate == 48000


def test_probe_without_tags_or_audio_stream_still_returns(monkeypatch, tmp_path):
    src = tmp_path / "silent.mkv"
    src.write_bytes(b"\0")
    payload = json.dumps({"format": {"duration": "3.0"}, "streams": []})
    _fake_ffprobe(monkeypatch, stdout=payload)

    info = audio.probe(src)
    assert info.duration == pytest.approx(3.0)
    assert (info.sample_rate, info.channels, info.codec) == (0, 0, "?")
    assert info.created == "" and info.memo_uuid == "" and info.device == ""


def test_probe_explains_what_ffprobe_refused_to_read(monkeypatch, tmp_path):
    src = tmp_path / "truncated.m4a"
    src.write_bytes(b"")
    _fake_ffprobe(
        monkeypatch, returncode=1,
        stderr="[mov,mp4] moov atom not found\n"
               "truncated.m4a: Invalid data found when processing input\n",
    )

    with pytest.raises(RuntimeError) as excinfo:
        audio.probe(src)

    message = str(excinfo.value)
    assert "truncated.m4a" in message
    assert "Invalid data found when processing input" in message


def test_probe_failure_without_stderr_still_says_something(monkeypatch, tmp_path):
    src = tmp_path / "mystery.wav"
    src.write_bytes(b"")
    _fake_ffprobe(monkeypatch, returncode=1, stderr="   \n")

    with pytest.raises(RuntimeError, match="ffprobe gave no reason"):
        audio.probe(src)


def test_probe_without_ffprobe_installed_degrades(monkeypatch, tmp_path):
    src = tmp_path / "anything.m4a"
    src.write_bytes(b"\0")
    monkeypatch.setattr(audio, "_ffprobe", lambda: None)

    info = audio.probe(src)
    assert (info.duration, info.sample_rate, info.channels, info.codec) == (0.0, 0, 0, "?")
    assert info.created == ""


# ------------------------------------------------------------------- BUG #3

@pytest.mark.xfail(
    raises=ValueError, strict=True,
    reason="BUG: audio.py:80-82 casts the ffprobe fields without guarding the "
           "'N/A' that ffprobe writes when a container carries no duration or "
           "no sample rate. probe() then dies with \"could not convert string "
           "to float: 'N/A'\", which is exactly the bare cast error the "
           "returncode branch above was written to replace.",
)
def test_probe_survives_a_container_with_no_duration(monkeypatch, tmp_path):
    src = tmp_path / "stream.aac"
    src.write_bytes(b"\0")
    payload = json.dumps({
        "format": {"duration": "N/A", "tags": {}},
        "streams": [{"codec_type": "audio", "codec_name": "aac",
                     "sample_rate": "N/A", "channels": 1}],
    })
    _fake_ffprobe(monkeypatch, stdout=payload)

    info = audio.probe(src)
    assert info.duration == 0.0
    assert info.sample_rate == 0


# ==========================================================================
# audio.recorded_at(): the date, and where the date came from
# ==========================================================================

def _info(path, created=""):
    return audio.AudioInfo(path=path, duration=1.0, sample_rate=SR, channels=1,
                           codec="pcm_s16le", created=created)


def test_container_creation_time_wins_and_is_labelled_recorded(tmp_path):
    src = tmp_path / "memo.m4a"
    src.write_bytes(b"\0")           # birthtime and mtime are now, i.e. today

    when, source = audio.recorded_at(_info(src, "2026-03-11T09:14:07.000000Z"))

    assert source == "recorded"
    # the same instant, in local time, worked out independently of the module
    expected = datetime.fromtimestamp(
        datetime(2026, 3, 11, 9, 14, 7, tzinfo=timezone.utc).timestamp())
    assert when == expected
    assert when.tzinfo is None        # naive local time, ready to format
    assert when.year == 2026          # and emphatically not today
    assert (datetime.now() - when).days > 1


def test_a_naive_creation_time_is_taken_as_written(tmp_path):
    src = tmp_path / "memo.m4a"
    src.write_bytes(b"\0")

    when, source = audio.recorded_at(_info(src, "2026-03-11T09:14:07"))

    assert source == "recorded"
    assert when == datetime(2026, 3, 11, 9, 14, 7)


def test_an_unparseable_creation_time_falls_through_instead_of_raising(tmp_path):
    src = tmp_path / "memo.m4a"
    src.write_bytes(b"\0")

    when, source = audio.recorded_at(_info(src, "sometime last spring"))

    assert source in ("file created", "file modified")
    assert when is not None


def test_without_container_metadata_the_filesystem_is_used_and_said_so(tmp_path):
    """An exported memo carries the timestamp of the export, and the document
    has to be able to say that this date is a filesystem guess, not a recording."""
    src = tmp_path / "exported.m4a"
    src.write_bytes(b"\0")

    when, source = audio.recorded_at(_info(src, ""))

    assert source != "recorded"
    assert source in ("file created", "file modified")
    assert when is not None
    assert abs((datetime.now() - when).total_seconds()) < 120


def test_birth_time_is_preferred_over_modification_time(tmp_path, monkeypatch):
    src = tmp_path / "copied.m4a"
    src.write_bytes(b"\0")
    birth = datetime(2026, 3, 11, 9, 14, 7).timestamp()
    mtime = datetime(2026, 7, 2, 18, 0, 0).timestamp()

    class _Stat:
        st_birthtime = birth
        st_mtime = mtime

    _patch_stat(monkeypatch, src, _Stat())

    when, source = audio.recorded_at(_info(src, ""))
    assert source == "file created"
    assert when == datetime(2026, 3, 11, 9, 14, 7)


def test_modification_time_is_the_last_resort(tmp_path, monkeypatch):
    """On a filesystem with no birth time (the Linux case) the label changes
    with the source, so the reader is told this is the weakest of the three."""
    src = tmp_path / "linux.m4a"
    src.write_bytes(b"\0")
    mtime = datetime(2026, 7, 2, 18, 0, 0).timestamp()

    class _Stat:                      # no st_birthtime attribute at all
        st_mtime = mtime

    _patch_stat(monkeypatch, src, _Stat())

    when, source = audio.recorded_at(_info(src, ""))
    assert source == "file modified"
    assert when == datetime(2026, 7, 2, 18, 0, 0)


def test_a_zero_birth_time_does_not_become_the_epoch(tmp_path, monkeypatch):
    src = tmp_path / "zero.m4a"
    src.write_bytes(b"\0")
    mtime = datetime(2026, 7, 2, 18, 0, 0).timestamp()

    class _Stat:
        st_birthtime = 0.0
        st_mtime = mtime

    _patch_stat(monkeypatch, src, _Stat())

    when, source = audio.recorded_at(_info(src, ""))
    assert source == "file modified"
    assert when.year == 2026


def test_the_full_priority_order(tmp_path, monkeypatch):
    src = tmp_path / "all-three.m4a"
    src.write_bytes(b"\0")

    class _Stat:
        st_birthtime = datetime(2026, 5, 1, 12, 0, 0).timestamp()
        st_mtime = datetime(2026, 7, 2, 18, 0, 0).timestamp()

    _patch_stat(monkeypatch, src, _Stat())

    recorded, source = audio.recorded_at(_info(src, "2026-03-11T09:14:07"))
    assert (recorded.month, source) == (3, "recorded")

    created, source = audio.recorded_at(_info(src, ""))
    assert (created.month, source) == (5, "file created")


def test_a_missing_file_yields_no_date_and_no_claim(tmp_path):
    missing = tmp_path / "gone.m4a"

    when, source = audio.recorded_at(_info(missing, ""))

    assert when is None
    assert source == ""      # nothing is asserted about a file that is not there


def _patch_stat(monkeypatch, target: Path, fake_stat):
    real_stat = Path.stat

    def stat(self, *args, **kwargs):
        if str(self) == str(target):
            return fake_stat
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)


# ==========================================================================
# audio.prepare(): convert through a scratch file, rename once
# ==========================================================================

@dataclass
class FFmpegSpy:
    cmds: list = field(default_factory=list)
    seen_dest: list = field(default_factory=list)   # did dest exist mid-conversion?


def _fake_ffmpeg(monkeypatch, dest: Path, *, fail=False, stderr="", written=b"RIFFdata"):
    spy = FFmpegSpy()
    monkeypatch.setattr(audio, "_ffmpeg", lambda: "/opt/homebrew/bin/ffmpeg")

    def fake_run(cmd, **kwargs):
        spy.cmds.append(cmd)
        out = Path(cmd[-1])
        out.write_bytes(written)                    # ffmpeg writes to the scratch name
        spy.seen_dest.append(dest.exists())
        if fail:
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr=stderr)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    return spy


def test_prepare_writes_through_a_part_file_and_renames(monkeypatch, tmp_path):
    src = tmp_path / "memo.m4a"
    src.write_bytes(b"\0")
    dest = tmp_path / "job" / "audio16k.wav"
    spy = _fake_ffmpeg(monkeypatch, dest)

    out = audio.prepare(src, dest)

    assert out == dest
    assert dest.exists()
    # ffmpeg never wrote to the final name
    assert Path(spy.cmds[0][-1]) == dest.with_name("audio16k.wav.part")
    assert spy.seen_dest == [False]
    # and the scratch file is gone afterwards
    assert not dest.with_name("audio16k.wav.part").exists()
    assert list(dest.parent.iterdir()) == [dest]


def test_prepare_creates_the_destination_folder(monkeypatch, tmp_path):
    src = tmp_path / "memo.m4a"
    src.write_bytes(b"\0")
    dest = tmp_path / "deep" / "nested" / "audio16k.wav"
    _fake_ffmpeg(monkeypatch, dest)

    audio.prepare(src, dest)
    assert dest.exists()


def test_prepare_asks_for_sixteen_kilohertz_mono_pcm(monkeypatch, tmp_path):
    src = tmp_path / "memo.m4a"
    src.write_bytes(b"\0")
    dest = tmp_path / "audio16k.wav"
    spy = _fake_ffmpeg(monkeypatch, dest)

    audio.prepare(src, dest)
    cmd = spy.cmds[0]

    assert cmd[cmd.index("-ar") + 1] == "16000"
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[cmd.index("-c:a") + 1] == "pcm_s16le"
    assert "-vn" in cmd                                  # drop video from a .mov
    assert cmd[cmd.index("-i") + 1] == str(src)
    assert "-nostdin" in cmd
    # the scratch name ends in .part, which tells ffmpeg nothing: say the format
    assert cmd[cmd.index("-f") + 1] == "wav"


def test_prepare_filters_by_default(monkeypatch, tmp_path):
    src = tmp_path / "memo.m4a"
    src.write_bytes(b"\0")
    dest = tmp_path / "audio16k.wav"
    spy = _fake_ffmpeg(monkeypatch, dest)

    audio.prepare(src, dest)
    filters = spy.cmds[0][spy.cmds[0].index("-af") + 1]

    assert filters == "highpass=f=60,loudnorm=I=-18:TP=-2:LRA=11"


def test_prepare_filters_can_be_turned_off(monkeypatch, tmp_path):
    src = tmp_path / "memo.m4a"
    src.write_bytes(b"\0")
    dest = tmp_path / "audio16k.wav"
    spy = _fake_ffmpeg(monkeypatch, dest)

    audio.prepare(src, dest, normalize=False, highpass=0)
    assert "-af" not in spy.cmds[0]


def test_prepare_highpass_is_configurable(monkeypatch, tmp_path):
    src = tmp_path / "memo.m4a"
    src.write_bytes(b"\0")
    dest = tmp_path / "audio16k.wav"
    spy = _fake_ffmpeg(monkeypatch, dest)

    audio.prepare(src, dest, normalize=False, highpass=120)
    assert spy.cmds[0][spy.cmds[0].index("-af") + 1] == "highpass=f=120"


def test_an_interrupted_conversion_leaves_nothing_behind(monkeypatch, tmp_path):
    """The point of the scratch file: a half conversion must not become a
    shorter WAV that every later stage happily reuses."""
    src = tmp_path / "memo.m4a"
    src.write_bytes(b"\0")
    dest = tmp_path / "job" / "audio16k.wav"
    _fake_ffmpeg(monkeypatch, dest, fail=True,
                 stderr="memo.m4a: Invalid data found when processing input\n",
                 written=b"RIFF-nine-tenths")

    with pytest.raises(RuntimeError) as excinfo:
        audio.prepare(src, dest)

    assert "memo.m4a" in str(excinfo.value)
    assert "Invalid data found when processing input" in str(excinfo.value)
    assert not dest.exists()
    assert list(dest.parent.iterdir()) == []      # the .part was cleaned up too


def test_a_failed_conversion_does_not_destroy_an_earlier_good_one(monkeypatch, tmp_path):
    src = tmp_path / "memo.m4a"
    src.write_bytes(b"\0")
    dest = tmp_path / "audio16k.wav"
    dest.write_bytes(b"RIFF-the-whole-thing")
    _fake_ffmpeg(monkeypatch, dest, fail=True, stderr="boom\n")

    with pytest.raises(RuntimeError):
        audio.prepare(src, dest)

    assert dest.read_bytes() == b"RIFF-the-whole-thing"


def test_prepare_failure_without_stderr_reports_the_exit_code(monkeypatch, tmp_path):
    src = tmp_path / "memo.m4a"
    src.write_bytes(b"\0")
    dest = tmp_path / "audio16k.wav"
    _fake_ffmpeg(monkeypatch, dest, fail=True, stderr="")

    with pytest.raises(RuntimeError, match="exit code 1"):
        audio.prepare(src, dest)


def test_prepare_without_ffmpeg_says_how_to_install_it(monkeypatch, tmp_path):
    src = tmp_path / "memo.m4a"
    src.write_bytes(b"\0")
    monkeypatch.setattr(audio.shutil, "which", lambda _name: None)

    with pytest.raises(audio.FFmpegMissing, match="brew install ffmpeg"):
        audio.prepare(src, tmp_path / "audio16k.wav")


# ==========================================================================
# audio.slice_wav() and audio.is_audio()
# ==========================================================================

def test_slice_wav_extracts_a_snippet(monkeypatch, tmp_path):
    src = tmp_path / "audio16k.wav"
    src.write_bytes(b"RIFF")
    dest = tmp_path / "snippets" / "speaker-01.wav"
    spy = _fake_ffmpeg(monkeypatch, dest)

    out = audio.slice_wav(src, dest, 12.5, 18.25)
    cmd = spy.cmds[0]

    assert out == dest
    assert dest.parent.is_dir()
    assert cmd[cmd.index("-ss") + 1] == "12.500"
    assert cmd[cmd.index("-to") + 1] == "18.250"
    assert cmd[cmd.index("-ar") + 1] == "16000"
    assert cmd[-1] == str(dest)


@pytest.mark.parametrize("name", [
    "memo.m4a", "MEMO.M4A", "call.mp3", "interview.wav", "clip.mov",
    "screen.mp4", "voice.opus", "tape.aiff",
])
def test_is_audio_accepts_what_the_pipeline_can_open(name):
    assert audio.is_audio(Path("/somewhere") / name) is True


@pytest.mark.parametrize("name", [
    "notes.txt", "transcript.md", "archive.zip", "memo", "memo.m4a.part",
])
def test_is_audio_rejects_the_rest(name):
    assert audio.is_audio(Path("/somewhere") / name) is False


@pytest.mark.parametrize("name", [
    ".DS_Store",
    "._memo.m4a",        # the AppleDouble sidecar that rides along on exFAT/SMB
    ".hidden.wav",
])
def test_is_audio_ignores_dot_files(name):
    assert audio.is_audio(Path("/somewhere") / name) is False
