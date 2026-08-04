"""Tests for where the decoder runs and in what numeric type.

Two ways this can be wrong and neither of them announces itself. Falling back to
the CPU on a machine that has the Metal build makes a seven-minute recording cost
seven minutes instead of one, and the only symptom is that it feels slow. Carrying
the CPU's int8 over to the GPU picks the slowest configuration the build offers,
which looks exactly like asking for the fast one: measured on an M4, MPS int8
57.6 s against MPS float16 24.9 s and CPU int8 50.5 s.

The Metal backend is not in any released ctranslate2. It comes from a branch that
has to be built on purpose, so `mps_available` asks the library for the function
that branch adds rather than asking macOS whether there is a GPU. Every stock
install must come back CPU, and that is what most of these check.
"""

from __future__ import annotations

import sys
import types

import pytest

from scriba import asr
from scriba.config import Settings


class FakeCT2:
    """Stands in for the ctranslate2 module, with or without the Metal branch."""

    def __init__(self, *, metal: bool, devices: int = 1, raises: bool = False):
        self.devices = devices
        self.raises = raises
        if metal:
            self.get_mps_device_count = self._count

    def _count(self):
        if self.raises:
            raise RuntimeError("Metal driver said no")
        return self.devices


@pytest.fixture
def ct2(monkeypatch):
    def install(**kwargs):
        module = FakeCT2(**kwargs)
        monkeypatch.setitem(sys.modules, "ctranslate2", module)
        return module
    return install


def test_a_stock_ctranslate2_reports_no_metal(ct2):
    ct2(metal=False)
    ok, why = asr.mps_available()
    assert ok is False
    # The reason has to name the cause. "Metal unavailable" on a machine with a
    # GPU sends somebody to look at their hardware for a missing library.
    assert "wheel" in why


def test_the_branch_build_reports_metal(ct2):
    ct2(metal=True)
    assert asr.mps_available() == (True, "")


def test_a_machine_with_the_build_and_no_device_is_not_metal(ct2):
    ct2(metal=True, devices=0)
    ok, why = asr.mps_available()
    assert (ok, why) == (False, "no Metal device")


def test_a_driver_that_throws_is_not_fatal(ct2):
    ct2(metal=True, raises=True)
    ok, why = asr.mps_available()
    assert ok is False and "Metal driver said no" in why


def test_auto_uses_the_gpu_when_it_is_there(ct2):
    ct2(metal=True)
    assert asr._asr_device(Settings(asr_device="auto", compute_type="int8")) \
        == ("mps", "float16")


def test_auto_stays_on_the_cpu_with_a_stock_wheel(ct2):
    ct2(metal=False)
    assert asr._asr_device(Settings(asr_device="auto", compute_type="int8")) \
        == ("cpu", "int8")


def test_cpu_is_obeyed_even_where_metal_exists(ct2):
    ct2(metal=True)
    assert asr._asr_device(Settings(asr_device="cpu", compute_type="int8")) \
        == ("cpu", "int8")


def test_asking_for_metal_without_it_fails_loudly(ct2):
    """Rather than quietly running on the CPU for twenty minutes.

    Somebody who wrote mps in the settings did it for a reason. A silent fallback
    means the machine is slow and nothing says why.
    """
    ct2(metal=False)
    with pytest.raises(RuntimeError) as caught:
        asr._asr_device(Settings(asr_device="mps"))
    assert "wheel" in str(caught.value)


def test_int8_becomes_float16_on_the_gpu_and_float32_is_left_alone(ct2):
    ct2(metal=True)
    assert asr._asr_device(Settings(asr_device="mps", compute_type="int8"))[1] == "float16"
    assert asr._asr_device(Settings(asr_device="mps", compute_type="float32"))[1] == "float32"
    assert asr._asr_device(Settings(asr_device="mps", compute_type="bfloat16"))[1] == "bfloat16"


def test_settings_reject_a_device_that_does_not_exist():
    with pytest.raises(ValueError, match="asr_device"):
        Settings(asr_device="cuda").validate()
    for good in ("auto", "cpu", "mps"):
        Settings(asr_device=good).validate()


def test_the_cache_notices_the_device_changing(tmp_path, monkeypatch):
    """A transcript decoded on the GPU is not the one decoded in int8 on the CPU.

    Without the device in the fingerprint, switching would print "reusing the
    cache" and hand back the other one, which is the kind of wrong answer that
    survives for months.
    """
    monkeypatch.setenv("SCRIBA_HOME", str(tmp_path))
    from scriba import config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cfg, "JOBS_DIR", tmp_path / "jobs")

    from scriba.pipeline import Job
    import scriba.pipeline as pipeline
    monkeypatch.setattr(pipeline, "JOBS_DIR", tmp_path / "jobs")

    source = tmp_path / "memo.wav"
    source.write_bytes(b"RIFF____WAVEfmt ")

    job = Job(source, Settings(asr_device="cpu"))
    on_cpu = job._asr_fingerprint()
    job.s = Settings(asr_device="mps")
    assert job._asr_fingerprint() != on_cpu
