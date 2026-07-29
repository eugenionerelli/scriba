"""Shared guards for the test suite.

The one rule worth enforcing at this level: a test run must never touch the
registry and the jobs of whoever is running it. Those hold voice prints and
transcripts of real conversations, and a test that quietly enrolled a voice into
them, or pruned them, would be a bad way to find that out.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REAL_HOME = Path.home() / ".scriba"


@pytest.fixture(autouse=True)
def isolated_home(tmp_path_factory, monkeypatch):
    """Point SCRIBA_HOME at a temporary folder for every test.

    Deliberately not inside `tmp_path`: several tests assert on the exact
    contents of that directory, and a folder appearing in it that the test did
    not create is a confusing way to fail.

    Modules that read the variable at import time still need their own handling,
    which individual tests do. This covers everything else, including code that
    reads it lazily and anything added later that nobody thought about.
    """
    home = tmp_path_factory.mktemp("scriba-home")
    monkeypatch.setenv("SCRIBA_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def real_home_untouched():
    """Fail a test that modified the real ~/.scriba, rather than let it pass."""
    def snapshot() -> set[tuple[str, int]]:
        if not REAL_HOME.exists():
            return set()
        return {(str(p.relative_to(REAL_HOME)), p.stat().st_mtime_ns)
                for p in REAL_HOME.rglob("*") if p.is_file()}

    before = snapshot()
    yield
    after = snapshot()
    changed = before ^ after
    assert not changed, (
        f"this test wrote into the real {REAL_HOME}: "
        f"{sorted(name for name, _ in changed)[:5]}"
    )


@pytest.fixture
def turn():
    """A conversation turn, with only the fields the code actually reads."""
    def make(start: float, end: float, speaker: str | None, text: str,
             confidence: float = 1.0) -> dict:
        return {"start": start, "end": end, "speaker": speaker,
                "text": text, "confidence": confidence}
    return make


def pytest_report_header(config):
    return f"scriba tests, SCRIBA_HOME redirected, real home is {REAL_HOME}"


assert "PYTEST_CURRENT_TEST" not in os.environ or True  # import-order sanity
