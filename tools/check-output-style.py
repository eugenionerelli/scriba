#!/usr/bin/env python3
"""Render every document scriba produces, then hold it to the writing rules.

Checking the source files is not the same as checking what comes out of them. The
templates in export.py and naming.py are built by string concatenation, so an em dash
can enter the finished document from a line that reads innocently on its own. This
renders each document from stand-in data and runs the checker over the result.

    python tools/check-output-style.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scriba import export, naming  # noqa: E402

# Stand-in data, deliberately not taken from any real recording. Two named speakers,
# one unnamed, one borderline registry match, so every branch of every template runs.
TURNS = [
    {"speaker": "SPEAKER_00", "start": 12.0, "end": 18.5,
     "text": "Right, so where did we land on the migration?"},
    {"speaker": "SPEAKER_01", "start": 19.0, "end": 26.0,
     "text": "Staging is done. Production waits for the backup window."},
    {"speaker": "SPEAKER_02", "start": 27.0, "end": 28.0, "text": "Mm."},
    {"speaker": "SPEAKER_00", "start": 29.0, "end": 35.0,
     "text": "Then we hold until Thursday and tell the team on the call."},
]
NAMES = {"SPEAKER_00": "Ada", "SPEAKER_01": "Rafiq"}
SPEECH = {"SPEAKER_00": 12.5, "SPEAKER_01": 7.0, "SPEAKER_02": 1.0}
MATCHES = {
    "SPEAKER_00": {"name": "Ada", "candidate": None, "score": 0.91, "reason": "accepted"},
    "SPEAKER_01": {"name": None, "candidate": "Rafiq", "score": 0.62,
                   "reason": "could be Rafiq (0.620). That is under the certainty "
                             "threshold (0.75), so confirm it yourself"},
    "SPEAKER_02": {"name": None, "candidate": None, "score": 0.0,
                   "reason": "only 1s of speech: voice print too thin to trust"},
}
META = {
    "title": "Team sync",
    "recorded": datetime(2026, 7, 18, 15, 30),
    "duration": 2832.0,
    "language": "en",
    "source_file": "team-sync.m4a",
    "speaker_stats": SPEECH,
    "unresolved": ["Voice 3"],
}


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="scriba-style-"))

    written = export.write_all(
        tmp, "team-sync", ["notebooklm", "md", "txt", "srt", "vtt"],
        turns=TURNS, segments=TURNS, names=NAMES, meta=META,
        matches=[{"speaker": k, **v} for k, v in MATCHES.items()],
    )

    profiles = naming.build_profiles(TURNS, SPEECH, "en", MATCHES)
    briefing = tmp / "who-is-who.md"
    briefing.write_text(naming.dossier(profiles, language="en", title="Team sync"))
    written.append(briefing)

    print(f"rendered {len(written)} documents into {tmp}")
    checker = Path(__file__).parent / "stylecheck.py"
    result = subprocess.run(
        [sys.executable, str(checker), *[str(p) for p in written]],
        text=True, capture_output=True,
    )
    print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
