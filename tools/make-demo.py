#!/usr/bin/env python3
"""Build a demo recording out of nothing, for screenshots and manual testing.

Every conversation scriba is useful on is somebody's private conversation, which
makes it a bad thing to put in a README. This writes a fake one instead: the
dialogue is invented, the voices are macOS speech synthesis, and the two speakers
are called Dana and Ray because no one is.

    python tools/make-demo.py            # writes demo/*.m4a
    SCRIBA_HOME=/tmp/scriba-demo scriba run demo/board-meeting.m4a

Point SCRIBA_HOME somewhere temporary before running scriba on it: the demo
should not land in the same registry as real recordings.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Two synthesised voices, alternating. `say -v ?` lists what is installed.
#
# The pair matters more than it looks. Samantha and Alex are both American, both
# synthesised the same way, and to a speaker embedding model they are nearly the
# same person: diarization put the whole conversation on one label. Daniel and
# Moira sit further apart.
#
# The filter is the second half of the fix. Two people in a room are never on the
# same channel: they sit at different distances, their voices hit the microphone
# differently, and speaker embeddings pick that up. Speech synthesis has none of
# it, so each voice here gets a mild, fixed colouration to stand in for a seat at
# the table. It makes the demo closer to a real recording, not easier than one.
VOICES = {
    "Dana": ("Moira",  "highpass=f=90,lowpass=f=7500,volume=1.0"),
    "Ray":  ("Daniel", "highpass=f=70,lowpass=f=6200,volume=0.85,aecho=0.8:0.5:12:0.15"),
}

SCRIPT: list[tuple[str, str]] = [
    ("Dana", "Right, let's start. The migration was supposed to be done on Friday "
             "and it is Tuesday, so I would like to understand what happened."),
    ("Ray",  "The migration itself took forty minutes. What took four days was the "
             "backfill, because the old rows have no timezone on the timestamp."),
    ("Dana", "No timezone at all, or a timezone we cannot trust?"),
    ("Ray",  "A timezone we cannot trust. Everything written before March is in "
             "local time with no offset recorded. Anything after is proper UTC."),
    ("Dana", "So how did you decide what March means for a row?"),
    ("Ray",  "I did not decide. I split the table. Rows we can place exactly went "
             "straight over, the ambiguous ones went into a second table with a flag, "
             "and the report reads both and marks the second group as approximate."),
    ("Dana", "How many rows are in the approximate group?"),
    ("Ray",  "About one in nine. Ninety thousand out of eight hundred thousand."),
    ("Dana", "That is more than I expected. Does it change the quarterly numbers?"),
    ("Ray",  "It moves them by less than a day either way, so the monthly totals are "
             "fine and the daily ones are not. I would not publish a daily chart from "
             "that table until we have gone back to the source exports."),
    ("Dana", "Do we still have the source exports?"),
    ("Ray",  "We have them until the end of last year. Before that, nobody kept them."),
    ("Dana", "Then let us write that down somewhere visible, because in six months "
             "somebody will build a dashboard on this and not know."),
    ("Ray",  "I will put it in the table comment as well as the wiki. People read the "
             "schema more often than they read the wiki."),
    ("Dana", "Agreed. Anything else before we close?"),
    ("Ray",  "One thing. The backfill job has no retry, so if it dies halfway it "
             "leaves the flag column half written. I would rather fix that now than "
             "explain it later."),
    ("Dana", "Fix it now. Same for anything else that fails silently."),
]

# A second, shorter recording with the same two people. One conversation shows
# transcription; two show the part that is actually hard, which is a tool
# recognising a voice it has heard before without being told twice.
FOLLOW_UP: list[tuple[str, str]] = [
    ("Ray",  "Quick one. The retry is in, and I ran the backfill again from a clean "
             "table to make sure it lands in the same place twice."),
    ("Dana", "Does it?"),
    ("Ray",  "To the row. Same counts, same flags, and the second run took nineteen "
             "minutes instead of four days because nothing had to be inferred."),
    ("Dana", "Good. Put the nineteen minutes in the ticket, because the next person "
             "who reads it will assume this is a four day job and plan around that."),
    ("Ray",  "Will do. The table comment is written as well."),
    ("Dana", "Then we are done. Thanks for staying on it."),
]


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", *args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    if shutil.which("say") is None:
        print("this needs macOS `say`", file=sys.stderr)
        return 1

    demo_dir = Path(__file__).resolve().parent.parent / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    for name, script in (("board-meeting", SCRIPT), ("follow-up", FOLLOW_UP)):
        _render(script, demo_dir / f"{name}.m4a")
    return 0


def _render(script: list[tuple[str, str]], out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        parts = []
        for i, (who, line) in enumerate(script):
            voice, colour = VOICES[who]
            raw = tmp / f"{i:03d}.aiff"
            subprocess.run(["say", "-v", voice, "-o", str(raw), line], check=True)
            # Every clip through the same filter before concatenating. `say` picks
            # its own sample rate per voice, and the concat demuxer will happily
            # join two files that disagree and produce audio that plays but is
            # subtly wrong. That version of this demo came out as one long turn
            # spoken by one person, which is exactly the failure it should show.
            part = tmp / f"{i:03d}.wav"
            _ffmpeg(["-i", str(raw), "-af", colour, "-ar", "16000", "-ac", "1", str(part)])
            parts.append(part)

        # Real conversations have gaps, and diarization uses them. Clips butted
        # together back to back give it a cleaner signal than any room ever would.
        gap = tmp / "gap.wav"
        _ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "0.45", str(gap)])

        listing = tmp / "list.txt"
        with listing.open("w") as fh:
            for part in parts:
                fh.write(f"file '{part}'\nfile '{gap}'\n")

        _ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing),
                 "-c:a", "aac", "-b:a", "64k", str(out)])

    minutes = sum(len(line.split()) for _, line in script) / 150
    print(f"{out}  (~{minutes:.1f} min, {len(script)} turns, "
          f"{len(VOICES)} synthetic speakers)")


if __name__ == "__main__":
    raise SystemExit(main())
