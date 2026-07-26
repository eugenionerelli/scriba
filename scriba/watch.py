"""Watched folder: drop an audio file in, a transcript comes out.

Why a folder and not Voice Memos directly: the Voice Memos library lives in
`~/Library/Group Containers/group.com.apple.VoiceMemos.shared/` and is protected by
TCC. Not even your own user can read it without granting Full Disk Access to the
process that tries. A plain folder, fed by a Shortcut or by iCloud, gets the same
result, and nobody has to weaken the system's protections.

The polling is deliberately naive: a loop with `sleep`. FSEvents would notify sooner.
It also fires *while* a 200 MB file is still being copied, and at that point you would
transcribe half the audio. Here a file enters processing only once its size has stayed
identical for two rounds.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .audio import is_audio
from .config import Settings
from .pipeline import Job

DONE_MARK = ".scriba-done"


def watch(
    folder: Path,
    settings: Settings | None = None,
    *,
    interval: float = 5.0,
    report: Callable[[str], None] = print,
) -> None:
    folder = Path(folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    done_dir = folder / DONE_MARK
    done_dir.mkdir(exist_ok=True)

    seen: dict[Path, int] = {}
    processed: set[str] = {p.name for p in done_dir.glob("*")}

    report(f"watching {folder}  (ctrl-c to stop)")
    if processed:
        report(f"{len(processed)} files already processed earlier: skipping them")

    while True:
        try:
            for path in sorted(folder.iterdir()):
                if not path.is_file() or not is_audio(path) or path.name in processed:
                    continue

                size = path.stat().st_size
                if seen.get(path) != size:
                    # Still being copied (or still syncing from iCloud): retry next round.
                    seen[path] = size
                    continue

                report(f"── {path.name}")
                try:
                    job = Job(path, settings, report=lambda m: report(f"   {m}"))
                    res = job.run()
                    report(f"   {len(res.outputs)} files written")
                    if res.unresolved:
                        report(f"   unidentified voices: {', '.join(res.unresolved)} "
                               f"(read {res.dossier_path.name} and use `scriba name`)")
                except Exception as exc:
                    report(f"   error: {exc}")
                finally:
                    # The marker gets written either way: a file that makes the
                    # pipeline fail should not retry every five seconds forever.
                    (done_dir / path.name).touch()
                    processed.add(path.name)
                    seen.pop(path, None)

            time.sleep(interval)
        except KeyboardInterrupt:
            report("\nstopping.")
            return
