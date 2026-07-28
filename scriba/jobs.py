"""What scriba has processed, and what it is holding on to.

The job folder is not somewhere anybody browses. Names are slugs with a hash on the
end, state lives in JSON, and "did I ever transcribe that one" has no answer short of
opening files. It is also where the disk goes: every job keeps a full 16 kHz copy of
the audio, which is worth about 2 MB a minute and can be rebuilt from the source in
seconds.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .config import JOBS_DIR


@dataclass
class JobRow:
    path: Path
    source_name: str
    source_path: str
    recorded: str
    duration: float
    state: str
    speakers: int
    names: dict[str, str] = field(default_factory=dict)
    size_mb: float = 0.0
    audio_mb: float = 0.0
    has_output: bool = False


def _dir_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1_048_576


def inventory() -> list[JobRow]:
    """Every job on disk, newest recording first.

    Jobs whose state cannot be read are still listed. A folder that failed halfway is
    exactly what somebody is looking for when they run this, so hiding it would defeat
    the purpose.
    """
    if not JOBS_DIR.exists():
        return []

    rows: list[JobRow] = []
    for job_dir in JOBS_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        state: dict = {}
        state_path = job_dir / "state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
            except (json.JSONDecodeError, OSError):
                state = {}

        out_dir = job_dir / "output"
        has_output = out_dir.exists() and any(out_dir.iterdir())
        has_transcript = (job_dir / "transcript.json").exists()
        has_diar = (job_dir / "diarization.json").exists()

        if has_output:
            what = "done"
        elif has_transcript and has_diar:
            what = "transcribed"
        elif has_diar:
            what = "voices only"
        elif has_transcript:
            what = "text only"
        else:
            what = "nothing"

        wav = job_dir / "audio16k.wav"
        source = state.get("source", "")
        rows.append(JobRow(
            path=job_dir,
            source_name=Path(source).name if source else job_dir.name,
            source_path=source,
            recorded=(state.get("recorded") or "")[:10],
            duration=float(state.get("duration") or 0.0),
            state=what,
            speakers=len(state.get("matches") or {}),
            names=state.get("names") or {},
            size_mb=_dir_size_mb(job_dir),
            audio_mb=(wav.stat().st_size / 1_048_576) if wav.exists() else 0.0,
            has_output=has_output,
        ))

    return sorted(rows, key=lambda r: (r.recorded or "0000", r.source_name), reverse=True)


def prune(rows: list[JobRow], *, audio: bool, empty: bool,
          dry_run: bool = True) -> list[tuple[Path, float]]:
    """Remove what can be rebuilt, and nothing else.

    Deliberately narrow. Transcripts and diarization cost minutes of CPU each and are
    never touched here. The prepared audio is a conversion of a file that still exists,
    and a job that produced nothing is the residue of a run that failed.
    """
    targets: list[tuple[Path, float]] = []

    for row in rows:
        if empty and row.state == "nothing":
            targets.append((row.path, row.size_mb))
            continue
        if audio and row.audio_mb > 0:
            targets.append((row.path / "audio16k.wav", row.audio_mb))

    if not dry_run:
        for path, _ in targets:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    return targets
