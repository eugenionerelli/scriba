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
    archived: bool = False
    # Where the recording sat, relative to the folder it was added from. Add a
    # tree of recordings and the subfolder is the only thing saying which
    # conversation belongs with which, so it travels into the job.
    collection: str = ""


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
            except (ValueError, OSError):
                # ValueError covers both a malformed document and invalid UTF-8,
                # which is what a file half-written by a killed run looks like.
                # Letting either escape lost the whole listing, so every other job
                # became invisible and prune could not run at all.
                state = {}
        if not isinstance(state, dict):
            # Valid JSON that is not an object: null, a list, a bare string. It
            # parses, so nothing above catches it, and the first .get() call ends
            # the listing with an AttributeError.
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
            archived=bool(state.get("archived")),
            collection=str(state.get("collection") or ""),
        ))

    return sorted(rows, key=lambda r: (r.recorded or "0000", r.source_name), reverse=True)


def _edit_state(job_dir: Path, **changes) -> bool:
    """Change a few fields of a job's state without disturbing the rest.

    Read, alter, write through the same atomic rename everything else uses. A
    job's state carries the transcript fingerprints and the names somebody typed
    in, so a truncated write here costs the work rather than a flag.
    """
    from .config import write_atomic

    path = job_dir / "state.json"
    state: dict = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            state = loaded if isinstance(loaded, dict) else {}
        except (ValueError, OSError):
            state = {}
    state.update(changes)
    write_atomic(path, json.dumps(state, indent=2, ensure_ascii=False, default=str))
    return True


def archive(job_dir: Path, *, value: bool = True) -> bool:
    """Take a recording out of the everyday list, keeping every byte of it.

    Not deletion and not a separate folder: a flag, so the work is still there
    and `scriba export` still finds it. The list is the thing being tidied, not
    the disk.
    """
    return _edit_state(job_dir, archived=bool(value))


def forget(job_dir: Path, *, with_source: bool = False) -> float:
    """Delete a job folder, and only the source recording if asked. Returns MB freed.

    The source is off by default and stays that way. Everything in the job folder
    can be made again from the recording; the recording cannot be made again from
    anything, and somebody clearing a list is not usually asking to lose the
    audio of a conversation that already happened.
    """
    if not job_dir.is_dir():
        return 0.0
    freed = _dir_size_mb(job_dir)

    source = ""
    state_path = job_dir / "state.json"
    if state_path.exists():
        try:
            loaded = json.loads(state_path.read_text())
            source = loaded.get("source", "") if isinstance(loaded, dict) else ""
        except (ValueError, OSError):
            source = ""

    shutil.rmtree(job_dir, ignore_errors=True)

    if with_source and source:
        path = Path(source)
        if path.is_file():
            freed += path.stat().st_size / 1_048_576
            path.unlink(missing_ok=True)
    return freed


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
