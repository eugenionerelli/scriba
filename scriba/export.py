"""Output formats. The one that really matters is `notebooklm`.

NotebookLM reasons over the text of a source, not over a data structure: anything
that is not readable text is noise that burns context. So no JSON, no millisecond
timestamps, no one-block-per-sentence. The header carries the metadata (who is
there, when, how long it runs) because that is what the user asks about first.
Then come real conversation turns, with the name up front.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# helper
# --------------------------------------------------------------------------- #

def hhmmss(seconds: float, *, always_hours: bool = False) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h or always_hours:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def vtt_time(seconds: float) -> str:
    return srt_time(seconds).replace(",", ".")


def display(speaker: str | None, names: dict[str, str] | None) -> str:
    if speaker is None:
        return "Unattributed"
    if names and speaker in names:
        return names[speaker]
    # SPEAKER_00 -> "Voice 1". pyannote counts from zero and pads with a leading
    # zero. Nobody reading a document needs either, so we count from one.
    if speaker.startswith("SPEAKER_"):
        try:
            return f"Voice {int(speaker.removeprefix('SPEAKER_')) + 1}"
        except ValueError:
            pass
    return speaker


# --------------------------------------------------------------------------- #
# NotebookLM
# --------------------------------------------------------------------------- #

def notebooklm(
    turns: list[dict],
    *,
    title: str,
    names: dict[str, str] | None = None,
    recorded: datetime | None = None,
    duration: float = 0.0,
    language: str = "it",
    source_file: str = "",
    speaker_stats: dict[str, float] | None = None,
    unresolved: list[str] | None = None,
) -> str:
    names = names or {}
    lines: list[str] = []

    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    if recorded:
        lines.append(f"- **Date**: {recorded.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **Duration**: {hhmmss(duration, always_hours=True)}")
    lines.append(f"- **Language**: {language}")
    if source_file:
        lines.append(f"- **Source file**: {source_file}")

    partecipanti = []
    for spk in sorted({t["speaker"] for t in turns if t.get("speaker")}):
        nome = display(spk, names)
        if speaker_stats and spk in speaker_stats:
            share = speaker_stats[spk]
            partecipanti.append(f"{nome} ({hhmmss(share)} of speech)")
        else:
            partecipanti.append(nome)
    if partecipanti:
        lines.append(f"- **Participants**: {', '.join(partecipanti)}")
    if unresolved:
        # Stating the uncertainty inside the source is the whole point: leave it out
        # and NotebookLM treats "Voice 2" as if it were an identified person.
        chi = ", ".join(sorted(unresolved))
        plurale = len(unresolved) > 1
        lines.append(
            f"- **Unidentified voices**: {chi}. "
            f"{'Distinct people' if plurale else 'A distinct person'}. "
            f"{'Their names are' if plurale else 'Their name is'} never spoken in "
            "the recording. Do not guess who they are."
        )
    lines.append("")
    lines.append("## Transcript")
    lines.append("")

    last_speaker = None
    for t in turns:
        nome = display(t.get("speaker"), names)
        stamp = hhmmss(t["start"])
        if nome != last_speaker:
            lines.append("")
        lines.append(f"**{nome}** [{stamp}]: {t['text'].strip()}")
        last_speaker = nome

    lines.append("")
    return "\n".join(lines).replace("\n\n\n", "\n\n")


# --------------------------------------------------------------------------- #
# other formats
# --------------------------------------------------------------------------- #

def markdown(turns: list[dict], *, names: dict[str, str] | None = None) -> str:
    return "\n\n".join(
        f"**{display(t.get('speaker'), names)}** [{hhmmss(t['start'])}]  \n{t['text'].strip()}"
        for t in turns
    ) + "\n"


def plain(turns: list[dict], *, names: dict[str, str] | None = None) -> str:
    return "\n".join(
        f"{display(t.get('speaker'), names)}: {t['text'].strip()}" for t in turns
    ) + "\n"


def srt(segments: list[dict], *, names: dict[str, str] | None = None) -> str:
    out = []
    for i, seg in enumerate(segments, 1):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        who = display(seg.get("speaker"), names)
        out.append(f"{i}\n{srt_time(seg['start'])} --> {srt_time(seg['end'])}\n[{who}] {text}\n")
    return "\n".join(out)


def vtt(segments: list[dict], *, names: dict[str, str] | None = None) -> str:
    out = ["WEBVTT", ""]
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        who = display(seg.get("speaker"), names)
        out.append(f"{vtt_time(seg['start'])} --> {vtt_time(seg['end'])}")
        out.append(f"<v {who}>{text}")
        out.append("")
    return "\n".join(out)


def payload_json(
    *,
    meta: dict[str, Any],
    segments: list[dict],
    turns: list[dict],
    names: dict[str, str],
    matches: list[dict],
) -> str:
    return json.dumps(
        {"meta": meta, "names": names, "speaker_matches": matches,
         "turns": turns, "segments": segments},
        indent=2, ensure_ascii=False,
    )


def write_all(
    outdir: Path,
    stem: str,
    formats: Iterable[str],
    *,
    turns: list[dict],
    segments: list[dict],
    names: dict[str, str],
    meta: dict[str, Any],
    matches: list[dict],
) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # This file used to carry an em dash in its name. Renaming it was right, and it
    # left a trap: write_all only ever writes, so a job exported under the old name
    # and then re-run ends up with two NotebookLM sources in the same folder. Once
    # both are uploaded there is no way to tell the stale one from the fresh one.
    # Remove the old name instead of leaving that in a folder people drag files out of.
    stale = outdir / f"{stem} — for NotebookLM.md"
    if stale.exists():
        stale.unlink()

    for fmt in formats:
        if fmt == "notebooklm":
            path = outdir / f"{stem} (for NotebookLM).md"
            path.write_text(notebooklm(
                turns,
                title=meta.get("title", stem),
                names=names,
                recorded=meta.get("recorded"),
                duration=meta.get("duration", 0.0),
                language=meta.get("language", "it"),
                source_file=meta.get("source_file", ""),
                speaker_stats=meta.get("speaker_stats"),
                unresolved=meta.get("unresolved"),
            ))
        elif fmt == "md":
            path = outdir / f"{stem}.md"
            path.write_text(markdown(turns, names=names))
        elif fmt == "txt":
            path = outdir / f"{stem}.txt"
            path.write_text(plain(turns, names=names))
        elif fmt == "srt":
            path = outdir / f"{stem}.srt"
            path.write_text(srt(segments, names=names))
        elif fmt == "vtt":
            path = outdir / f"{stem}.vtt"
            path.write_text(vtt(segments, names=names))
        elif fmt == "json":
            path = outdir / f"{stem}.json"
            path.write_text(payload_json(meta={k: (v.isoformat() if isinstance(v, datetime) else v)
                                                for k, v in meta.items()},
                                         segments=segments, turns=turns,
                                         names=names, matches=matches))
        else:
            continue
        written.append(path)
    return written
