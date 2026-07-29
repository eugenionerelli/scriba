"""Output formats. The one that really matters is `source`.

A source is the document you hand to something else: a colleague, a notes app, a
model with a context window. Whatever reads it reads prose, not a data structure,
so anything that is not readable text is noise. No JSON, no millisecond timestamps,
no one block per sentence. The header carries the metadata (who is there, when, how
long it runs) because that is what anyone asks about first. Then come real
conversation turns, with the name up front.

The other formats exist for tools that want them: subtitles for a video editor,
JSON for a script, plain text for grep.
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
# source
# --------------------------------------------------------------------------- #

def source_doc(
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
    recorded_source: str = "",
    device: str = "",
    uncertain_below: float = 0.8,
) -> str:
    names = names or {}
    lines: list[str] = []

    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    if recorded:
        stamp = recorded.strftime("%Y-%m-%d %H:%M")
        # Say where the date came from. Only "recorded" is the time the recorder
        # wrote down; the others are the filesystem's guess, and an exported voice
        # memo carries the timestamp of the export rather than of the conversation.
        if recorded_source and recorded_source != "recorded":
            lines.append(f"- **Date**: {stamp} (from the file's {recorded_source} "
                         "time, which may not be when the conversation happened)")
        else:
            lines.append(f"- **Date**: {stamp}")
    lines.append(f"- **Duration**: {hhmmss(duration, always_hours=True)}")
    lines.append(f"- **Language**: {language}")
    if source_file:
        lines.append(f"- **Source file**: {source_file}")
    if device:
        lines.append(f"- **Recorded with**: {device}")

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
        # and whatever reads the file treats "Voice 2" as an identified person.
        chi = ", ".join(sorted(unresolved))
        plurale = len(unresolved) > 1
        lines.append(
            f"- **Unidentified voices**: {chi}. "
            f"{'Distinct people' if plurale else 'A distinct person'}. "
            f"{'Their names are' if plurale else 'Their name is'} never spoken in "
            "the recording. Do not guess who they are."
        )
    shaky = [t for t in turns if float(t.get("confidence", 1.0)) < uncertain_below]
    if shaky:
        lines.append(
            f"- **Attribution**: {len(shaky)} of {len(turns)} turns are marked "
            "*(uncertain)*. The words are transcribed; which of the speakers said "
            "them is a guess, usually where two people talk over each other. Do not "
            "attribute a quote from those turns to a named person."
        )
    lines.append(
        "- **Note**: this is automatic speech recognition. Figures, dates and proper "
        "names are the parts it gets wrong most often, and they are worth checking "
        "against the audio before relying on them."
    )
    lines.append("")
    lines.append("## Transcript")
    lines.append("")

    last_speaker = None
    for t in turns:
        nome = display(t.get("speaker"), names)
        stamp = hhmmss(t["start"])
        # Repeat the caveat on the line itself. The header says which voices went
        # unidentified, and by the middle of a long document that header is far away.
        # A model quoting a line reads the line, not the preamble.
        if t.get("speaker") and t["speaker"] not in names:
            nome = f"{nome} (unidentified)"
        if float(t.get("confidence", 1.0)) < uncertain_below:
            nome = f"{nome} (uncertain)"
        if nome != last_speaker:
            lines.append("")
        lines.append(f"**{nome}** [{stamp}]: {t['text'].strip()}")
        last_speaker = nome

    if shaky:
        lines.append("")
        lines.append("## Turns to check")
        lines.append("")
        lines.append("Speaker attribution is weakest here. The timestamps are for "
                     "going back to the audio.")
        lines.append("")
        for t in shaky:
            who = display(t.get("speaker"), names)
            snippet = t["text"].strip()
            if len(snippet) > 90:
                snippet = snippet[:90].rstrip() + "…"
            lines.append(f"- [{hhmmss(t['start'])}] {who}: {snippet}")

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


# Every file this module writes, in one place. _sweep needs to know what a stale
# output looks like, and a second list would drift from the first.
FILENAMES = {
    "source": "{stem} (source).md",
    "md": "{stem}.md",
    "txt": "{stem}.txt",
    "srt": "{stem}.srt",
    "vtt": "{stem}.vtt",
    "json": "{stem}.json",
}
SUFFIXES = {".md", ".txt", ".srt", ".vtt", ".json"}


def _sweep(outdir: Path, stem: str, formats: Iterable[str]) -> None:
    """Delete outputs for this recording that no longer correspond to a format.

    Only files this module could have written are considered, and only for this
    stem: the output folder belongs to one job, but people do drop things in it.
    """
    keep = {FILENAMES[f].format(stem=stem) for f in formats if f in FILENAMES}
    for path in outdir.glob(f"{stem}*"):
        if path.is_file() and path.name not in keep and path.suffix in SUFFIXES:
            path.unlink()


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

    # write_all only ever writes. If the name of a format ever changes again, the
    # old file stays in the folder next to the new one and there is no telling the
    # stale copy from the fresh one. Sweep anything matching this stem that we are
    # not about to write.
    _sweep(outdir, stem, formats)

    for fmt in formats:
        if fmt == "source":
            path = outdir / FILENAMES["source"].format(stem=stem)
            path.write_text(source_doc(
                turns,
                title=meta.get("title", stem),
                names=names,
                recorded=meta.get("recorded"),
                duration=meta.get("duration", 0.0),
                language=meta.get("language", "it"),
                source_file=meta.get("source_file", ""),
                speaker_stats=meta.get("speaker_stats"),
                unresolved=meta.get("unresolved"),
                recorded_source=meta.get("recorded_source", ""),
                device=meta.get("device", ""),
            ))
        elif fmt == "md":
            path = outdir / FILENAMES["md"].format(stem=stem)
            path.write_text(markdown(turns, names=names))
        elif fmt == "txt":
            path = outdir / FILENAMES["txt"].format(stem=stem)
            path.write_text(plain(turns, names=names))
        elif fmt == "srt":
            path = outdir / FILENAMES["srt"].format(stem=stem)
            path.write_text(srt(segments, names=names))
        elif fmt == "vtt":
            path = outdir / FILENAMES["vtt"].format(stem=stem)
            path.write_text(vtt(segments, names=names))
        elif fmt == "json":
            path = outdir / FILENAMES["json"].format(stem=stem)
            path.write_text(payload_json(meta={k: (v.isoformat() if isinstance(v, datetime) else v)
                                                for k, v in meta.items()},
                                         segments=segments, turns=turns,
                                         names=names, matches=matches))
        else:
            raise ValueError(
                f"unknown output format {fmt!r}. A typo here used to cost one file "
                "with no warning, after the transcription had already run.")
        written.append(path)
    return written
