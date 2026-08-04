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
    # Round to milliseconds first, then split. Rounding the fraction on its own
    # lets 0.9996 come out as "00:00:00,1000", a four-digit field that strict
    # subtitle parsers reject outright.
    total_ms = int(round(max(0.0, float(seconds)) * 1000))
    ms = total_ms % 1000
    h, rem = divmod(total_ms // 1000, 3600)
    m, s = divmod(rem, 60)
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
    untranscribed_seconds: float = 0.0,
    speech_seconds: float = 0.0,
    untranscribed_gaps: list[list[float]] | None = None,
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
            f"- **Attribution**: {len(shaky)} of {len(turns)} turns "
            f"{'is' if len(shaky) == 1 else 'are'} marked "
            "*(uncertain)*. The words are transcribed; which of the speakers said "
            "them is a guess, usually where two people talk over each other. Do not "
            "attribute a quote from those turns to a named person."
        )
    missing, speech = float(untranscribed_seconds or 0.0), float(speech_seconds or 0.0)
    if missing >= 5.0:
        # Say what is not here, and say where. A transcript reads as complete
        # whatever it leaves out, and somebody reading it a month later has no way
        # to tell. The timestamps are the point: they turn "something is missing"
        # into a place to go and listen.
        where = ", ".join(hhmmss(a) for a, _ in
                          sorted(untranscribed_gaps or [], key=lambda g: g[0] - g[1])[:5])
        lines.append(
            f"- **Not transcribed**: {hhmmss(missing)} of the {hhmmss(speech)} of speech "
            "in this recording produced no text: the diarizer heard somebody there and "
            "the transcriber wrote nothing. Do not read the absence of a topic as "
            + (f"evidence it was never raised. The longest of these start at {where}."
               if where else "evidence it was never raised.")
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
            # Same rule as the transcript. A quote lifted from this list without
            # the marker reads as though "Voice 2" were somebody identified.
            if t.get("speaker") and t["speaker"] not in names:
                who = f"{who} (unidentified)"
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
    # Counted over what is written, not over what was offered. Numbering the
    # source segments and then skipping the empty ones leaves holes in the
    # sequence, which strict parsers treat as a damaged file.
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        who = display(seg.get("speaker"), names)
        out.append(f"{len(out) + 1}\n{srt_time(seg['start'])} --> "
                   f"{srt_time(seg['end'])}\n[{who}] {text}\n")
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


def _sweep(outdir: Path, stem: str, formats: Iterable[str]) -> None:
    """Delete outputs for this recording that no longer correspond to a format.

    Only files this module could have written are considered, and only for this
    stem: the output folder belongs to one job, but people do drop things in it.
    """
    wanted = {FILENAMES[f].format(stem=stem) for f in formats if f in FILENAMES}
    # Every name this module could have written for this stem, minus the ones it
    # is about to write. Matching on the suffix instead was close enough to work
    # and wrong: with a stem like "Nota vocale 2026-03-04" it also matched the
    # user's own "Nota vocale 2026-03-04 appunti.md" and deleted it. Globbing was
    # the same mistake from the other side, since a stem containing [ or ] reads
    # as a character class and quietly matches nothing at all.
    for pattern in FILENAMES.values():
        name = pattern.format(stem=stem)
        if name in wanted:
            continue
        stale = outdir / name
        if stale.is_file():
            stale.unlink()


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
    # Check the names before touching the folder. This used to sweep, write what
    # it understood and raise afterwards, which left a half-updated folder and
    # deleted the previous file for the format that was misspelled, after the
    # transcription had already been paid for.
    formats = list(formats)
    unknown = [f for f in formats if f not in FILENAMES]
    if unknown:
        raise ValueError(
            f"unknown output format {unknown[0]!r}. A typo here used to cost one "
            "file with no warning, after the transcription had already run.")

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
                untranscribed_seconds=meta.get("untranscribed_seconds", 0.0),
                speech_seconds=meta.get("speech_seconds", 0.0),
                untranscribed_gaps=meta.get("untranscribed_gaps"),
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
        written.append(path)
    return written
