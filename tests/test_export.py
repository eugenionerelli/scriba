"""Tests for scriba.export.

Everything here is built by hand: no audio, no models, no network. A "turn" is
just a dict with a speaker label, a start time, some text and (sometimes) a
confidence, which is exactly what the export layer is contracted to accept.

All names and content are invented.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from scriba import export
from scriba.export import (
    FILENAMES,
    _sweep,
    display,
    hhmmss,
    markdown,
    payload_json,
    plain,
    source_doc,
    srt,
    srt_time,
    vtt,
    vtt_time,
    write_all,
)


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #

def turn(speaker, start, text, confidence=None, end=None):
    t = {"speaker": speaker, "start": start, "end": end if end is not None else start + 3.0,
         "text": text}
    if confidence is not None:
        t["confidence"] = confidence
    return t


def transcript_lines(doc: str) -> list[str]:
    """The lines of the Transcript section that carry a spoken turn."""
    body = doc.split("## Transcript", 1)[1]
    body = body.split("## Turns to check", 1)[0]
    return [ln for ln in body.splitlines() if ln.startswith("**")]


def check_lines(doc: str) -> list[str]:
    """The bullet lines of the "Turns to check" section."""
    if "## Turns to check" not in doc:
        return []
    body = doc.split("## Turns to check", 1)[1]
    return [ln for ln in body.splitlines() if ln.startswith("- [")]


@pytest.fixture
def conversation():
    """Two voices, one of them never named, one shaky turn, one unattributed."""
    return [
        turn("SPEAKER_00", 0.0, "  Comincio io, se non vi dispiace.  ", 0.97),
        turn("SPEAKER_01", 5.0, "Fai pure.", 0.41),
        turn("SPEAKER_00", 12.0, "Il punto e' la scadenza di novembre."),
        turn(None, 20.0, "(rumore di fondo)", 0.99),
        turn("SPEAKER_01", 25.0, "Su quello non sono d'accordo.", 0.99),
    ]


@pytest.fixture
def segments():
    return [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.5, "text": "Comincio io."},
        {"speaker": "SPEAKER_01", "start": 2.5, "end": 6.0, "text": " Fai pure. "},
    ]


# --------------------------------------------------------------------------- #
# hhmmss
# --------------------------------------------------------------------------- #

def test_hhmmss_omits_hours_under_one_hour():
    assert hhmmss(0) == "00:00"
    assert hhmmss(9) == "00:09"
    assert hhmmss(65) == "01:05"
    assert hhmmss(3599) == "59:59"


def test_hhmmss_adds_hours_when_needed():
    assert hhmmss(3600) == "01:00:00"
    assert hhmmss(3661) == "01:01:01"
    assert hhmmss(37230) == "10:20:30"


def test_hhmmss_always_hours_pads_short_durations():
    assert hhmmss(0, always_hours=True) == "00:00:00"
    assert hhmmss(65, always_hours=True) == "00:01:05"
    assert hhmmss(3661, always_hours=True) == "01:01:01"


def test_hhmmss_truncates_the_fractional_second():
    # A timestamp you use to scrub back through audio should never round forward
    # past the moment the words start.
    assert hhmmss(59.999) == "00:59"
    assert hhmmss(3599.9) == "59:59"
    assert hhmmss(0.4) == "00:00"


def test_hhmmss_clamps_negative_input_to_zero():
    assert hhmmss(-1) == "00:00"
    assert hhmmss(-3600.5) == "00:00"
    assert hhmmss(-1, always_hours=True) == "00:00:00"


# --------------------------------------------------------------------------- #
# srt_time / vtt_time
# --------------------------------------------------------------------------- #

def test_srt_time_always_carries_hours_and_milliseconds():
    assert srt_time(0) == "00:00:00,000"
    assert srt_time(5) == "00:00:05,000"
    assert srt_time(3661.5) == "01:01:01,500"


def test_srt_time_renders_sub_second_precision():
    assert srt_time(1.25) == "00:00:01,250"
    assert srt_time(12.007) == "00:00:12,007"
    assert srt_time(90.75) == "00:01:30,750"


def test_srt_time_clamps_negative_input():
    assert srt_time(-4.2) == "00:00:00,000"


def test_vtt_time_is_the_srt_stamp_with_a_dot():
    assert vtt_time(0) == "00:00:00.000"
    assert vtt_time(3661.5) == "01:01:01.500"
    assert vtt_time(-1) == "00:00:00.000"


# Regression. Was: scriba/export.py:38: the millisecond part is rounded independently of  the whole-second part, so any time whose fraction rounds up to 1.000 emits a  four-digit field: srt_time(0.9996 == '00:00:00,1000'. That is not a valid  SRT/VTT timestamp and a subtitle track containing it is rejected by p
def test_srt_time_never_emits_a_four_digit_millisecond_field():
    assert srt_time(0.9996) == "00:00:01,000"
    assert srt_time(1.9999) == "00:00:02,000"


# --------------------------------------------------------------------------- #
# display
# --------------------------------------------------------------------------- #

def test_display_counts_voices_from_one():
    assert display("SPEAKER_00", None) == "Voice 1"
    assert display("SPEAKER_01", None) == "Voice 2"
    assert display("SPEAKER_09", None) == "Voice 10"
    assert display("SPEAKER_12", None) == "Voice 13"


def test_display_prefers_a_known_name():
    names = {"SPEAKER_00": "Ada Verdolini", "SPEAKER_01": "Bruno Cassaro"}
    assert display("SPEAKER_00", names) == "Ada Verdolini"
    assert display("SPEAKER_01", names) == "Bruno Cassaro"
    # a label with no entry still falls back to the numbered voice
    assert display("SPEAKER_02", names) == "Voice 3"


def test_display_of_no_speaker_is_unattributed():
    assert display(None, None) == "Unattributed"
    assert display(None, {"SPEAKER_00": "Ada Verdolini"}) == "Unattributed"


def test_display_empty_names_mapping_behaves_like_none():
    assert display("SPEAKER_00", {}) == "Voice 1"


def test_display_returns_a_malformed_label_verbatim():
    # Better an odd-looking label in the document than a crash or a wrong number.
    assert display("SPEAKER_XY", None) == "SPEAKER_XY"
    assert display("SPEAKER_", None) == "SPEAKER_"
    assert display("SPEAKER_0a", None) == "SPEAKER_0a"


def test_display_returns_an_unrecognised_label_verbatim():
    assert display("intervistatore", None) == "intervistatore"
    assert display("intervistatore", {"SPEAKER_00": "Ada Verdolini"}) == "intervistatore"


def test_display_name_wins_even_over_a_malformed_label():
    assert display("SPEAKER_XY", {"SPEAKER_XY": "Ada Verdolini"}) == "Ada Verdolini"


# --------------------------------------------------------------------------- #
# source_doc: header
# --------------------------------------------------------------------------- #

def test_source_doc_has_a_title_and_the_two_sections(conversation):
    doc = source_doc(conversation, title="Riunione di redazione")
    assert doc.startswith("# Riunione di redazione\n")
    assert "## Overview" in doc
    assert "## Transcript" in doc
    assert doc.index("## Overview") < doc.index("## Transcript")


def test_source_doc_date_is_plain_when_it_comes_from_the_recorder(conversation):
    doc = source_doc(conversation, title="T", recorded=datetime(2026, 3, 4, 9, 30),
                     recorded_source="recorded")
    assert "- **Date**: 2026-03-04 09:30" in doc
    assert "may not be when the conversation happened" not in doc


def test_source_doc_date_is_plain_when_no_source_is_declared(conversation):
    doc = source_doc(conversation, title="T", recorded=datetime(2026, 3, 4, 9, 30))
    assert "- **Date**: 2026-03-04 09:30" in doc
    assert "may not be" not in doc


def test_source_doc_date_flags_a_filesystem_timestamp(conversation):
    doc = source_doc(conversation, title="T", recorded=datetime(2026, 3, 4, 9, 30),
                     recorded_source="modification")
    assert "2026-03-04 09:30" in doc
    assert "from the file's modification time" in doc
    assert "may not be when the conversation happened" in doc


def test_source_doc_omits_the_date_when_there_is_none(conversation):
    doc = source_doc(conversation, title="T")
    assert "**Date**" not in doc


def test_source_doc_duration_always_shows_hours(conversation):
    assert "- **Duration**: 00:01:35" in source_doc(conversation, title="T", duration=95.0)
    assert "- **Duration**: 01:02:05" in source_doc(conversation, title="T", duration=3725.0)
    assert "- **Duration**: 00:00:00" in source_doc(conversation, title="T")


def test_source_doc_reports_language(conversation):
    assert "- **Language**: it" in source_doc(conversation, title="T")
    assert "- **Language**: es" in source_doc(conversation, title="T", language="es")


def test_source_doc_source_file_and_device_are_optional(conversation):
    bare = source_doc(conversation, title="T")
    assert "**Source file**" not in bare
    assert "**Recorded with**" not in bare

    full = source_doc(conversation, title="T", source_file="nota vocale 12.m4a",
                      device="Registratore da tasca")
    assert "- **Source file**: nota vocale 12.m4a" in full
    assert "- **Recorded with**: Registratore da tasca" in full


def test_source_doc_lists_participants_with_their_speech_share(conversation):
    doc = source_doc(
        conversation, title="T",
        names={"SPEAKER_00": "Ada Verdolini"},
        speaker_stats={"SPEAKER_00": 620.0, "SPEAKER_01": 95.0},
    )
    assert "- **Participants**: Ada Verdolini (10:20 of speech), Voice 2 (01:35 of speech)" in doc


def test_source_doc_participants_without_stats_are_just_names(conversation):
    doc = source_doc(conversation, title="T", names={"SPEAKER_00": "Ada Verdolini"})
    assert "- **Participants**: Ada Verdolini, Voice 2" in doc


def test_source_doc_participants_ignore_unattributed_turns():
    turns = [turn(None, 0.0, "solo rumore")]
    doc = source_doc(turns, title="T")
    assert "**Participants**" not in doc


def test_source_doc_unidentified_line_singular_wording(conversation):
    doc = source_doc(conversation, title="T", names={"SPEAKER_00": "Ada Verdolini"},
                     unresolved=["Voice 2"])
    line = next(ln for ln in doc.splitlines() if ln.startswith("- **Unidentified voices**"))
    assert "Voice 2." in line
    assert "A distinct person." in line
    assert "Their name is never spoken" in line
    assert "Distinct people" not in line
    assert "Their names are" not in line
    assert "Do not guess who they are." in line


def test_source_doc_unidentified_line_plural_wording(conversation):
    doc = source_doc(conversation, title="T", unresolved=["Voice 2", "Voice 1"])
    line = next(ln for ln in doc.splitlines() if ln.startswith("- **Unidentified voices**"))
    # sorted, so the list reads in a stable order regardless of the caller's
    assert line.startswith("- **Unidentified voices**: Voice 1, Voice 2.")
    assert "Distinct people." in line
    assert "Their names are never spoken" in line
    assert "A distinct person" not in line


def test_source_doc_has_no_unidentified_line_when_everyone_is_named(conversation):
    doc = source_doc(conversation, title="T",
                     names={"SPEAKER_00": "Ada Verdolini", "SPEAKER_01": "Bruno Cassaro"})
    assert "**Unidentified voices**" not in doc


def test_source_doc_attribution_line_counts_the_shaky_turns(conversation):
    doc = source_doc(conversation, title="T")
    assert "- **Attribution**: 1 of 5 turns is marked *(uncertain)*." in doc


def test_source_doc_has_no_attribution_line_when_nothing_is_shaky():
    turns = [turn("SPEAKER_00", 0.0, "Tutto chiaro.", 0.99)]
    doc = source_doc(turns, title="T")
    assert "**Attribution**" not in doc
    assert "## Turns to check" not in doc


def test_source_doc_always_warns_that_this_is_asr(conversation):
    assert "- **Note**: this is automatic speech recognition." in source_doc(conversation, title="T")


# --------------------------------------------------------------------------- #
# source_doc: transcript body
# --------------------------------------------------------------------------- #

def test_source_doc_turn_lines_carry_name_and_timestamp(conversation):
    doc = source_doc(conversation, title="T",
                     names={"SPEAKER_00": "Ada Verdolini", "SPEAKER_01": "Bruno Cassaro"})
    lines = transcript_lines(doc)
    assert lines[0] == "**Ada Verdolini** [00:00]: Comincio io, se non vi dispiace."
    assert "**Bruno Cassaro (uncertain)** [00:05]: Fai pure." in lines
    assert "**Unattributed** [00:20]: (rumore di fondo)" in lines


def test_source_doc_marks_unidentified_on_every_line_not_only_in_the_header(conversation):
    # The requirement: a reader (or a model) quoting one line out of the middle of
    # a long document must see that the speaker was never identified.
    doc = source_doc(conversation, title="T", names={"SPEAKER_00": "Ada Verdolini"},
                     unresolved=["Voice 2"])
    voice_two = [ln for ln in transcript_lines(doc) if ln.startswith("**Voice 2")]
    assert len(voice_two) == 2, voice_two
    for line in voice_two:
        assert "(unidentified)" in line, line


def test_source_doc_never_marks_a_named_speaker_as_unidentified(conversation):
    doc = source_doc(conversation, title="T",
                     names={"SPEAKER_00": "Ada Verdolini", "SPEAKER_01": "Bruno Cassaro"})
    assert "(unidentified)" not in doc


def test_source_doc_does_not_mark_an_unattributed_turn_as_unidentified():
    # No speaker at all is a different thing from a speaker nobody could name.
    turns = [turn(None, 0.0, "colpo di tosse", 0.99)]
    doc = source_doc(turns, title="T")
    assert transcript_lines(doc) == ["**Unattributed** [00:00]: colpo di tosse"]


def test_source_doc_marks_low_confidence_turns_uncertain(conversation):
    doc = source_doc(conversation, title="T",
                     names={"SPEAKER_00": "Ada Verdolini", "SPEAKER_01": "Bruno Cassaro"})
    uncertain = [ln for ln in transcript_lines(doc) if "(uncertain)" in ln]
    assert uncertain == ["**Bruno Cassaro (uncertain)** [00:05]: Fai pure."]


def test_source_doc_treats_a_missing_confidence_as_certain():
    turns = [turn("SPEAKER_00", 0.0, "Nessun punteggio qui.")]
    doc = source_doc(turns, title="T", names={"SPEAKER_00": "Ada Verdolini"})
    assert "(uncertain)" not in doc


def test_source_doc_uncertainty_threshold_is_configurable():
    turns = [turn("SPEAKER_00", 0.0, "Forse.", 0.9)]
    names = {"SPEAKER_00": "Ada Verdolini"}
    assert "(uncertain)" not in source_doc(turns, title="T", names=names, uncertain_below=0.8)
    assert "(uncertain)" in source_doc(turns, title="T", names=names, uncertain_below=0.95)


def test_source_doc_carries_both_markers_when_they_apply():
    turns = [turn("SPEAKER_03", 61.0, "Non saprei.", 0.2)]
    doc = source_doc(turns, title="T")
    assert transcript_lines(doc) == ["**Voice 4 (unidentified) (uncertain)** [01:01]: Non saprei."]


def test_source_doc_separates_speaker_changes_with_a_blank_line(conversation):
    doc = source_doc(conversation, title="T",
                     names={"SPEAKER_00": "Ada Verdolini", "SPEAKER_01": "Bruno Cassaro"})
    body = doc.split("## Transcript", 1)[1].split("## Turns to check", 1)[0]
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("**") and i > 0:
            assert lines[i - 1] == "", f"no blank line before {line!r}"


def test_source_doc_never_leaves_three_blank_lines_in_a_row(conversation):
    doc = source_doc(conversation, title="T", recorded=datetime(2026, 3, 4, 9, 30),
                     unresolved=["Voice 2"])
    assert "\n\n\n" not in doc


def test_source_doc_survives_an_empty_transcript():
    doc = source_doc([], title="Registrazione vuota")
    assert doc.startswith("# Registrazione vuota")
    assert "## Transcript" in doc
    assert "**Participants**" not in doc
    assert transcript_lines(doc) == []


# --------------------------------------------------------------------------- #
# source_doc: "Turns to check"
# --------------------------------------------------------------------------- #

def test_source_doc_turns_to_check_lists_only_the_shaky_turns(conversation):
    doc = source_doc(conversation, title="T",
                     names={"SPEAKER_00": "Ada Verdolini", "SPEAKER_01": "Bruno Cassaro"})
    assert "## Turns to check" in doc
    assert "Speaker attribution is weakest here." in doc
    assert check_lines(doc) == ["- [00:05] Bruno Cassaro: Fai pure."]


def test_source_doc_turns_to_check_comes_after_the_transcript(conversation):
    doc = source_doc(conversation, title="T")
    assert doc.index("## Transcript") < doc.index("## Turns to check")


def test_source_doc_turns_to_check_truncates_a_long_quote():
    long_text = ("Questa e' una battuta molto lunga che serve soltanto a superare "
                 "il limite di novanta caratteri previsto dal sommario finale.")
    doc = source_doc([turn("SPEAKER_00", 0.0, long_text, 0.3)], title="T",
                     names={"SPEAKER_00": "Ada Verdolini"})
    line = check_lines(doc)[0]
    snippet = line.split(": ", 1)[1]
    assert snippet.endswith("…")
    assert len(snippet) <= 91
    assert snippet.startswith("Questa e' una battuta molto lunga")
    # the full sentence is still in the transcript, only the summary is clipped
    assert long_text in doc


def test_source_doc_turns_to_check_keeps_a_short_quote_whole():
    doc = source_doc([turn("SPEAKER_00", 0.0, "Breve.", 0.3)], title="T",
                     names={"SPEAKER_00": "Ada Verdolini"})
    assert check_lines(doc) == ["- [00:00] Ada Verdolini: Breve."]


# Regression. Was: scriba/export.py:166: the 'Turns to check' section calls display(  without the '(unidentified' suffix that every transcript line carries, so a  quote lifted from that section reads as if 'Voice 2' were an identified  person. Same failure mode the per-line marker exists to prevent.
def test_source_doc_turns_to_check_also_marks_unidentified_speakers(conversation):
    doc = source_doc(conversation, title="T", names={"SPEAKER_00": "Ada Verdolini"},
                     unresolved=["Voice 2"])
    assert all("(unidentified)" in ln for ln in check_lines(doc)), check_lines(doc)


# --------------------------------------------------------------------------- #
# the tool-facing formats
# --------------------------------------------------------------------------- #

def test_markdown_puts_the_name_and_stamp_above_the_text(conversation):
    out = markdown(conversation, names={"SPEAKER_00": "Ada Verdolini"})
    assert out.startswith("**Ada Verdolini** [00:00]  \nComincio io, se non vi dispiace.")
    assert out.endswith("\n")
    assert "**Voice 2** [00:05]  \nFai pure." in out


def test_plain_is_one_line_per_turn(conversation):
    out = plain(conversation, names={"SPEAKER_00": "Ada Verdolini"})
    lines = out.rstrip("\n").split("\n")
    assert len(lines) == len(conversation)
    assert lines[0] == "Ada Verdolini: Comincio io, se non vi dispiace."
    assert lines[3] == "Unattributed: (rumore di fondo)"


def test_srt_blocks_carry_index_times_and_speaker(segments):
    out = srt(segments, names={"SPEAKER_00": "Ada Verdolini"})
    assert out.startswith("1\n00:00:00,000 --> 00:00:02,500\n[Ada Verdolini] Comincio io.\n")
    assert "[Voice 2] Fai pure.\n" in out
    assert "," in out and "-->" in out


def test_srt_skips_blank_segments(segments):
    padded = [segments[0], {"speaker": "SPEAKER_00", "start": 2.5, "end": 2.6, "text": "  "},
              segments[1]]
    out = srt(padded)
    assert "00:00:02,500 --> 00:00:02,600" not in out
    assert out.count("-->") == 2


# Regression. Was: scriba/export.py:195: the cue number comes from enumerate( over all  segments, but blank segments are skipped afterwards, so the numbering has  holes (1, 3, 4.... SRT cue numbers are supposed to be consecutive; strict  parsers and some editors reject or renumber the file.
def test_srt_numbers_cues_consecutively(segments):
    padded = [segments[0], {"speaker": "SPEAKER_00", "start": 2.5, "end": 2.6, "text": ""},
              segments[1]]
    out = srt(padded)
    numbers = [b.splitlines()[0] for b in out.strip().split("\n\n")]
    assert numbers == ["1", "2"]


def test_vtt_starts_with_the_header_and_tags_the_voice(segments):
    out = vtt(segments, names={"SPEAKER_00": "Ada Verdolini"})
    assert out.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:02.500" in out
    assert "<v Ada Verdolini>Comincio io." in out
    assert "<v Voice 2>Fai pure." in out
    assert "," not in out.split("\n\n")[1].splitlines()[0]


def test_vtt_skips_blank_segments(segments):
    padded = [segments[0], {"speaker": "SPEAKER_01", "start": 2.5, "end": 2.6, "text": None},
              segments[1]]
    assert vtt(padded).count("-->") == 2


def test_payload_json_round_trips(conversation, segments):
    raw = payload_json(meta={"duration": 30.0, "language": "it"}, segments=segments,
                       turns=conversation, names={"SPEAKER_00": "Ada Verdolini"},
                       matches=[{"label": "SPEAKER_00", "score": 0.81}])
    data = json.loads(raw)
    assert set(data) == {"meta", "names", "speaker_matches", "turns", "segments"}
    assert data["meta"]["language"] == "it"
    assert data["names"]["SPEAKER_00"] == "Ada Verdolini"
    assert data["speaker_matches"][0]["score"] == 0.81
    assert len(data["turns"]) == len(conversation)
    assert len(data["segments"]) == len(segments)


def test_payload_json_keeps_non_ascii_readable(segments):
    raw = payload_json(meta={}, segments=segments,
                       turns=[turn("SPEAKER_00", 0.0, "perché però")],
                       names={}, matches=[])
    assert "perché però" in raw


# --------------------------------------------------------------------------- #
# write_all / FILENAMES / _sweep
# --------------------------------------------------------------------------- #

STEM = "Nota vocale 2026-03-04"


@pytest.fixture
def job(conversation, segments):
    return {
        "turns": conversation,
        "segments": segments,
        "names": {"SPEAKER_00": "Ada Verdolini"},
        "meta": {
            "title": "Riunione di redazione",
            "recorded": datetime(2026, 3, 4, 9, 30),
            "duration": 95.0,
            "language": "it",
            "source_file": "nota vocale.m4a",
            "speaker_stats": {"SPEAKER_00": 60.0, "SPEAKER_01": 35.0},
            "unresolved": ["Voice 2"],
            "recorded_source": "modification",
            "device": "Registratore da tasca",
        },
        "matches": [{"label": "SPEAKER_00", "score": 0.81}],
    }


@pytest.mark.parametrize("fmt,expected", [
    ("source", f"{STEM} (source).md"),
    ("md", f"{STEM}.md"),
    ("txt", f"{STEM}.txt"),
    ("srt", f"{STEM}.srt"),
    ("vtt", f"{STEM}.vtt"),
    ("json", f"{STEM}.json"),
])
def test_write_all_names_each_format(tmp_path, job, fmt, expected):
    written = write_all(tmp_path, STEM, [fmt], **job)
    assert [p.name for p in written] == [expected]
    assert written[0].exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [expected]


def test_write_all_writes_every_requested_format(tmp_path, job):
    formats = ["source", "md", "txt", "srt", "vtt", "json"]
    written = write_all(tmp_path, STEM, formats, **job)
    assert [p.name for p in written] == [FILENAMES[f].format(stem=STEM) for f in formats]
    assert all(p.exists() for p in written)


def test_write_all_creates_the_output_directory(tmp_path, job):
    outdir = tmp_path / "esiti" / "marzo"
    written = write_all(outdir, STEM, ["txt"], **job)
    assert outdir.is_dir()
    assert written[0].read_text().startswith("Ada Verdolini: ")


def test_write_all_source_file_holds_the_source_document(tmp_path, job):
    path = write_all(tmp_path, STEM, ["source"], **job)[0]
    text = path.read_text()
    assert text.startswith("# Riunione di redazione")
    assert "- **Date**: 2026-03-04 09:30 (from the file's modification time" in text
    assert "- **Duration**: 00:01:35" in text
    assert "- **Recorded with**: Registratore da tasca" in text
    assert "- **Unidentified voices**: Voice 2." in text
    assert "**Voice 2 (unidentified) (uncertain)** [00:05]: Fai pure." in text


def test_write_all_source_falls_back_to_the_stem_as_title(tmp_path, job):
    job["meta"].pop("title")
    text = write_all(tmp_path, STEM, ["source"], **job)[0].read_text()
    assert text.startswith(f"# {STEM}")


def test_write_all_json_serialises_the_recorded_datetime(tmp_path, job):
    data = json.loads(write_all(tmp_path, STEM, ["json"], **job)[0].read_text())
    assert data["meta"]["recorded"] == "2026-03-04T09:30:00"
    assert data["meta"]["duration"] == 95.0
    assert data["speaker_matches"] == job["matches"]


def test_write_all_rejects_an_unknown_format(tmp_path, job):
    with pytest.raises(ValueError) as excinfo:
        write_all(tmp_path, STEM, ["txt", "tzt"], **job)
    assert "tzt" in str(excinfo.value)


def test_filenames_and_suffixes_agree():
    # _sweep can only clean up what it recognises, so every name the module writes
    # has to end in a suffix the sweep looks at.
    for pattern in FILENAMES.values():
        assert pattern.endswith((".md", ".txt", ".srt", ".vtt", ".json")), pattern


def test_write_all_removes_a_stale_output_of_a_dropped_format(tmp_path, job):
    write_all(tmp_path, STEM, ["source", "md", "txt", "json"], **job)
    write_all(tmp_path, STEM, ["md"], **job)
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{STEM}.md"]


def test_write_all_leaves_the_folder_of_another_recording_alone(tmp_path, job):
    other = tmp_path / "Riunione di aprile.md"
    other.write_text("altro verbale")
    write_all(tmp_path, STEM, ["md"], **job)
    assert other.exists()
    assert other.read_text() == "altro verbale"


def test_sweep_keeps_files_the_module_never_writes(tmp_path, job):
    audio = tmp_path / f"{STEM}.wav"
    picture = tmp_path / f"{STEM}.png"
    audio.write_bytes(b"RIFF-not-really")
    picture.write_bytes(b"\x89PNG")
    write_all(tmp_path, STEM, ["md"], **job)
    assert audio.exists() and picture.exists()
    assert audio.read_bytes() == b"RIFF-not-really"


def test_sweep_deletes_only_the_formats_that_are_gone(tmp_path):
    for name in [f"{STEM}.md", f"{STEM}.txt", f"{STEM}.srt", f"{STEM} (source).md"]:
        (tmp_path / name).write_text("vecchio")
    _sweep(tmp_path, STEM, ["md", "srt"])
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{STEM}.md", f"{STEM}.srt"]


def test_sweep_ignores_an_unknown_format_name(tmp_path):
    (tmp_path / f"{STEM}.md").write_text("vecchio")
    _sweep(tmp_path, STEM, ["md", "tzt"])
    assert (tmp_path / f"{STEM}.md").exists()


def test_sweep_does_not_touch_a_directory(tmp_path):
    (tmp_path / f"{STEM}.txt").mkdir()
    _sweep(tmp_path, STEM, ["md"])
    assert (tmp_path / f"{STEM}.txt").is_dir()


def test_sweep_on_an_empty_folder_is_a_no_op(tmp_path):
    _sweep(tmp_path, STEM, ["md", "txt"])
    assert list(tmp_path.iterdir()) == []


# Regression. Was: scriba/export.py:252: _sweep globs '{stem}*' and deletes anything with  a known suffix, so it destroys files it could never have written as long as  the name starts with the stem: the user's own '<stem> appunti.md', or the  output of a neighbouring recording called '<stem>-2'. Its own docstrin
def test_sweep_keeps_a_file_that_merely_starts_with_the_stem(tmp_path):
    notes = tmp_path / f"{STEM} appunti miei.md"
    neighbour = tmp_path / f"{STEM}-2.md"
    notes.write_text("appunti scritti a mano")
    neighbour.write_text("altra registrazione")
    _sweep(tmp_path, STEM, ["md"])
    assert notes.exists(), "a hand-written note in the output folder was deleted"
    assert neighbour.exists(), "another recording's output was deleted"


# Regression. Was: scriba/export.py:252: the stem is interpolated into a glob pattern  without escaping, so square brackets in a filename become a character class.  For a recording called 'Memo [2026]' the glob matches nothing, the sweep  silently does nothing, and stale outputs survive next to the fresh ones - 
def test_sweep_handles_a_stem_containing_glob_characters(tmp_path):
    bracketed = "Memo [2026]"
    (tmp_path / f"{bracketed}.md").write_text("fresco")
    stale = tmp_path / f"{bracketed}.txt"
    stale.write_text("vecchio")
    _sweep(tmp_path, bracketed, ["md"])
    assert not stale.exists(), "stale output survived the sweep"


# Regression. Was: scriba/export.py:275 vs 311: write_all sweeps first and validates the  format names last, so a typo in one format wipes the previous outputs of the  formats that were dropped from `keep` and writes the ones before the typo,  then raises. The transcription has already run at that point, so the 
def test_write_all_validates_formats_before_deleting_anything(tmp_path, job):
    write_all(tmp_path, STEM, ["md", "txt"], **job)
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(ValueError):
        write_all(tmp_path, STEM, ["md", "tzt"], **job)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_module_exposes_the_expected_format_names():
    assert set(FILENAMES) == {"source", "md", "txt", "srt", "vtt", "json"}
    assert export.FILENAMES["source"].format(stem="X") == "X (source).md"
