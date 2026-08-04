"""Tests for the cross-check between two engines.

``verify.compare`` exists to answer one question: are the words in the transcript
where they were actually said. It answers it by asking a second engine and looking
at where the two disagree, which means the tests here are mostly about not crying
wolf. A check that reports a disagreement on every recording is a check nobody
reads, and then the one real finding goes past unnoticed with the rest.

The recording that prompted all this had exactly one: eleven words placed eleven
seconds before they were said. Everything else agreed to a hundredth of a second.

Nothing here loads a model. Both sides are hand-written word lists.
"""

from __future__ import annotations

from scriba.verify import compare


def said(words: list[tuple[str, float]]) -> list[dict]:
    """One segment carrying the words given, in the shape the aligner produces."""
    return [{
        "start": words[0][1], "end": words[-1][1] + 0.3,
        "text": " ".join(w for w, _ in words),
        "words": [{"word": w, "start": t, "end": t + 0.3} for w, t in words],
    }]


BASE = [("hola", 1.0), ("que", 1.5), ("tal", 2.0), ("todo", 2.5), ("bien", 3.0)]


def test_two_engines_that_agree_report_no_zones():
    rep = compare(said(BASE), said(BASE))
    assert rep.zones == []
    assert rep.compared == 5
    assert rep.median_offset == 0.0
    assert rep.within_one_second == 1.0


def test_small_differences_are_the_aligner_s_noise_and_not_reported():
    drifted = [(w, t + 0.4) for w, t in BASE]
    rep = compare(said(BASE), said(drifted))
    assert rep.zones == []
    assert rep.within_one_second == 1.0


def test_a_sentence_filed_in_the_wrong_place_is_reported_with_both_positions():
    """The failure this module was written for.

    One engine put a whole sentence eleven seconds before it was said. Cutting
    those seconds out of the recording and playing them back is what settled it,
    so the report has to carry the two positions, not just a complaint.
    """
    late = [("hola", 1.0), ("que", 1.5), ("tal", 2.0),
            ("todo", 13.0), ("bien", 13.5)]
    rep = compare(said(late), said(BASE))
    assert len(rep.zones) == 1
    z = rep.zones[0]
    assert (z.start, z.other_start) == (2.5, 13.0)
    assert z.words == 2
    assert z.text == "todo bien"


def test_words_only_one_engine_heard_are_counted_and_not_treated_as_a_gap():
    extra = BASE + [("gracias", 3.5), ("adios", 4.0)]
    rep = compare(said(BASE), said(extra))
    assert rep.only_there == 2      # the second engine heard two words more
    assert rep.only_here == 0
    assert rep.zones == []


def test_a_disagreement_that_moves_a_line_to_another_speaker_says_so():
    """Timing only matters because attribution is decided by it.

    A sentence in the wrong place is a curiosity until the wrong place belongs to
    somebody else, and then it is the one failure this tool cannot have quietly.
    """
    turns = [{"start": 0.0, "end": 6.0, "speaker": "SPEAKER_00"},
             {"start": 6.0, "end": 20.0, "speaker": "SPEAKER_01"}]
    late = [("hola", 1.0), ("que", 1.5), ("tal", 2.0), ("todo", 13.0), ("bien", 13.5)]
    rep = compare(said(late), said(BASE), turns=turns)
    assert rep.zones[0].changes_speaker is True

    # And the same displacement inside one person's turn does not.
    turns_one = [{"start": 0.0, "end": 20.0, "speaker": "SPEAKER_00"}]
    rep = compare(said(late), said(BASE), turns=turns_one)
    assert rep.zones[0].changes_speaker is False


def test_neighbouring_disagreements_are_one_finding_not_eleven():
    late = [(w, t) for w, t in BASE[:2]] + [(w, t + 11.0) for w, t in BASE[2:]]
    rep = compare(said(late), said(BASE))
    assert len(rep.zones) == 1
    assert rep.zones[0].words == 3


def test_punctuation_and_case_are_not_a_disagreement():
    """One engine writes "Vale." and the other "vale", and neither is wrong."""
    a = said([("Vale,", 1.0), ("QUE", 1.5), ("tal.", 2.0)])
    b = said([("vale", 1.0), ("qué", 1.5), ("tal", 2.0)])
    rep = compare(a, b)
    assert rep.compared == 3
    assert rep.zones == []


def test_an_empty_second_opinion_reports_nothing_rather_than_dividing_by_zero():
    rep = compare(said(BASE), [])
    assert rep.compared == 0
    assert rep.zones == []
    assert rep.only_here == 5


def test_a_stretch_the_transcript_never_wrote_down_is_reported():
    """The failure that timing comparison cannot see.

    On a public benchmark clip of 21 seconds and 51 words, whisper large-v3 with
    this project's settings produced two words and stopped. There is no timing
    disagreement to find, because there are no words to compare: they are absent,
    and the document reads perfectly without them.
    """
    # Twenty seconds of speech at a normal rate, which is what the failing clip
    # was, against the two words the transcript came back with.
    words = ("el uso adecuado de los blogs puede empoderar a los estudiantes "
             "para escribir sobre lo que han aprendido durante el curso").split()
    spoken = [(w, 1.0 + i * 0.8) for i, w in enumerate(words)]
    rep = compare(said([("oravec", 1.0), ("2002", 1.4)]), said(spoken))

    assert len(rep.silences) == 1
    s = rep.silences[0]
    assert s.words_there == len(words) and s.words_here == 2
    assert s.start == 1.0
    assert "empoderar" in s.text


def test_a_transcript_that_has_the_words_reports_no_silence():
    spoken = [(w, 1.0 + i * 0.8) for i, w in enumerate(
        "el uso adecuado de los blogs puede empoderar a los estudiantes".split())]
    assert compare(said(spoken), said(spoken)).silences == []


def test_a_displaced_passage_is_one_finding_and_not_two():
    """The stretch it left empty is the same defect as the stretch it filled early.

    Reported separately, one displacement reads as a displacement plus a hole,
    and the hole is the more alarming of the two. On a real conversation this is
    exactly what happened: eleven words filed eleven seconds early, and the
    seconds they came from looked like lost speech.
    """
    words = ("el martes por la manana quedamos en la oficina de siempre "
             "y traemos los papeles firmados").split()
    late = [(w, 1.0 + i * 0.7) for i, w in enumerate(words)]      # transcript, early
    spoken = [(w, 13.0 + i * 0.7) for i, w in enumerate(words)]   # audio, later
    rep = compare(said(late), said(spoken))

    assert len(rep.zones) == 1
    assert rep.silences == []


def test_a_short_utterance_is_not_a_silence():
    """Two words missing from a two-second aside is noise, not a finding.

    A check that fires on every recording is a check nobody reads, and then the
    twenty-one seconds that really are gone go past with the rest.
    """
    brief = [("vale", 1.0), ("si", 1.3), ("claro", 1.6), ("bueno", 1.9), ("ya", 2.2)]
    assert compare(said([("hm", 1.0)]), said(brief)).silences == []
