"""Tests for the one number that says what the transcript is missing.

``Diarization.gaps`` compares two engines that looked at the same audio: the
diarizer, which says where somebody was speaking, and the transcript, which says
where words came out. Where the first has something and the second has nothing,
either the recogniser's gate threw away real speech or the diarizer heard a chair
move. The distinction cannot be made from the timestamps alone, so the method
does not try: it returns the places, so a person can go and listen.

The threshold carries all the weight. Counting every gap, a real 405 s
conversation came out at 81.8 s missing, which read as a quarter of the meeting
lost. It was not: the median gap was 0.14 s, because pyannote's turns include the
pauses inside them and no transcript writes down breathing. The tests below pin
the behaviour that keeps that number honest, in particular that a gap under the
threshold is not reported at all.
"""

from __future__ import annotations

from scriba.diarize import Diarization, SpeakerTurn


def dia(*spans: tuple[float, float]) -> Diarization:
    return Diarization(
        turns=[SpeakerTurn(a, b, f"SPEAKER_{i:02d}") for i, (a, b) in enumerate(spans)],
        embeddings={},
    )


def segs(*spans: tuple[float, float]) -> list[dict]:
    return [{"start": a, "end": b, "text": "..."} for a, b in spans]


def test_a_transcript_covering_every_turn_reports_nothing_missing():
    assert dia((0, 30), (40, 60)).gaps(segs((0, 30), (40, 60))) == []


def test_a_stretch_of_speech_with_no_words_is_reported_with_its_position():
    # The five seconds at the end of the turn are the whole point: the caller gets
    # 25.0 so somebody can scrub there, not just the fact that something is gone.
    assert dia((0, 30)).gaps(segs((0, 25))) == [(25.0, 30.0)]


def test_breathing_inside_a_turn_is_not_reported_as_missing_speech():
    """The failure this file exists for.

    pyannote keeps short pauses inside a turn. Counting them made a transcript
    that had missed nothing at all look as though it had lost a quarter of the
    conversation, which is worse than not measuring: it is a number that cries
    wolf on every run.
    """
    quiet = segs((0, 10), (10.4, 20), (20.3, 30))
    assert dia((0, 30)).gaps(quiet) == []
    # And the total agrees with the list, rather than being counted separately.
    missing, speech = dia((0, 30)).untranscribed(quiet)
    assert (missing, speech) == (0.0, 30.0)


def test_the_threshold_is_the_caller_s_to_move():
    quiet = segs((0, 10), (10.4, 30))
    assert dia((0, 30)).gaps(quiet, min_gap=0.1) == [(10.0, 10.4)]


def test_words_with_no_timestamps_do_not_count_as_coverage():
    """Digits come back from the aligner with no ``start``.

    The Spanish alignment vocabulary has no characters for digits, so "40" comes
    out unplaced. Treating that as covered audio would hide exactly the kind of
    line worth checking.
    """
    holes = dia((0, 30)).gaps([{"start": None, "end": None, "text": "40"},
                               {"start": 0, "end": 20, "text": "..."}])
    assert holes == [(20.0, 30.0)]


def test_overlapping_segments_count_once():
    # Two speakers talking over each other used to make the covered time larger
    # than the file, which turned a real gap into a negative number.
    d = dia((0, 30))
    assert d.gaps(segs((0, 20), (5, 25))) == [(25.0, 30.0)]
    assert d.untranscribed(segs((0, 20), (5, 25)))[0] == 5.0


def test_a_transcript_running_past_the_diarized_speech_reports_nothing():
    assert dia((10, 20)).gaps(segs((0, 40))) == []


def test_speech_total_is_the_diarized_time_not_the_file_length():
    _, speech = dia((0, 10), (20, 30)).untranscribed(segs((0, 10), (20, 30)))
    assert speech == 20.0
