"""Tests for the two decisions nothing else in the suite looks at.

* ``scriba.diarize.assign`` — which speaker said which words. ``to_turns`` has plenty
  of coverage, but it only glues together what ``assign`` already decided: hand it
  segments whose ``speaker`` is wrong and it merges them into a wrong transcript
  without a single test noticing. The tests here build diarization turns and whisper
  segments by hand, word timings included, and check the decision itself: the speaker
  with the *most* overlap wins, a segment that overlaps nobody stays unattributed
  instead of being guessed, and a word stranded in silence inherits from its
  neighbour only if the neighbour is within half a second.

* ``scriba.watch`` — when a dropped file is ready to be transcribed, and what gets
  remembered afterwards. Three rules carry all the weight: a file is processed only
  once its size has stopped changing (otherwise a 200 MB copy in flight is
  transcribed half-written, and the transcript stops mid-conversation with nothing
  to show that it did); a file that came out with a transcript is recorded in
  ``.scriba-done`` so a restart does not do it again; and a file that failed is
  remembered in memory only, so it stops eating the log for the rest of the run and
  comes round again when the watcher is restarted.

Nothing here loads a model, decodes audio, or reaches the network. The pipeline is
replaced wholesale by a recorder, ``time.sleep`` by a counter, and every path lives
under ``tmp_path``. The files "audio" are a few bytes of invented content; nothing
ever reads them.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scriba import diarize, watch


# --------------------------------------------------------------------------- #
# PART ONE — diarize.assign
# --------------------------------------------------------------------------- #

def dia(*specs: tuple[float, float, str]) -> diarize.Diarization:
    """A diarization from (start, end, speaker) triples, in the order given.

    The order is deliberate and not sorted here: `assign` walks `dia.turns` as it
    finds them, and one test below depends on that.
    """
    return diarize.Diarization(
        turns=[diarize.SpeakerTurn(start=s, end=e, speaker=spk) for s, e, spk in specs]
    )


def seg(start: float, end: float, text: str = "una frase", words: list | None = None) -> dict:
    """A whisper segment with only the keys `assign` reads."""
    out: dict = {"start": start, "end": end, "text": text}
    if words is not None:
        out["words"] = words
    return out


def word(text: str, start: float | None = None, end: float | None = None) -> dict:
    """An aligned word. Omit the times to get the untimed kind the aligner emits
    for digits and symbols."""
    out: dict = {"word": text}
    if start is not None:
        out["start"] = start
    if end is not None:
        out["end"] = end
    return out


# --- which speaker a segment belongs to ------------------------------------- #

def test_segment_goes_to_the_speaker_it_overlaps_most():
    """Ada holds 0–3, Bruno 3–10. A segment from 2 to 6 is one second of Ada and
    three of Bruno, so it is Bruno's."""
    segments = diarize.assign([seg(2.0, 6.0, "y entonces le dije")],
                              dia((0.0, 3.0, "ADA"), (3.0, 10.0, "BRUNO")))
    assert segments[0]["speaker"] == "BRUNO"


def test_segment_goes_to_the_other_speaker_when_the_overlap_leans_the_other_way():
    """Same two turns, the segment moved: three seconds of Ada against one of
    Bruno. Together with the test above this pins *most* overlap and not least,
    and not "whichever turn comes first"."""
    segments = diarize.assign([seg(0.0, 4.0, "y entonces le dije")],
                              dia((0.0, 3.0, "ADA"), (3.0, 10.0, "BRUNO")))
    assert segments[0]["speaker"] == "ADA"


def test_confidence_is_the_share_of_the_segment_the_winner_covers():
    segments = diarize.assign([seg(2.0, 6.0)],
                              dia((0.0, 3.0, "ADA"), (3.0, 10.0, "BRUNO")))
    # three of the segment's four seconds belong to Bruno
    assert segments[0]["speaker_confidence"] == pytest.approx(0.75)


def test_a_third_speaker_in_the_middle_does_not_win_on_a_sliver():
    """Three turns, the segment straddling all three. The one it barely clips must
    not be the answer."""
    segments = diarize.assign(
        [seg(1.0, 9.0)],
        dia((0.0, 4.0, "ADA"), (4.0, 4.5, "CARLA"), (4.5, 12.0, "BRUNO")),
    )
    # Ada 3.0, Carla 0.5, Bruno 4.5
    assert segments[0]["speaker"] == "BRUNO"


def test_segment_that_overlaps_nobody_is_left_unattributed():
    """Silence between two turns. Nothing is guessed: no speaker, no confidence.

    The alternative, handing it to the nearest turn, is how a cough or a passing
    car ends up quoted as somebody's sentence.
    """
    segments = diarize.assign([seg(3.2, 3.8, "(ruido)")],
                              dia((0.0, 3.0, "ADA"), (4.0, 10.0, "BRUNO")))
    assert segments[0]["speaker"] is None
    assert segments[0]["speaker_confidence"] == 0.0


def test_a_segment_that_touches_a_turn_only_at_its_edge_does_not_count_as_overlap():
    """Ada ends exactly where the segment starts: shared instant, zero duration."""
    segments = diarize.assign([seg(3.0, 3.5)], dia((0.0, 3.0, "ADA")))
    assert segments[0]["speaker"] is None


def test_a_perfect_tie_resolves_to_the_first_turn_and_says_it_is_half_sure():
    """One second each. The winner is the first turn in diarization order, and the
    confidence is honest about it."""
    turns = dia((0.0, 2.0, "ADA"), (2.0, 4.0, "BRUNO"))
    segments = diarize.assign([seg(1.0, 3.0)], turns)
    assert segments[0]["speaker"] == "ADA"
    assert segments[0]["speaker_confidence"] == pytest.approx(0.5)


def test_the_tie_break_follows_the_turn_order_it_was_handed():
    """Same tie, turns listed the other way round. Documented rather than admired:
    on a true tie there is no better answer, but the result should at least be
    reproducible instead of depending on dictionary weather."""
    turns = dia((2.0, 4.0, "BRUNO"), (0.0, 2.0, "ADA"))
    segments = diarize.assign([seg(1.0, 3.0)], turns)
    assert segments[0]["speaker"] == "BRUNO"


def test_a_segment_with_no_end_is_zero_length_and_gets_nobody():
    """`end` missing falls back to `start`, which overlaps nothing at all."""
    segments = diarize.assign([{"start": 1.0, "text": "sí"}], dia((0.0, 10.0, "ADA")))
    assert segments[0]["speaker"] is None


# --- which speaker a word belongs to ---------------------------------------- #

def test_words_are_attributed_one_by_one_when_they_carry_timings():
    """The segment as a whole is Bruno's, but its first two words are Ada's.

    This is the case whisper produces constantly: it cuts on silence and audio
    length, not on who is talking, so a single segment routinely spans a handover.
    """
    words = [
        word("buenos", 0.5, 1.0),
        word("días", 1.0, 1.5),
        word("gracias", 4.0, 4.5),
        word("por", 4.5, 5.0),
        word("venir", 5.0, 5.5),
    ]
    segments = diarize.assign([seg(0.0, 6.0, "buenos días gracias por venir", words)],
                              dia((0.0, 2.0, "ADA"), (2.0, 10.0, "BRUNO")))
    got = [w["speaker"] for w in segments[0]["words"]]
    assert got == ["ADA", "ADA", "BRUNO", "BRUNO", "BRUNO"]
    assert segments[0]["speaker"] == "BRUNO"


def test_a_word_straddling_the_handover_goes_to_the_speaker_it_overlaps_most():
    """Half a second of Ada against a whole second of Bruno, inside one word."""
    words = [word("entonces", 2.5, 4.0)]
    segments = diarize.assign([seg(2.5, 4.0, "entonces", words)],
                              dia((0.0, 3.0, "ADA"), (3.0, 10.0, "BRUNO")))
    assert segments[0]["words"][0]["speaker"] == "BRUNO"


def test_orphan_word_inherits_from_the_previous_word_within_half_a_second():
    """A word landing in the silence after Ada, a quarter of a second later, is
    still Ada's — even though the segment as a whole belongs to Bruno."""
    words = [word("mira", 2.5, 2.75), word("eh", 3.0, 3.25)]
    segments = diarize.assign([seg(0.0, 10.0, "mira eh", words)],
                              dia((0.0, 3.0, "ADA"), (4.0, 10.0, "BRUNO")))
    assert [w["speaker"] for w in segments[0]["words"]] == ["ADA", "ADA"]


def test_orphan_word_exactly_at_the_gap_still_inherits():
    """Half a second on the nose is inside the window."""
    words = [word("mira", 2.25, 2.5), word("eh", 3.0, 3.25)]
    segments = diarize.assign([seg(0.0, 10.0, "mira eh", words)],
                              dia((0.0, 3.0, "ADA"), (4.0, 10.0, "BRUNO")))
    assert segments[0]["words"][1]["speaker"] == "ADA"


def test_orphan_word_just_outside_the_gap_does_not_inherit():
    """A full second after Ada's last word, in silence: too far to be hers.

    It falls back to the segment's own speaker, Bruno. Widen that window and every
    stray word starts attaching itself to whoever spoke last, which is exactly the
    one-word-turn-attributed-to-the-wrong-person defect the docstring is about.
    """
    words = [word("mira", 2.25, 2.5), word("eh", 3.5, 3.75)]
    segments = diarize.assign([seg(0.0, 10.0, "mira eh", words)],
                              dia((0.0, 3.0, "ADA"), (4.0, 10.0, "BRUNO")))
    assert [w["speaker"] for w in segments[0]["words"]] == ["ADA", "BRUNO"]


def test_the_gap_is_measured_from_the_previous_word_not_from_the_segment():
    """Two stranded words in a row, 0.25 s apart each time, walking away from Ada.

    Both stay Ada's: the window is renewed word by word. The test exists because a
    gap measured from the segment start, or from the last *attributed* word, gives
    Bruno for the second one.
    """
    words = [word("mira", 2.5, 2.75), word("eh", 3.0, 3.25), word("bueno", 3.5, 3.75)]
    segments = diarize.assign([seg(0.0, 10.0, "mira eh bueno", words)],
                              dia((0.0, 3.0, "ADA"), (4.0, 10.0, "BRUNO")))
    assert [w["speaker"] for w in segments[0]["words"]] == ["ADA", "ADA", "ADA"]


def test_the_first_word_of_a_segment_has_nothing_to_inherit_from():
    """Stranded with no predecessor: the segment's speaker, not a crash."""
    words = [word("eh", 3.2, 3.4), word("gracias", 5.0, 5.5)]
    segments = diarize.assign([seg(0.0, 10.0, "eh gracias", words)],
                              dia((0.0, 3.0, "ADA"), (4.0, 10.0, "BRUNO")))
    assert [w["speaker"] for w in segments[0]["words"]] == ["BRUNO", "BRUNO"]


def test_words_without_timestamps_take_the_speaker_of_the_word_before_them():
    """The aligner leaves digits and symbols untimed. They follow the context."""
    words = [word("son", 0.5, 1.0), word("15"), word("euros", 1.5, 2.0)]
    segments = diarize.assign([seg(0.0, 2.0, "son 15 euros", words)],
                              dia((0.0, 2.0, "ADA"),))
    assert [w["speaker"] for w in segments[0]["words"]] == ["ADA", "ADA", "ADA"]


def test_an_untimed_word_at_the_head_of_a_segment_takes_the_segment_speaker():
    words = [word("15"), word("euros", 1.5, 2.0)]
    segments = diarize.assign([seg(0.0, 2.0, "15 euros", words)],
                              dia((0.0, 2.0, "ADA"),))
    assert [w["speaker"] for w in segments[0]["words"]] == ["ADA", "ADA"]


def test_words_of_an_unattributed_segment_stay_unattributed():
    """Nobody was talking, so no word invents a speaker."""
    words = [word("mm", 3.2, 3.4), word("hm", 3.5, 3.7)]
    segments = diarize.assign([seg(3.0, 3.9, "mm hm", words)],
                              dia((0.0, 3.0, "ADA"), (4.0, 10.0, "BRUNO")))
    assert segments[0]["speaker"] is None
    assert [w["speaker"] for w in segments[0]["words"]] == [None, None]


def test_word_level_off_leaves_the_words_alone():
    """The segment still gets a speaker; the words are not touched at all."""
    words = [word("buenos", 0.5, 1.0)]
    segments = diarize.assign([seg(0.0, 2.0, "buenos", words)],
                              dia((0.0, 2.0, "ADA"),), word_level=False)
    assert segments[0]["speaker"] == "ADA"
    assert "speaker" not in segments[0]["words"][0]


# --- the empty ends --------------------------------------------------------- #

def test_no_segments_gives_no_segments():
    assert diarize.assign([], dia((0.0, 10.0, "ADA"))) == []


def test_a_diarization_with_no_turns_leaves_the_segments_as_they_were():
    """Nothing to attribute from, so nothing is claimed: no speaker key appears.

    Downstream `to_turns` reads `seg.get("speaker")` and gets None, which is the
    honest answer for a file diarization found nobody in.
    """
    segments = [seg(0.0, 2.0, "hola")]
    out = diarize.assign(segments, diarize.Diarization(turns=[]))
    assert out is segments
    assert "speaker" not in segments[0]


def test_segments_are_annotated_in_place_and_handed_back():
    """The caller in pipeline.py uses the return value; other code keeps the list it
    passed in. Both have to see the speakers."""
    segments = [seg(0.0, 2.0, "hola")]
    out = diarize.assign(segments, dia((0.0, 2.0, "ADA")))
    assert out is segments
    assert segments[0]["speaker"] == "ADA"


def test_every_segment_of_a_conversation_is_decided_separately():
    """A short exchange end to end, to catch anything that decides once and reuses
    the answer for the rest of the file."""
    conversation = [
        seg(0.0, 2.5, "buenos días"),
        seg(3.0, 5.5, "gracias por venir"),
        seg(6.0, 8.0, "cuando quiera"),
    ]
    out = diarize.assign(conversation, dia((0.0, 2.8, "ADA"),
                                           (2.8, 5.8, "BRUNO"),
                                           (5.8, 9.0, "ADA")))
    assert [s["speaker"] for s in out] == ["ADA", "BRUNO", "ADA"]


# --------------------------------------------------------------------------- #
# PART TWO — scriba.watch
# --------------------------------------------------------------------------- #

class Harness:
    """A watched folder with the pipeline and the clock replaced.

    `Job` becomes a recorder: it notes the size of the file at the moment the job
    was built (that is the whole point of the stability check — a job built too
    early sees a half-copied file) and never transcribes anything. `time.sleep`
    becomes a round counter that runs whatever the test wants to happen between
    rounds and stops the loop with a KeyboardInterrupt, which is how `watch`
    terminates in real life too.
    """

    def __init__(self, monkeypatch, tmp_path: Path) -> None:
        self._monkeypatch = monkeypatch
        self.folder = tmp_path / "inbox"
        self.folder.mkdir()
        self.failing: set[str] = set()
        self.unresolved: dict[str, list[str]] = {}
        self.between: dict[int, object] = {}
        self.built: list[tuple[str, int]] = []   # (name, size when the job was built)
        self.ran: list[str] = []
        self.log: list[str] = []
        self.slept: list[float] = []
        self.round = 0

    # -- the folder ------------------------------------------------------- #

    def drop(self, name: str, size: int = 64) -> Path:
        path = self.folder / name
        path.write_bytes(b"n" * size)
        return path

    def grow(self, name: str, size: int) -> None:
        (self.folder / name).write_bytes(b"n" * size)

    def marker(self, name: str) -> Path:
        return self.folder.resolve() / watch.DONE_MARK / name

    def markers(self) -> list[str]:
        return sorted(p.name for p in (self.folder.resolve() / watch.DONE_MARK).glob("*"))

    # -- the run ---------------------------------------------------------- #

    def run(self, rounds: int = 3, interval: float = 0.01) -> None:
        """Scan the folder `rounds` times, then stop as a ctrl-c would.

        Counters reset on every call, so a test can run the watcher twice over the
        same folder and ask what the second, fresh process did.
        """
        self.built, self.ran, self.slept, self.round = [], [], [], 0
        harness = self

        class FakeJob:
            def __init__(self, path, settings=None, *, report=lambda m: None):
                self.path = Path(path)
                harness.built.append((self.path.name, self.path.stat().st_size))

            def run(self):
                harness.ran.append(self.path.name)
                if self.path.name in harness.failing:
                    raise RuntimeError("ffmpeg fell over")
                return SimpleNamespace(
                    outputs=[self.path.with_suffix(".md"), self.path.with_suffix(".txt")],
                    unresolved=harness.unresolved.get(self.path.name, []),
                    dossier_path=self.path.with_name("who-is-who.md"),
                )

        def fake_sleep(seconds: float) -> None:
            harness.slept.append(seconds)
            harness.round += 1
            hook = harness.between.get(harness.round)
            if hook is not None:
                hook()
            if harness.round >= rounds:
                raise KeyboardInterrupt

        self._monkeypatch.setattr(watch, "Job", FakeJob)
        self._monkeypatch.setattr(watch, "time", SimpleNamespace(sleep=fake_sleep))
        watch.watch(self.folder, None, interval=interval, report=self.log.append)


@pytest.fixture
def harness(monkeypatch, tmp_path):
    return Harness(monkeypatch, tmp_path)


# --- waiting for the file to stop growing ----------------------------------- #

def test_a_file_is_not_touched_the_first_time_it_is_seen(harness):
    """One scan is not enough: a file seen once has an unknown size, not a stable
    one. It takes a second scan agreeing with the first."""
    harness.drop("entrevista-ada.m4a", size=100)
    harness.run(rounds=1)
    assert harness.ran == []


def test_a_file_still_being_copied_is_left_alone_until_its_size_settles(harness):
    """The 200 MB case, in miniature: the file is 100 bytes on the first scan and
    400 on the second, and only the third scan, which agrees with the second, hands
    it over. The job must see the finished file.

    Without the stability check the job is built on scan one, over 100 bytes of a
    400-byte file, and the transcript stops in the middle of the conversation with
    nothing anywhere to say that it was cut short.
    """
    harness.drop("entrevista-ada.m4a", size=100)
    harness.between[1] = lambda: harness.grow("entrevista-ada.m4a", 400)
    harness.run(rounds=4)
    assert harness.built == [("entrevista-ada.m4a", 400)]
    assert harness.ran == ["entrevista-ada.m4a"]


def test_a_file_that_keeps_growing_is_never_handed_over(harness):
    """A slow copy, or an iCloud download that trickles in: it grows on every round
    and the watcher simply waits."""
    harness.drop("larga.m4a", size=100)
    for i in range(1, 6):
        harness.between[i] = (lambda n: lambda: harness.grow("larga.m4a", 100 + 100 * n))(i)
    harness.run(rounds=6)
    assert harness.built == []
    assert harness.ran == []


def test_a_file_that_shrinks_back_also_waits(harness):
    """Same rule in the other direction. Any change of size restarts the count, so
    a rewrite in place does not slip through on the comparison with a stale value.
    """
    harness.drop("regrabada.m4a", size=400)
    harness.between[1] = lambda: harness.grow("regrabada.m4a", 120)
    harness.run(rounds=3)
    assert harness.built == [("regrabada.m4a", 120)]


def test_a_settled_file_is_handed_over_exactly_once(harness):
    """Six scans, one transcription. The size stays stable throughout, which under
    a rule that only checked stability would be reason to run it again and again.
    """
    harness.drop("quieta.m4a", size=100)
    harness.run(rounds=6)
    assert harness.ran == ["quieta.m4a"]


def test_the_interval_it_was_given_is_the_interval_it_waits(harness):
    harness.drop("quieta.m4a")
    harness.run(rounds=3, interval=0.25)
    assert harness.slept == [0.25, 0.25, 0.25]


# --- what gets remembered --------------------------------------------------- #

def test_a_finished_file_is_recorded_in_the_done_folder(harness):
    harness.drop("entrevista-ada.m4a")
    harness.run(rounds=3)
    assert harness.marker("entrevista-ada.m4a").exists()
    assert harness.markers() == ["entrevista-ada.m4a"]


def test_a_file_already_recorded_is_not_transcribed_again_by_a_new_watcher(harness):
    """The reason the marker is on disk and not only in memory: the watcher is a
    process someone restarts, and the folder still holds every file ever dropped in
    it. Without the marker the second run transcribes the lot from scratch.
    """
    harness.drop("entrevista-ada.m4a")
    harness.run(rounds=3)
    assert harness.ran == ["entrevista-ada.m4a"]

    harness.run(rounds=2)          # a fresh watcher over the same folder
    assert harness.ran == []
    assert harness.built == []


def test_a_watcher_starting_over_a_folder_of_done_files_says_so_and_skips_them(harness):
    """The marker is read at startup, before the first scan."""
    harness.drop("vieja.m4a")
    done = harness.folder / watch.DONE_MARK
    done.mkdir(exist_ok=True)
    (done / "vieja.m4a").touch()
    harness.run(rounds=2)
    assert harness.ran == []
    assert any("already processed" in line for line in harness.log)


def test_the_done_folder_is_not_mistaken_for_something_to_transcribe(harness):
    harness.run(rounds=2)
    assert harness.ran == []
    assert harness.markers() == []


def test_files_that_are_not_audio_are_ignored(harness):
    """Notes, `.DS_Store`, and the half-written dotfiles a sync client leaves
    behind. None of them are a recording, and none of them get a marker either."""
    harness.drop("notas.txt")
    harness.drop(".entrevista-ada.m4a")
    harness.drop("captura.png")
    harness.run(rounds=3)
    assert harness.ran == []
    assert harness.markers() == []


def test_several_settled_files_are_taken_in_name_order(harness):
    harness.drop("b-bruno.m4a")
    harness.drop("a-ada.wav")
    harness.drop("c-carla.mp3")
    harness.run(rounds=3)
    assert harness.ran == ["a-ada.wav", "b-bruno.m4a", "c-carla.mp3"]
    assert harness.markers() == ["a-ada.wav", "b-bruno.m4a", "c-carla.mp3"]


def test_a_file_that_vanishes_between_the_listing_and_the_size_check_is_skipped(harness, monkeypatch):
    """The race: `iterdir` lists it, and by the time its size is read the Shortcut
    has moved it away, or iCloud has evicted it.

    Modelled by letting the first `stat` of that name through — the one `is_file`
    does, which is what makes the file look present — and failing every later one.
    That is the exact window the guard covers. An unguarded `stat` raises out of
    `watch()` entirely, since the only handler in the loop is for KeyboardInterrupt,
    and the watcher is then gone: no message, no process, and a folder that quietly
    stops producing transcripts. A watcher that has silently exited looks exactly
    like one with nothing to do.
    """
    harness.drop("a-se-esfuma.m4a")
    harness.drop("b-se-queda.m4a")

    real_stat = Path.stat
    seen_once: set[str] = set()

    def flaky_stat(self, *args, **kwargs):
        if self.name == "a-se-esfuma.m4a":
            if "a-se-esfuma.m4a" in seen_once:
                raise FileNotFoundError(2, "No such file or directory", str(self))
            seen_once.add("a-se-esfuma.m4a")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)

    harness.run(rounds=4)
    assert harness.ran == ["b-se-queda.m4a"]       # the watcher lived; the neighbour ran
    assert harness.markers() == ["b-se-queda.m4a"]  # and nothing was recorded for the ghost


def test_a_file_dropped_in_while_the_watcher_runs_is_picked_up(harness):
    harness.drop("primera.m4a")
    harness.between[2] = lambda: harness.drop("segunda.m4a")
    harness.run(rounds=5)
    assert harness.ran == ["primera.m4a", "segunda.m4a"]


def test_unidentified_voices_are_named_in_the_report_with_the_dossier_to_read(harness):
    """This branch reads three fields off the pipeline's result. A typo in any of
    them raises inside the `try`, and the failure is swallowed and printed as an
    ordinary pipeline error, so the transcript looks fine and nobody is ever told
    there are voices left to name.
    """
    harness.drop("entrevista-ada.m4a")
    harness.unresolved["entrevista-ada.m4a"] = ["SPEAKER_01", "SPEAKER_02"]
    harness.run(rounds=3)
    joined = "\n".join(harness.log)
    assert "SPEAKER_01" in joined and "SPEAKER_02" in joined
    assert "who-is-who.md" in joined
    assert not any("error:" in line for line in harness.log)


# --- when the pipeline fails ------------------------------------------------ #

def test_a_failing_file_does_not_bring_the_watcher_down(harness):
    """The error is reported and the loop carries on to the next file, which gets
    its transcript and its marker as though nothing had happened."""
    harness.drop("a-rota.m4a")
    harness.drop("b-buena.m4a")
    harness.failing.add("a-rota.m4a")
    harness.run(rounds=3)
    assert harness.ran == ["a-rota.m4a", "b-buena.m4a"]
    assert any("error: ffmpeg fell over" in line for line in harness.log)
    assert harness.markers() == ["b-buena.m4a"]


def test_a_failing_file_is_not_retried_every_five_seconds_forever(harness):
    """Six scans, one attempt: the name goes into the in-memory `failed` set.

    Retrying it every interval would fill the log with the same traceback until
    somebody noticed, and would keep a broken file ahead of the ones that work.
    Nothing is written to `.scriba-done` for it, though — that folder means "this
    file has a transcript", and this one has none.
    """
    harness.drop("rota.m4a")
    harness.failing.add("rota.m4a")
    harness.run(rounds=6)
    assert harness.ran == ["rota.m4a"]
    assert not harness.marker("rota.m4a").exists()
    assert harness.markers() == []


def test_the_watcher_says_what_it_did_with_the_file_it_could_not_transcribe(harness):
    """The user has to be able to tell "skipped, restart me" from "done, ignored"
    by reading the log, because those two look identical in the folder."""
    harness.drop("rota.m4a")
    harness.failing.add("rota.m4a")
    harness.run(rounds=3)
    assert any("left in place" in line for line in harness.log)


def test_a_transient_failure_is_tried_again_by_the_next_watcher(harness):
    """The point of holding failures in memory instead of on disk.

    Most of the ways the pipeline fails are not about the file: ffmpeg not on the
    PATH yet, a HuggingFace token not exported, the model not downloaded, a machine
    out of memory or disk. Those fail every recording dropped in that afternoon. If
    the failures were marked done, fixing the cause would leave a folder full of
    recordings that the watcher then skips in silence, with nothing in the log to
    say why. Here the second watcher picks the file up again.
    """
    harness.drop("rota.m4a")
    harness.failing.add("rota.m4a")
    harness.run(rounds=3)
    assert harness.ran == ["rota.m4a"]

    harness.failing.clear()        # ffmpeg installed, token exported, disk freed
    harness.run(rounds=3)
    assert harness.ran == ["rota.m4a"]
    assert harness.marker("rota.m4a").exists()


def test_a_file_that_failed_and_then_succeeds_is_marked_done_only_once(harness):
    """The retry writes exactly one marker, and a third watcher skips it."""
    harness.drop("rota.m4a")
    harness.failing.add("rota.m4a")
    harness.run(rounds=3)
    harness.failing.clear()
    harness.run(rounds=3)
    assert harness.markers() == ["rota.m4a"]

    harness.run(rounds=3)
    assert harness.ran == []


def test_a_failed_file_replaced_while_the_watcher_runs_waits_for_the_next_one(harness):
    """Current shape of `failed`, pinned rather than admired.

    The set holds names, and the name is checked before the size, so re-exporting
    the recording and dropping it in under the same name — the first thing anyone
    does when told a file could not be transcribed — does not get it looked at again
    this run. The 900-byte replacement below sits there untouched, and only the
    restart the log promised picks it up. Keying `failed` on the size as well, or
    dropping the name from it when the size changes, would make the obvious fix work
    without the restart. If that changes, this test changes with it: the second
    assertion is the whole difference.
    """
    harness.drop("rota.m4a", size=100)
    harness.failing.add("rota.m4a")

    def re_export():
        harness.failing.clear()            # whatever broke it is gone
        harness.grow("rota.m4a", 900)      # exported again, this time whole

    harness.between[2] = re_export         # right after the attempt that failed
    harness.run(rounds=6)
    assert harness.ran == ["rota.m4a"]                  # attempted once
    assert harness.built == [("rota.m4a", 100)]         # the good copy never looked at
    assert harness.markers() == []

    harness.run(rounds=3)                  # the restart the user was told about
    assert harness.built == [("rota.m4a", 900)]
    assert harness.markers() == ["rota.m4a"]
