"""Tests for the recurring-voice search, the who-is-who briefing, and the pure
helpers that turn a diarization into conversational turns.

Nothing here loads a model, opens an audio file, or touches the network. Every
embedding is a hand-built unit vector, every transcript is invented, and the only
paths used are the ones pytest hands out under tmp_path.

The three modules under test:

* ``scriba.recurring`` — which voice comes back across recordings, and where to put
  the threshold that decides "same person".
* ``scriba.naming`` — the briefing a human (or an LLM) reads to put names to voices.
* ``scriba.diarize`` — ``to_turns`` and ``_unpack``, the two pure helpers.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from scriba import diarize, naming, recurring
from scriba.recurring import (
    MIN_USABLE_THRESHOLD,
    Cluster,
    Sample,
    cluster,
    cross_file_similarities,
    suggest_threshold,
)

EMB_DIM = 16


# --------------------------------------------------------------------------- #
# helpers: embeddings with a cosine similarity we choose in advance
# --------------------------------------------------------------------------- #

def unit(angle_deg: float, dim: int = EMB_DIM) -> np.ndarray:
    """A unit vector in the first two dimensions, at `angle_deg` from the x axis.

    Two of these have cosine similarity cos(a - b), so a test can ask for exactly
    the similarity it wants instead of hoping random vectors land there.
    """
    t = math.radians(angle_deg)
    v = np.zeros(dim, dtype=np.float32)
    v[0] = math.cos(t)
    v[1] = math.sin(t)
    return v


def sample(tmp_path, filename: str, label: str, angle: float,
           speech: float = 60.0) -> Sample:
    return Sample(
        file=tmp_path / filename,
        label=label,
        embedding=unit(angle),
        speech_seconds=speech,
        longest_start=0.0,
        longest_end=speech,
    )


def test_unit_helper_gives_the_similarity_it_promises():
    from scriba.voices import cosine
    a, b = unit(0), unit(60)
    got = float(cosine(a, b[None, :]).reshape(-1)[0])
    assert got == pytest.approx(0.5, abs=1e-6)


# --------------------------------------------------------------------------- #
# recurring.cross_file_similarities
# --------------------------------------------------------------------------- #

def test_cross_file_skips_pairs_from_inside_one_recording(tmp_path):
    """Two voices in one room were separated because they sound different.

    That pair says nothing about whether a person sounds like themselves on another
    day, which is the only question the threshold has to answer, so it must not be
    in the sample at all.
    """
    samples = [
        sample(tmp_path, "monday.m4a", "SPEAKER_00", 0),
        sample(tmp_path, "monday.m4a", "SPEAKER_01", 60),
        sample(tmp_path, "tuesday.m4a", "SPEAKER_00", 0),
        sample(tmp_path, "tuesday.m4a", "SPEAKER_01", 60),
    ]
    sims = cross_file_similarities(samples)
    # 6 unordered pairs in total, 2 of them inside a single file.
    assert len(sims) == 4
    # monday/tuesday same angle twice (1.0), and the two 60-degree crossings (0.5).
    assert sorted(round(float(x), 3) for x in sims) == [0.5, 0.5, 1.0, 1.0]


def test_cross_file_would_have_had_more_pairs_without_the_exclusion(tmp_path):
    """The exclusion is what the function is for: prove it actually drops something."""
    import itertools

    from scriba.voices import cosine

    samples = [
        sample(tmp_path, "a.m4a", "SPEAKER_00", 0),
        sample(tmp_path, "a.m4a", "SPEAKER_01", 45),
        sample(tmp_path, "a.m4a", "SPEAKER_02", 90),
        sample(tmp_path, "b.m4a", "SPEAKER_00", 5),
    ]
    naive = [float(cosine(x.embedding, y.embedding[None, :]).reshape(-1)[0])
             for x, y in itertools.combinations(samples, 2)]
    assert len(naive) == 6
    assert len(cross_file_similarities(samples)) == 3


def test_cross_file_same_name_in_two_files_is_kept(tmp_path):
    """Diarizer labels restart at SPEAKER_00 on every file, so an equal label across
    two recordings is not the same person and an unequal one is not two people. Only
    the file identity is allowed to exclude a pair."""
    samples = [
        sample(tmp_path, "one.m4a", "SPEAKER_00", 0),
        sample(tmp_path, "two.m4a", "SPEAKER_00", 90),
    ]
    sims = cross_file_similarities(samples)
    assert len(sims) == 1
    assert float(sims[0]) == pytest.approx(0.0, abs=1e-6)


def test_cross_file_is_sorted_ascending(tmp_path):
    samples = [
        sample(tmp_path, "a.m4a", "S0", 0),
        sample(tmp_path, "b.m4a", "S0", 90),
        sample(tmp_path, "c.m4a", "S0", 45),
        sample(tmp_path, "d.m4a", "S0", 10),
    ]
    sims = cross_file_similarities(samples)
    assert len(sims) == 6
    assert list(sims) == sorted(sims)


def test_cross_file_all_speakers_in_one_recording_gives_nothing(tmp_path):
    samples = [
        sample(tmp_path, "solo.m4a", "SPEAKER_00", 0),
        sample(tmp_path, "solo.m4a", "SPEAKER_01", 30),
        sample(tmp_path, "solo.m4a", "SPEAKER_02", 70),
    ]
    sims = cross_file_similarities(samples)
    assert len(sims) == 0
    assert isinstance(sims, np.ndarray)


@pytest.mark.parametrize("samples_in", [0, 1])
def test_cross_file_degenerate_inputs(tmp_path, samples_in):
    samples = [sample(tmp_path, f"f{i}.m4a", "S0", i * 10) for i in range(samples_in)]
    assert len(cross_file_similarities(samples)) == 0


def test_cross_file_values_are_plain_floats(tmp_path):
    """They get fed to numpy statistics and printed in messages; float32 leaking out
    of the comparison would show up as 0.5000000074505806 in the report."""
    samples = [sample(tmp_path, "a.m4a", "S0", 0), sample(tmp_path, "b.m4a", "S0", 60)]
    sims = cross_file_similarities(samples)
    assert sims.dtype == np.float64


# --------------------------------------------------------------------------- #
# recurring.suggest_threshold
# --------------------------------------------------------------------------- #

def realistic_mixture(seed: int, *, n_strangers: int = 300,
                      same=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A corpus-shaped sample: a dense band of stranger pairs, a few same-person ones.

    Returns (all_sims_sorted, strangers, same_person) so a test can check *which*
    pairs ended up above the cut, not merely where the cut landed.
    """
    rng = np.random.default_rng(seed)
    strangers = rng.normal(0.06, 0.07, n_strangers)
    same = np.linspace(0.45, 0.90, 24) if same is None else np.asarray(same, dtype=float)
    allsims = np.array(sorted(np.concatenate([strangers, same])))
    return allsims, strangers, same


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_threshold_on_a_realistic_mixture_is_sane(seed):
    sims, strangers, same = realistic_mixture(seed)
    cut, why = suggest_threshold(sims)

    # Above the floor, so cluster() will accept it.
    assert cut >= MIN_USABLE_THRESHOLD
    # And not so high that the recurring voice disappears.
    assert cut < float(same.max())
    # No stranger pair is called a match. This is the property that matters: a false
    # link merges two people into one "recurring voice" and the answer is wrong.
    assert (strangers > cut).sum() == 0
    # Most of the same-person pairs survive the cut.
    assert (same > cut).sum() >= len(same) // 2
    assert "stand out" in why


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_threshold_is_nowhere_near_the_otsu_failure(seed):
    """Otsu's method splits the stranger mass down its middle and returns ~0.08,
    which calls half the strangers a match. The median+MAD rule must not land there.
    """
    sims, strangers, _ = realistic_mixture(seed)
    cut, _ = suggest_threshold(sims)
    otsu_like = 0.08
    assert cut > 3 * otsu_like
    # Concretely: the bad threshold would have accepted a pile of stranger pairs.
    assert (strangers > otsu_like).sum() > 50
    assert (strangers > cut).sum() == 0


def test_threshold_does_not_cut_a_spread_out_voice_off_at_its_widest_gap():
    """One person recorded across many days spreads from 0.4 to 0.9 instead of
    clustering. The widest-gap rule finds its biggest hole *inside* that spread and
    halves the person. Here the cut has to sit below the whole spread's lower end,
    or at worst leave the bulk of it above."""
    person = np.linspace(0.40, 0.90, 40)
    sims, strangers, same = realistic_mixture(3, same=person)
    cut, _ = suggest_threshold(sims)

    widest_gap_cut = float(person[np.argmax(np.diff(person)) + 1])
    # The widest gap inside a near-uniform spread is somewhere in the middle of it.
    assert 0.40 < widest_gap_cut < 0.90
    assert (strangers > cut).sum() == 0
    # A clear majority of the person's own pairs stay above the cut, and single
    # linkage in cluster() then chains the rest of them back in.
    assert (same > cut).sum() >= len(same) // 3


def test_threshold_needs_at_least_thirty_pairs():
    sims = np.array(sorted(np.concatenate([np.linspace(0.0, 0.1, 25),
                                           np.linspace(0.8, 0.9, 4)])))
    assert len(sims) == 29
    cut, why = suggest_threshold(sims)
    assert cut == 0.0
    assert "29" in why and "too few" in why


def test_threshold_thirty_pairs_is_enough_to_try():
    rng = np.random.default_rng(11)
    sims = np.array(sorted(np.concatenate([rng.normal(0.05, 0.05, 27),
                                           np.array([0.80, 0.85, 0.90])])))
    assert len(sims) == 30
    cut, why = suggest_threshold(sims)
    assert cut > MIN_USABLE_THRESHOLD
    assert "too few" not in why


def test_threshold_on_identical_pairs_returns_the_sentinel():
    """A degenerate input must not produce a nonsense threshold: with no spread at
    all, median + 5 * 0 would be the median itself, and clustering at the median of
    everything joins the whole corpus into one group."""
    sims = np.full(80, 0.42)
    cut, why = suggest_threshold(sims)
    assert cut == 0.0
    # Nothing stands above the background because there is nothing but background,
    # which is the one case where this wording is the true one.
    assert why == "every pair is identical, so there is nothing to separate"
    # And the sentinel is refused downstream rather than used.
    with pytest.raises(ValueError):
        cluster([], cut)


def test_threshold_when_nothing_stands_out_returns_the_sentinel():
    """A corpus of strangers only: nobody recurs, and saying so is the right answer."""
    rng = np.random.default_rng(7)
    sims = np.array(sorted(rng.normal(0.06, 0.07, 400)))
    cut, why = suggest_threshold(sims)
    assert cut == 0.0
    assert "no voice repeats" in why


def test_threshold_on_a_flat_uniform_spread_returns_the_sentinel():
    """Uniform noise from 0 to 1 has no background band and no outliers. The rule
    must decline rather than cut somewhere in the middle of a continuum."""
    sims = np.linspace(0.0, 1.0, 200)
    cut, why = suggest_threshold(sims)
    assert cut == 0.0
    assert "no pair stands out" in why


def test_threshold_never_returns_a_low_but_nonzero_cut_on_stranger_noise():
    """Anything between 0 and the 0.30 floor would be worse than the sentinel: it
    reads like an answer and groups strangers together."""
    for seed in range(20):
        rng = np.random.default_rng(seed)
        sims = np.array(sorted(rng.normal(0.06, 0.07, 300)))
        cut, _ = suggest_threshold(sims)
        assert cut == 0.0 or cut >= MIN_USABLE_THRESHOLD


def test_threshold_can_land_below_the_floor_and_cluster_is_what_refuses_it(tmp_path):
    """A small corpus can produce a cut that separates *this* data while sitting far
    too low to mean "same person". The threshold search does not know that, and the
    floor in cluster() is the second half of the safety net."""
    rng = np.random.default_rng(7)
    sims = np.array(sorted(np.concatenate([rng.normal(0.05, 0.06, 32),
                                           rng.uniform(0.30, 0.45, 4)])))
    cut, why = suggest_threshold(sims)
    assert 0.0 < cut < MIN_USABLE_THRESHOLD, why

    samples = [sample(tmp_path, "a.m4a", "S0", 0), sample(tmp_path, "b.m4a", "S0", 60)]
    with pytest.raises(ValueError) as err:
        cluster(samples, cut)
    assert "below the 0.30 floor" in str(err.value)


def test_threshold_with_a_nan_among_the_pairs_falls_back_to_the_sentinel():
    """A NaN cannot come from a finite embedding and diarize.run drops the non-finite
    ones, but arithmetic that quietly yields a NaN threshold would be clustered with
    afterwards. It comes out as the sentinel instead."""
    sims = np.array(sorted(np.concatenate([np.linspace(0.0, 0.2, 40), [np.nan]])))
    cut, _ = suggest_threshold(sims)
    assert cut == 0.0


def test_threshold_uses_the_median_not_the_mean():
    """The recurring voice is inside the sample. A mean-and-standard-deviation rule
    is dragged upward by it; the median and MAD are not.

    Constructed so the two disagree loudly: the same-person pairs are numerous
    enough to move a mean, and the cut still has to sit above the stranger band and
    below the recurring voice.
    """
    rng = np.random.default_rng(2)
    strangers = rng.normal(0.05, 0.04, 200)
    same = np.linspace(0.70, 0.95, 60)
    sims = np.array(sorted(np.concatenate([strangers, same])))

    mean_rule = float(sims.mean()) + 5.0 * float(sims.std())
    cut, _ = suggest_threshold(sims)

    assert mean_rule > 1.0            # the mean rule cuts above any cosine: useless
    assert MIN_USABLE_THRESHOLD < cut < float(same.min())
    assert (strangers > cut).sum() == 0
    assert (same > cut).sum() == len(same)


def test_threshold_message_reports_how_far_out_the_closest_pair_sits():
    sims, _, _ = realistic_mixture(0)
    cut, why = suggest_threshold(sims)
    assert "deviations above" in why
    assert f"of {len(sims)} pairs" in why


def test_threshold_returns_a_python_float():
    sims, _, _ = realistic_mixture(0)
    cut, why = suggest_threshold(sims)
    assert type(cut) is float
    assert isinstance(why, str) and why


def test_threshold_ignores_the_input_order():
    sims, _, _ = realistic_mixture(1)
    shuffled = sims.copy()
    np.random.default_rng(0).shuffle(shuffled)
    assert suggest_threshold(shuffled)[0] == pytest.approx(suggest_threshold(sims)[0])


def test_threshold_on_a_mostly_constant_input_with_real_outliers():
    """More than half the pairs being *exactly* equal drives the MAD to zero, so the
    spread cannot be measured even though a group of clear outliers sits on top.

    The value returned is the sentinel either way. What the message must not do is
    call this "every pair is identical", which is false as soon as anything stands
    above them and sends the reader looking for the wrong problem.
    """
    sims = np.array(sorted(np.concatenate([np.full(60, 0.05),
                                           np.linspace(0.50, 0.90, 20)])))
    cut, why = suggest_threshold(sims)

    assert cut == 0.0                     # safe: cluster() will refuse it
    assert "identical" not in why
    assert "20 of 80 pairs stand apart" in why
    assert "single value" in why
    assert "Duplicate copies" in why      # the actual cause, and what to do about it


def test_threshold_on_duplicate_copies_of_one_recording(tmp_path):
    """Where the flattened MAD comes from in practice. Four byte-identical copies of
    a one-voice memo produce cross-file pairs that are all exactly 1.0."""
    vec = unit(23)
    samples = [Sample(tmp_path / f"copy{i}.m4a", "SPEAKER_00", vec, 120.0, 0.0, 120.0)
               for i in range(9)]
    sims = cross_file_similarities(samples)
    assert len(sims) == 36
    assert float(sims.min()) == pytest.approx(1.0)

    cut, why = suggest_threshold(sims)
    assert cut == 0.0
    assert "every pair is identical" in why
    with pytest.raises(ValueError):
        cluster(samples, cut)


# --------------------------------------------------------------------------- #
# recurring.cluster
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("threshold", [0.0, 0.1, 0.29, 0.2999, -0.5])
def test_cluster_refuses_a_threshold_under_the_floor(tmp_path, threshold):
    samples = [sample(tmp_path, "a.m4a", "S0", 0), sample(tmp_path, "b.m4a", "S0", 0)]
    with pytest.raises(ValueError) as err:
        cluster(samples, threshold)
    assert f"{MIN_USABLE_THRESHOLD:.2f}" in str(err.value)


def test_cluster_refuses_the_sentinel_before_joining_everyone_to_everyone(tmp_path):
    """Clustering at zero links every pair and returns one group holding the whole
    corpus, which on screen reads exactly like a confident answer."""
    samples = [sample(tmp_path, f"f{i}.m4a", "S0", i * 30) for i in range(6)]
    with pytest.raises(ValueError):
        cluster(samples, 0.0)


def test_cluster_accepts_exactly_the_floor(tmp_path):
    samples = [sample(tmp_path, "a.m4a", "S0", 0), sample(tmp_path, "b.m4a", "S0", 0)]
    groups = cluster(samples, MIN_USABLE_THRESHOLD)
    assert len(groups) == 1


def test_cluster_empty_corpus_with_a_usable_threshold(tmp_path):
    assert cluster([], 0.75) == []


def test_cluster_groups_the_voice_that_recurs(tmp_path):
    """Three recordings, the same person in all three, a different guest in two."""
    me = [sample(tmp_path, f"day{i}.m4a", "SPEAKER_00", i * 2) for i in range(3)]
    guests = [
        sample(tmp_path, "day0.m4a", "SPEAKER_01", 80),
        sample(tmp_path, "day1.m4a", "SPEAKER_01", 175),
    ]
    groups = cluster(me + guests, 0.80)

    assert len(groups[0].files) == 3
    assert {s.label for s in groups[0].samples} == {"SPEAKER_00"}
    assert all(len(g.files) < 3 for g in groups[1:])


def test_cluster_never_joins_two_speakers_of_one_recording_directly(tmp_path):
    """Even with identical embeddings: the diarizer already ruled they are two
    people, and a direct link would contradict it."""
    a = sample(tmp_path, "meeting.m4a", "SPEAKER_00", 0)
    b = sample(tmp_path, "meeting.m4a", "SPEAKER_01", 0)
    assert np.array_equal(a.embedding, b.embedding)
    groups = cluster([a, b], 0.5)
    assert len(groups) == 2
    assert all(len(g.samples) == 1 for g in groups)


def test_cluster_single_linkage_chains_through_a_middle_recording(tmp_path):
    """A person in a quiet room and the same person in a corridor may not look
    similar to each other, while both look similar to a third recording between
    them. Chaining is a flaw in general and the point here."""
    quiet = sample(tmp_path, "quiet.m4a", "S0", 0)
    middle = sample(tmp_path, "middle.m4a", "S0", 36.87)     # cos == 0.80 to quiet
    corridor = sample(tmp_path, "corridor.m4a", "S0", 73.74)  # cos == 0.28 to quiet

    from scriba.voices import cosine
    direct = float(cosine(quiet.embedding, corridor.embedding[None, :]).reshape(-1)[0])
    assert direct < 0.5

    groups = cluster([quiet, middle, corridor], 0.75)
    assert len(groups) == 1
    assert len(groups[0].samples) == 3


def test_cluster_leaves_unrelated_voices_apart(tmp_path):
    samples = [
        sample(tmp_path, "a.m4a", "S0", 0),
        sample(tmp_path, "b.m4a", "S0", 90),
        sample(tmp_path, "c.m4a", "S0", 180),
    ]
    groups = cluster(samples, 0.75)
    assert len(groups) == 3


def test_cluster_orders_by_recordings_first_then_speech(tmp_path):
    """A voice in three files beats a voice that talks for an hour in one."""
    wide = [sample(tmp_path, f"w{i}.m4a", "S0", 0, speech=30.0) for i in range(3)]
    loud = [sample(tmp_path, "loud.m4a", "S0", 90, speech=3600.0)]
    groups = cluster(wide + loud, 0.9)

    assert [len(g.files) for g in groups] == [3, 1]
    assert groups[0].speech_seconds == pytest.approx(90.0)
    assert groups[1].speech_seconds == pytest.approx(3600.0)


def test_cluster_breaks_ties_on_speech(tmp_path):
    quiet = [sample(tmp_path, f"q{i}.m4a", "S0", 0, speech=10.0) for i in range(2)]
    talkative = [sample(tmp_path, f"t{i}.m4a", "S0", 120, speech=500.0) for i in range(2)]
    groups = cluster(quiet + talkative, 0.9)

    assert [len(g.files) for g in groups] == [2, 2]
    assert groups[0].speech_seconds > groups[1].speech_seconds


def test_cluster_keeps_every_sample_exactly_once(tmp_path):
    samples = [
        sample(tmp_path, "a.m4a", "S0", 0),
        sample(tmp_path, "a.m4a", "S1", 88),
        sample(tmp_path, "b.m4a", "S0", 3),
        sample(tmp_path, "b.m4a", "S1", 200),
        sample(tmp_path, "c.m4a", "S0", 6),
    ]
    groups = cluster(samples, 0.9)
    placed = [s for g in groups for s in g.samples]
    assert len(placed) == len(samples)
    assert {id(s) for s in placed} == {id(s) for s in samples}


def test_cluster_higher_threshold_never_produces_fewer_groups(tmp_path):
    samples = [sample(tmp_path, f"f{i}.m4a", "S0", i * 12) for i in range(8)]
    sizes = [len(cluster(samples, t)) for t in (0.30, 0.60, 0.90, 0.99)]
    assert sizes == sorted(sizes)


def test_cluster_chaining_can_still_gather_two_speakers_of_one_file(tmp_path):
    """Documents a known cost of single linkage rather than blessing it.

    Same-file pairs are skipped, but nothing stops two speakers of one recording
    from being joined through a shared neighbour in another file. Real embeddings of
    two people in one room rarely both clear the threshold against a third, so this
    is the price of the chaining the docstring asks for, not a defect to fix here.
    """
    a0 = sample(tmp_path, "room.m4a", "SPEAKER_00", 0)
    a1 = sample(tmp_path, "room.m4a", "SPEAKER_01", 40)
    bridge = sample(tmp_path, "other.m4a", "SPEAKER_00", 20)

    groups = cluster([a0, a1, bridge], 0.90)
    assert len(groups) == 1
    assert len(groups[0].files) == 2
    assert len(groups[0].samples) == 3


# --------------------------------------------------------------------------- #
# recurring.Cluster
# --------------------------------------------------------------------------- #

def test_cluster_dataclass_files_and_speech(tmp_path):
    c = Cluster(samples=[
        sample(tmp_path, "a.m4a", "S0", 0, speech=10.0),
        sample(tmp_path, "a.m4a", "S1", 30, speech=5.0),
        sample(tmp_path, "b.m4a", "S0", 0, speech=7.5),
    ])
    assert c.files == {tmp_path / "a.m4a", tmp_path / "b.m4a"}
    assert c.speech_seconds == pytest.approx(22.5)


def test_cluster_centroid_is_a_unit_vector_between_its_members(tmp_path):
    c = Cluster(samples=[
        sample(tmp_path, "a.m4a", "S0", -30),
        sample(tmp_path, "b.m4a", "S0", 30),
    ])
    centroid = c.centroid()
    assert float(np.linalg.norm(centroid)) == pytest.approx(1.0, abs=1e-5)
    # Halfway between -30 and +30 degrees is the x axis.
    assert centroid[0] == pytest.approx(1.0, abs=1e-5)
    assert centroid[1] == pytest.approx(0.0, abs=1e-5)


def test_cluster_centroid_of_a_single_sample_is_that_sample(tmp_path):
    s = sample(tmp_path, "a.m4a", "S0", 17)
    centroid = Cluster(samples=[s]).centroid()
    assert centroid == pytest.approx(s.embedding, abs=1e-6)


def test_empty_cluster_has_no_files_and_no_speech():
    c = Cluster()
    assert c.files == set()
    assert c.speech_seconds == 0.0


# --------------------------------------------------------------------------- #
# the three recurring functions in sequence
# --------------------------------------------------------------------------- #

def test_one_recording_gives_no_evidence_and_the_chain_says_so(tmp_path):
    """A single file produces no cross-file pairs, the threshold falls back to its
    sentinel, and clustering refuses it instead of returning one confident group."""
    samples = [
        sample(tmp_path, "only.m4a", "SPEAKER_00", 0),
        sample(tmp_path, "only.m4a", "SPEAKER_01", 45),
    ]
    sims = cross_file_similarities(samples)
    cut, why = suggest_threshold(sims)
    assert len(sims) == 0
    assert cut == 0.0 and "too few" in why
    with pytest.raises(ValueError):
        cluster(samples, cut)


def a_corpus(tmp_path, seed: int, *, n_files: int = 9, drift: float = 0.75):
    """Nine recordings: the same owner in every one, a different guest each time.

    Embeddings are 256-dimensional like the real ones. Random vectors in that many
    dimensions are nearly orthogonal, which is what gives the stranger pairs their
    narrow band around zero; the owner is the same direction jittered by `drift`,
    which puts their pairs around 0.6 the way two recordings of one person on
    different days do.
    """
    from scriba.voices import l2norm

    rng = np.random.default_rng(seed)

    def rand_unit():
        return l2norm(rng.normal(0, 1, 256).astype(np.float32)).astype(np.float32)

    owner = rand_unit()
    samples = []
    for i in range(n_files):
        path = tmp_path / f"memo{i:02d}.m4a"
        drifted = l2norm(owner + drift * rand_unit()).astype(np.float32)
        samples.append(Sample(path, "SPEAKER_00", drifted, 200.0, 0.0, 200.0))
        samples.append(Sample(path, "SPEAKER_01", rand_unit(), 90.0, 0.0, 90.0))
    return samples


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_corpus_with_a_recurring_voice_survives_the_whole_chain(tmp_path, seed):
    samples = a_corpus(tmp_path, seed)

    sims = cross_file_similarities(samples)
    assert len(sims) == 144          # 18 speakers, minus the 9 same-file pairs
    cut, why = suggest_threshold(sims)
    assert cut >= MIN_USABLE_THRESHOLD, why

    groups = cluster(samples, cut)
    owner = groups[0]
    assert len(owner.files) == 9
    assert {s.label for s in owner.samples} == {"SPEAKER_00"}
    # The nine guests never met each other, so nothing else groups.
    assert len(groups) == 10


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_corpus_of_strangers_only_refuses_to_invent_a_recurring_voice(tmp_path, seed):
    """Same shape, but nobody comes back: eighteen different people. The chain has
    to end in the sentinel and the refusal, not in a group."""
    from scriba.voices import l2norm

    rng = np.random.default_rng(1000 + seed)
    samples = []
    for i in range(9):
        path = tmp_path / f"memo{i:02d}.m4a"
        for label in ("SPEAKER_00", "SPEAKER_01"):
            vec = l2norm(rng.normal(0, 1, 256).astype(np.float32)).astype(np.float32)
            samples.append(Sample(path, label, vec, 120.0, 0.0, 120.0))

    cut, why = suggest_threshold(cross_file_similarities(samples))
    assert cut == 0.0, why
    with pytest.raises(ValueError):
        cluster(samples, cut)


# --------------------------------------------------------------------------- #
# naming.extract_cues
# --------------------------------------------------------------------------- #

def a_turn(speaker, start, end, text, conf=1.0):
    return {"speaker": speaker, "start": start, "end": end,
            "text": text, "confidence": conf}


def test_cues_attribute_the_sentence_to_whoever_said_it():
    """The distinction automatic attribution always gets wrong: SPEAKER_00 greeting
    Marilena is evidence about SPEAKER_01, and the cue belongs to SPEAKER_00 because
    SPEAKER_00 is the one who uttered it."""
    turns = [
        a_turn("SPEAKER_00", 0.0, 3.0, "Buongiorno Marilena, cominciamo quando vuoi."),
        a_turn("SPEAKER_01", 3.5, 8.0, "Grazie. Mi chiamo Marilena e lavoro qui da due anni."),
    ]
    cues = naming.extract_cues(turns, "it")
    assert set(cues) == {"SPEAKER_00", "SPEAKER_01"}
    assert cues["SPEAKER_00"] == ["Buongiorno Marilena, cominciamo quando vuoi."]
    assert "Mi chiamo Marilena" in cues["SPEAKER_01"][0]


def test_cues_return_the_whole_sentence_not_the_bare_name():
    turns = [a_turn("A", 0.0, 4.0, "Senti Ottavia, il documento lo mandi tu?")]
    cues = naming.extract_cues(turns, "it")
    assert cues["A"] == ["Senti Ottavia, il documento lo mandi tu?"]


def test_cues_english_self_introduction():
    turns = [a_turn("SPEAKER_00", 0.0, 4.0, "Hello, my name is Fenwick and I take the notes.")]
    cues = naming.extract_cues(turns, "en")
    assert cues["SPEAKER_00"] == ["Hello, my name is Fenwick and I take the notes."]


def test_cues_spanish_patterns():
    turns = [a_turn("S", 0.0, 4.0, "Hola Ludovica, soy Tazio y empiezo yo.")]
    cues = naming.extract_cues(turns, "es")
    assert len(cues["S"]) == 1


def test_cues_unknown_language_falls_back_to_english():
    turns = [a_turn("S", 0.0, 4.0, "Hi Ludovica, thanks for coming.")]
    assert naming.extract_cues(turns, "de") == naming.extract_cues(turns, "en")
    assert naming.extract_cues(turns, "de")["S"]


def test_cues_skip_capitalised_words_that_are_not_names():
    """`grazie Vale` and `hola Bueno` trip the greeting pattern; neither is a person."""
    turns = [
        a_turn("S", 0.0, 2.0, "Grazie Vale, allora Bueno."),
        a_turn("T", 2.0, 4.0, "Hola Vale, gracias Bueno."),
    ]
    assert naming.extract_cues([turns[0]], "it") == {}
    assert naming.extract_cues([turns[1]], "es") == {}


def test_cues_skip_very_short_candidates():
    turns = [a_turn("S", 0.0, 2.0, "Ciao Bo, come stai?")]
    assert naming.extract_cues(turns, "it") == {}


def test_cues_skip_lowercase_words_after_a_trigger():
    """The patterns run case-insensitively, so `[A-Z]` also matches a lowercase
    letter; the explicit capital check is what keeps ordinary words out."""
    turns = [
        a_turn("S", 0.0, 3.0, "Allora vediamo il documento insieme."),
        a_turn("S", 3.0, 6.0, "Non sono sicuro di aver capito."),
    ]
    assert naming.extract_cues(turns, "it") == {}


def test_cues_deduplicate_repeated_sentences():
    line = "Ciao Ludovica, ci sentiamo dopo."
    turns = [a_turn("S", 0.0, 2.0, line), a_turn("S", 2.5, 4.0, line)]
    assert naming.extract_cues(turns, "it")["S"] == [line]


def test_cues_shorten_a_very_long_turn_around_the_name():
    filler = "e poi abbiamo parlato del calendario delle riunioni per un bel po. "
    text = filler * 5 + "Ciao Ludovica, ci vediamo domani. " + filler * 5
    assert len(text) > 240
    cues = naming.extract_cues([a_turn("S", 0.0, 90.0, text)], "it")
    snippet = cues["S"][0]
    assert snippet.startswith("…") and snippet.endswith("…")
    assert "Ludovica" in snippet
    assert len(snippet) < len(text)


def test_cues_from_a_turn_without_a_speaker_are_filed_under_question_mark():
    turns = [{"speaker": None, "start": 0.0, "end": 3.0,
              "text": "Ciao Ludovica, tutto bene?"}]
    assert list(naming.extract_cues(turns, "it")) == ["?"]


def test_cues_tolerate_a_turn_without_text():
    turns = [{"speaker": "S", "start": 0.0, "end": 1.0}]
    assert naming.extract_cues(turns, "it") == {}


def test_cues_on_an_empty_transcript():
    assert naming.extract_cues([], "it") == {}


# --------------------------------------------------------------------------- #
# naming.build_profiles
# --------------------------------------------------------------------------- #

def two_speaker_transcript():
    return [
        a_turn("SPEAKER_00", 0.0, 5.0, "Buongiorno Marilena, cominciamo quando vuoi."),
        a_turn("SPEAKER_01", 5.5, 20.0,
             "Grazie. Mi chiamo Marilena e seguo il progetto da due anni ormai."),
        a_turn("SPEAKER_00", 20.5, 24.0, "Perfetto."),
    ]


def test_profiles_are_sorted_by_speech_time():
    profiles = naming.build_profiles(
        two_speaker_transcript(), {"SPEAKER_00": 8.5, "SPEAKER_01": 14.5}, "it")
    assert [p.label for p in profiles] == ["SPEAKER_01", "SPEAKER_00"]
    assert profiles[0].speech_seconds == 14.5


def test_profiles_count_turns_and_remember_when_the_voice_first_speaks():
    profiles = naming.build_profiles(
        two_speaker_transcript(), {"SPEAKER_00": 8.5, "SPEAKER_01": 14.5}, "it")
    by_label = {p.label: p for p in profiles}
    assert by_label["SPEAKER_00"].turn_count == 2
    assert by_label["SPEAKER_00"].first_seen == 0.0
    assert by_label["SPEAKER_01"].first_seen == 5.5


def test_profiles_default_to_zero_speech_when_the_diarizer_has_no_entry():
    profiles = naming.build_profiles(two_speaker_transcript(), {}, "it")
    assert all(p.speech_seconds == 0.0 for p in profiles)


def test_profiles_carry_no_identity_when_no_registry_match_is_given():
    """Nothing in the transcript may become a name on its own. The name Marilena is
    said out loud twice and must appear only as a cue to read."""
    profiles = naming.build_profiles(two_speaker_transcript(), {}, "it")
    for p in profiles:
        assert p.registry_name is None
        assert p.registry_candidate is None
        assert p.registry_score == 0.0
        assert p.registry_note == ""
    assert any("Marilena" in c for p in profiles for c in p.name_cues)


def test_profiles_copy_the_registry_verdict_verbatim():
    matches = {
        "SPEAKER_01": {"name": "Marilena", "candidate": None,
                       "score": 0.83, "reason": "accepted"},
        "SPEAKER_00": {"name": None, "candidate": "Tazio",
                       "score": 0.61, "reason": "could be Tazio (0.610)"},
    }
    profiles = naming.build_profiles(
        two_speaker_transcript(), {"SPEAKER_00": 8.5, "SPEAKER_01": 14.5},
        "it", matches)
    by_label = {p.label: p for p in profiles}
    assert by_label["SPEAKER_01"].registry_name == "Marilena"
    assert by_label["SPEAKER_01"].registry_candidate is None
    assert by_label["SPEAKER_00"].registry_name is None
    assert by_label["SPEAKER_00"].registry_candidate == "Tazio"
    assert by_label["SPEAKER_00"].registry_score == pytest.approx(0.61)


def test_profiles_keep_the_longest_turns_in_chronological_order():
    turns = [
        a_turn("S", 0.0, 2.0, "Sì."),
        a_turn("S", 5.0, 30.0, "Una risposta molto lunga che spiega tutto per bene."),
        a_turn("S", 40.0, 60.0, "Una risposta media di lunghezza."),
        a_turn("S", 70.0, 71.0, "No."),
    ]
    p = naming.build_profiles(turns, {"S": 48.0}, "it", n_samples=2)[0]
    assert len(p.samples) == 2
    assert p.samples[0].startswith("Una risposta molto lunga")
    assert p.samples[1].startswith("Una risposta media")


def test_profiles_truncate_a_very_long_sample():
    long_text = "parola " * 200
    p = naming.build_profiles([a_turn("S", 0.0, 300.0, long_text)], {"S": 300.0}, "it")[0]
    assert len(p.samples[0]) == 400


def test_profiles_cap_the_number_of_cues():
    turns = [a_turn("S", float(i), float(i) + 1.0, f"Ciao Ludovica, messaggio numero {i}.")
             for i in range(12)]
    p = naming.build_profiles(turns, {"S": 12.0}, "it")[0]
    assert len(p.name_cues) == 8


def test_profiles_group_speakerless_turns_under_one_label():
    turns = [
        {"speaker": None, "start": 0.0, "end": 2.0, "text": "Una frase."},
        {"speaker": None, "start": 3.0, "end": 5.0, "text": "Un'altra frase."},
    ]
    profiles = naming.build_profiles(turns, {}, "it")
    assert [p.label for p in profiles] == ["?"]
    assert profiles[0].turn_count == 2


def test_profiles_of_an_empty_transcript():
    assert naming.build_profiles([], {}, "it") == []


# --------------------------------------------------------------------------- #
# naming.dossier
# --------------------------------------------------------------------------- #

def test_dossier_never_states_a_name_it_was_not_given():
    """Marilena is spoken aloud twice in the transcript. The briefing may quote the
    sentences; it may not turn them into an attribution.

    The property is about attributions, not about how many lines the document has.
    An earlier version of this test asserted that the phrase "Voice registry" was
    absent, which stopped being the point the moment every voice started getting a
    line saying nobody identified it: that line names no one, so it cannot break
    what this test is for.
    """
    profiles = naming.build_profiles(two_speaker_transcript(), {}, "it")
    text = naming.dossier(profiles, language="it", title="colloquio")

    # Attributions are the bold ones, and no voice's block contains any.
    assert "matches **" not in text
    assert "resembles **" not in text
    assert all("**" not in block for block in sections(text).values())
    # Every line that carries the name is quoted material, never a statement.
    named = [line for line in text.splitlines() if "Marilena" in line]
    assert named
    assert all(line.lstrip().startswith("- «") for line in named)
    for p in profiles:
        assert f"## {p.label}" in text  # still identified by its diarizer label


def test_dossier_warns_that_a_spoken_name_is_usually_the_other_persons():
    profiles = naming.build_profiles(two_speaker_transcript(), {}, "it")
    text = naming.dossier(profiles, language="it", title="colloquio")
    assert "usually the name of the *other* one" in text


def test_dossier_says_plainly_when_nobody_was_identified():
    """The unidentified case has to be visible in the document, not implied by the
    absence of a line."""
    matches = {
        "SPEAKER_00": {"name": None, "candidate": None, "score": 0.0,
                       "reason": "empty registry"},
        "SPEAKER_01": {"name": None, "candidate": None, "score": 0.0,
                       "reason": "only 4s of speech: voice print too thin to trust"},
    }
    profiles = naming.build_profiles(two_speaker_transcript(), {}, "it", matches)
    text = naming.dossier(profiles, language="it", title="colloquio")

    assert text.count("Voice registry: no match") == 2
    assert "empty registry" in text
    assert "too thin to trust" in text
    assert "matches **" not in text


def sections(text: str) -> dict[str, str]:
    """The dossier split into one block per voice, keyed by its label."""
    out: dict[str, str] = {}
    label = None
    for line in text.splitlines():
        if line.startswith("## "):
            label = line[3:].strip()
            out[label] = ""
        elif label is not None:
            out[label] += line + "\n"
    return out


# Used to be xfailed: a voice with no entry in `matches` got no registry line at
# all, so the briefing left an empty space where its verdict belonged, right next
# to a voice reading "matches **Marilena**". Now every voice gets a line.
def test_dossier_describes_every_unidentified_voice_as_unidentified():
    """The reachable case: diarize.run (diarize.py:163) drops the zero vector
    pyannote fills in for a speaker it found no centroid for, and pipeline.identify
    only builds an entry for labels that have an embedding. That speaker reaches the
    briefing with nothing attached to it, and has to be described as such."""
    matches = {"SPEAKER_01": {"name": "Marilena", "candidate": None,
                              "score": 0.90, "reason": "accepted"}}
    profiles = naming.build_profiles(two_speaker_transcript(), {}, "it", matches)
    blocks = sections(naming.dossier(profiles, language="it", title="colloquio"))

    assert "matches **Marilena**" in blocks["SPEAKER_01"]
    unknown = blocks["SPEAKER_00"]
    assert "Voice registry: no verdict was recorded" in unknown
    assert "Nothing here says who it is." in unknown
    # And the distinction it exists to draw: never checked reads differently from
    # checked and found nothing.
    assert "no match" not in unknown


def test_dossier_claims_nothing_about_a_voice_with_no_registry_entry():
    """Saying nobody identified it is not the same as offering a guess: the block
    still contains no attribution."""
    matches = {"SPEAKER_01": {"name": "Marilena", "candidate": None,
                              "score": 0.90, "reason": "accepted"}}
    profiles = naming.build_profiles(two_speaker_transcript(), {}, "it", matches)
    blocks = sections(naming.dossier(profiles, language="it", title="colloquio"))

    unnamed = blocks["SPEAKER_00"]
    # The name Marilena is in this block, but only inside a quoted cue sentence:
    # never as an attribution, which the document writes in bold.
    assert "«Buongiorno Marilena" in unnamed
    assert "**" not in unnamed
    assert "matches" not in unnamed and "resembles" not in unnamed


def test_dossier_gives_every_voice_exactly_one_registry_line():
    """Whatever the registry had to say, each voice gets one verdict and one only.
    Two lines would mean the reader is shown a name and a doubt about it at once;
    none would mean a voice whose identity the document never mentions."""
    matches = {
        "SPEAKER_00": {"name": "Ottavia", "candidate": None,
                       "score": 0.88, "reason": "accepted"},
        "SPEAKER_01": {"name": None, "candidate": "Marilena",
                       "score": 0.60, "reason": "could be Marilena (0.600)"},
    }
    variants = [matches, {"SPEAKER_00": matches["SPEAKER_00"]}, {}]
    for m in variants:
        profiles = naming.build_profiles(two_speaker_transcript(), {}, "it", m)
        blocks = sections(naming.dossier(profiles, language="it", title="colloquio"))
        assert len(blocks) == 2
        for block in blocks.values():
            assert block.count("Voice registry") == 1


def test_dossier_marks_a_borderline_match_as_undecided():
    matches = {"SPEAKER_01": {"name": None, "candidate": "Marilena", "score": 0.61,
                              "reason": "could be Marilena (0.610)"}}
    profiles = naming.build_profiles(two_speaker_transcript(), {}, "it", matches)
    text = naming.dossier(profiles, language="it", title="colloquio")

    assert "resembles **Marilena**" in text
    assert "too low to decide" in text
    assert "Confirm" in text or "confirm" in text
    assert "matches **Marilena**" not in text


def test_dossier_states_an_accepted_match_with_its_score():
    matches = {"SPEAKER_01": {"name": "Marilena", "candidate": None,
                              "score": 0.912, "reason": "accepted"}}
    profiles = naming.build_profiles(two_speaker_transcript(), {}, "it", matches)
    text = naming.dossier(profiles, language="it", title="colloquio")
    assert "matches **Marilena**" in text
    assert "0.912" in text


def test_dossier_prefers_the_certain_match_over_the_other_two_lines():
    """A voice the registry is certain about gets the certain line and nothing else,
    so the reader is never shown a name and a doubt about it at the same time.

    Counted inside that voice's own block: every other voice has a registry line of
    its own now, so counting across the whole document measures nothing.
    """
    matches = {"SPEAKER_01": {"name": "Marilena", "candidate": "Ottavia",
                              "score": 0.90, "reason": "accepted"}}
    profiles = naming.build_profiles(two_speaker_transcript(), {}, "it", matches)
    text = naming.dossier(profiles, language="it", title="colloquio")

    identified = sections(text)["SPEAKER_01"]
    assert identified.count("Voice registry") == 1
    assert "matches **Marilena**" in identified
    assert "Ottavia" not in text          # the runner-up is never floated
    assert "too low to decide" not in identified


def test_dossier_header_reports_language_title_and_voice_count():
    profiles = naming.build_profiles(
        two_speaker_transcript(), {"SPEAKER_00": 8.5, "SPEAKER_01": 14.5}, "it")
    text = naming.dossier(profiles, language="it", title="riunione del lunedì")
    assert text.startswith("# Who is who: riunione del lunedì")
    assert "Language: it. Distinct voices found: 2." in text


def test_dossier_formats_minutes_seconds_and_the_first_timestamp():
    profiles = [naming.SpeakerProfile(label="SPEAKER_00", speech_seconds=125.0,
                                      turn_count=4, first_seen=65.0,
                                      samples=["una frase"])]
    text = naming.dossier(profiles, language="it", title="x")
    assert "Speech: 2m 05s across 4 turns" in text
    assert "first turn at 1:05" in text


def test_dossier_of_an_empty_transcript_is_still_a_document():
    text = naming.dossier([], language="it", title="vuoto")
    assert "Distinct voices found: 0." in text
    assert "##" not in text


# --------------------------------------------------------------------------- #
# naming.apply_names
# --------------------------------------------------------------------------- #

def test_apply_names_takes_the_certain_registry_matches():
    profiles = [
        naming.SpeakerProfile("SPEAKER_00", 10.0, 2, 0.0, registry_name="Marilena"),
        naming.SpeakerProfile("SPEAKER_01", 8.0, 2, 1.0),
    ]
    assert naming.apply_names({}, profiles) == {"SPEAKER_00": "Marilena"}


def test_apply_names_ignores_a_mere_candidate():
    """A borderline resemblance is a suggestion for a human, never an attribution.
    Letting it through is how a wrong name lands in a summary as if it were a fact."""
    profiles = [
        naming.SpeakerProfile("SPEAKER_00", 10.0, 2, 0.0,
                              registry_candidate="Marilena", registry_score=0.62),
    ]
    assert naming.apply_names({}, profiles) == {}


def test_apply_names_ignores_the_cues_found_in_the_text():
    """Names spoken out loud reach the briefing and stop there."""
    profiles = naming.build_profiles(two_speaker_transcript(), {}, "it")
    assert any(p.name_cues for p in profiles)
    assert naming.apply_names({}, profiles) == {}


def test_apply_names_lets_the_human_decision_win():
    profiles = [naming.SpeakerProfile("SPEAKER_00", 10.0, 2, 0.0,
                                      registry_name="Marilena")]
    assert naming.apply_names({"SPEAKER_00": "Ottavia"}, profiles) == {
        "SPEAKER_00": "Ottavia"}


def test_apply_names_drops_empty_manual_entries():
    """An empty box in the naming form means "I do not know", and must not erase a
    name the registry was certain about."""
    profiles = [naming.SpeakerProfile("SPEAKER_00", 10.0, 2, 0.0,
                                      registry_name="Marilena")]
    assert naming.apply_names({"SPEAKER_00": "", "SPEAKER_01": None}, profiles) == {
        "SPEAKER_00": "Marilena"}


def test_apply_names_keeps_a_manual_name_for_a_label_that_is_not_in_this_run():
    """Current behaviour, recorded rather than endorsed.

    The mapping is not filtered against the profiles it is handed, so a name decided
    for SPEAKER_02 survives into a run whose diarization produced no SPEAKER_02.
    Harmless here, and the seed of a wrong name once the caller stores the result:
    pipeline.run feeds this back into state["names"] on every pass, and
    _drop_stale_cache (pipeline.py:159-162) does not clear "names" when the source
    audio changes, so the entry is still there to be reapplied to whoever SPEAKER_02
    turns out to be next time.
    """
    assert naming.apply_names({"SPEAKER_09": "Tazio"}, []) == {"SPEAKER_09": "Tazio"}


def test_apply_names_on_nothing_at_all():
    assert naming.apply_names({}, []) == {}


# --------------------------------------------------------------------------- #
# diarize.to_turns
# --------------------------------------------------------------------------- #

def seg(speaker, start, end, text, conf=1.0):
    return {"speaker": speaker, "start": start, "end": end,
            "text": text, "speaker_confidence": conf}


def test_to_turns_merges_consecutive_segments_of_one_speaker():
    """whisper cuts every ~30s regardless of who is talking, so its segments are
    fragments, not turns."""
    segments = [
        seg("SPEAKER_00", 0.0, 4.0, " La prima parte del discorso."),
        seg("SPEAKER_00", 4.5, 9.0, " E la seconda, senza cambiare voce."),
    ]
    turns = diarize.to_turns(segments)
    assert len(turns) == 1
    assert turns[0]["text"] == "La prima parte del discorso. E la seconda, senza cambiare voce."
    assert turns[0]["start"] == 0.0
    assert turns[0]["end"] == 9.0


def test_to_turns_splits_on_a_change_of_speaker_even_with_no_gap():
    segments = [
        seg("SPEAKER_00", 0.0, 4.0, "Domanda."),
        seg("SPEAKER_01", 4.0, 8.0, "Risposta."),
    ]
    turns = diarize.to_turns(segments)
    assert [t["speaker"] for t in turns] == ["SPEAKER_00", "SPEAKER_01"]


def test_to_turns_merges_at_exactly_max_gap():
    segments = [
        seg("S", 0.0, 10.0, "Prima."),
        seg("S", 12.0, 14.0, "Dopo."),
    ]
    turns = diarize.to_turns(segments, max_gap=2.0)
    assert len(turns) == 1
    assert turns[0]["end"] == 14.0


def test_to_turns_splits_just_over_max_gap():
    segments = [
        seg("S", 0.0, 10.0, "Prima."),
        seg("S", 12.5, 14.0, "Dopo una pausa lunga."),
    ]
    turns = diarize.to_turns(segments, max_gap=2.0)
    assert len(turns) == 2
    assert turns[0]["end"] == 10.0
    assert turns[1]["start"] == 12.5


def test_to_turns_max_gap_is_configurable():
    segments = [seg("S", 0.0, 10.0, "Prima."), seg("S", 20.0, 22.0, "Molto dopo.")]
    assert len(diarize.to_turns(segments, max_gap=2.0)) == 2
    assert len(diarize.to_turns(segments, max_gap=30.0)) == 1


def test_to_turns_merges_a_long_run_and_keeps_the_outer_boundaries():
    segments = [seg("S", float(i * 5), float(i * 5 + 4), f"frammento {i}")
                for i in range(6)]
    turns = diarize.to_turns(segments)
    assert len(turns) == 1
    assert turns[0]["start"] == 0.0
    assert turns[0]["end"] == 29.0
    assert turns[0]["text"].split() == [w for i in range(6)
                                        for w in ("frammento", str(i))]


def test_to_turns_alternating_speakers_produce_one_turn_each():
    segments = [seg("A" if i % 2 == 0 else "B", float(i), float(i) + 0.9, f"t{i}")
                for i in range(6)]
    turns = diarize.to_turns(segments)
    assert [t["speaker"] for t in turns] == ["A", "B", "A", "B", "A", "B"]


def test_to_turns_speaker_returning_after_someone_else_starts_a_new_turn():
    segments = [
        seg("A", 0.0, 2.0, "Uno."),
        seg("B", 2.1, 3.0, "Due."),
        seg("A", 3.1, 5.0, "Tre."),
    ]
    turns = diarize.to_turns(segments)
    assert len(turns) == 3
    assert turns[2]["text"] == "Tre."


def test_to_turns_single_segment():
    turns = diarize.to_turns([seg("S", 1.5, 4.0, "  Una sola frase.  ", 0.42)])
    assert turns == [{"speaker": "S", "start": 1.5, "end": 4.0,
                      "text": "Una sola frase.", "confidence": 0.42}]


def test_to_turns_empty_input():
    assert diarize.to_turns([]) == []


def test_to_turns_drops_empty_and_blank_segments():
    segments = [
        seg("S", 0.0, 1.0, ""),
        seg("S", 1.0, 2.0, "   "),
        seg("S", 2.0, 3.0, "\n\t "),
    ]
    assert diarize.to_turns(segments) == []


def test_to_turns_survives_a_segment_with_no_text_key():
    segments = [{"speaker": "S", "start": 0.0, "end": 1.0},
                seg("S", 1.0, 3.0, "Con testo.")]
    turns = diarize.to_turns(segments)
    assert len(turns) == 1
    assert turns[0]["text"] == "Con testo."


def test_to_turns_treats_a_none_text_as_empty():
    segments = [{"speaker": "S", "start": 0.0, "end": 1.0, "text": None}]
    assert diarize.to_turns(segments) == []


def test_to_turns_a_blank_segment_does_not_break_the_merge_around_it():
    segments = [
        seg("S", 0.0, 4.0, "Prima parte."),
        seg("S", 4.1, 4.3, "   "),
        seg("S", 4.5, 8.0, "Seconda parte."),
    ]
    turns = diarize.to_turns(segments)
    assert len(turns) == 1
    assert turns[0]["text"] == "Prima parte. Seconda parte."
    assert turns[0]["end"] == 8.0


def test_to_turns_keeps_segments_with_no_speaker():
    """A word overlapping no diarization turn is left without a speaker rather than
    handed to the nearest one. The text still has to reach the transcript."""
    segments = [seg(None, 0.0, 3.0, "Frase senza voce assegnata.", 0.0)]
    turns = diarize.to_turns(segments)
    assert len(turns) == 1
    assert turns[0]["speaker"] is None
    assert turns[0]["text"] == "Frase senza voce assegnata."


def test_to_turns_merges_consecutive_speakerless_segments():
    segments = [seg(None, 0.0, 3.0, "Prima."), seg(None, 3.2, 6.0, "Seconda.")]
    turns = diarize.to_turns(segments)
    assert len(turns) == 1
    assert turns[0]["speaker"] is None
    assert turns[0]["text"] == "Prima. Seconda."


def test_to_turns_does_not_merge_a_speakerless_segment_into_a_named_one():
    segments = [
        seg("SPEAKER_00", 0.0, 3.0, "Detto da qualcuno."),
        seg(None, 3.2, 6.0, "Non attribuito."),
        seg("SPEAKER_00", 6.2, 9.0, "Di nuovo attribuito."),
    ]
    turns = diarize.to_turns(segments)
    assert [t["speaker"] for t in turns] == ["SPEAKER_00", None, "SPEAKER_00"]


def test_to_turns_confidence_is_the_minimum_not_the_mean():
    """The weakest segment sets the confidence for the merged turn. Averaging would
    let one solid minute bury the doubtful sentence inside it, and the doubtful
    sentence is the whole reason for carrying the number."""
    segments = [
        seg("S", 0.0, 30.0, "Un minuto solido di parlato.", 0.98),
        seg("S", 30.5, 31.5, "una frase dubbia", 0.11),
        seg("S", 32.0, 60.0, "e poi di nuovo solido.", 0.97),
    ]
    turns = diarize.to_turns(segments)
    assert len(turns) == 1
    assert turns[0]["confidence"] == pytest.approx(0.11)
    mean = (0.98 + 0.11 + 0.97) / 3
    assert turns[0]["confidence"] < mean


def test_to_turns_confidence_minimum_holds_whatever_the_order():
    weak_first = [seg("S", 0.0, 2.0, "a", 0.2), seg("S", 2.5, 4.0, "b", 0.9)]
    weak_last = [seg("S", 0.0, 2.0, "a", 0.9), seg("S", 2.5, 4.0, "b", 0.2)]
    assert diarize.to_turns(weak_first)[0]["confidence"] == pytest.approx(0.2)
    assert diarize.to_turns(weak_last)[0]["confidence"] == pytest.approx(0.2)


def test_to_turns_confidence_is_not_dragged_down_across_a_boundary():
    """A doubtful segment poisons its own turn and no other."""
    segments = [
        seg("A", 0.0, 3.0, "Sicuro.", 0.95),
        seg("B", 3.2, 6.0, "Dubbio.", 0.05),
        seg("A", 6.2, 9.0, "Sicuro di nuovo.", 0.93),
    ]
    turns = diarize.to_turns(segments)
    assert [round(t["confidence"], 2) for t in turns] == [0.95, 0.05, 0.93]


def test_to_turns_confidence_defaults_to_one_when_absent():
    segments = [{"speaker": "S", "start": 0.0, "end": 2.0, "text": "Senza punteggio."}]
    assert diarize.to_turns(segments)[0]["confidence"] == 1.0


def test_to_turns_missing_timestamps_default_to_zero():
    turns = diarize.to_turns([{"speaker": "S", "text": "Senza tempi."}])
    assert turns[0]["start"] == 0.0 and turns[0]["end"] == 0.0


def test_to_turns_overlapping_segments_of_one_speaker_still_merge():
    segments = [seg("S", 0.0, 5.0, "Prima."), seg("S", 4.0, 8.0, "Sovrapposta.")]
    turns = diarize.to_turns(segments)
    assert len(turns) == 1
    assert turns[0]["end"] == 8.0


def test_to_turns_sorts_segments_that_arrive_out_of_order():
    """The aligner hands them over in order today. A caller that does not used to
    get back a turn whose end came before its start, and every downstream timestamp
    inherited it."""
    segments = [
        seg("S", 10.0, 12.0, "detto dopo"),
        seg("S", 0.0, 2.0, "detto prima"),
    ]
    turns = diarize.to_turns(segments)
    assert len(turns) == 2
    assert [t["text"] for t in turns] == ["detto prima", "detto dopo"]
    assert all(t["end"] >= t["start"] for t in turns)


def test_to_turns_sorting_makes_the_merge_follow_the_clock():
    """Two fragments of one speaker three seconds apart in the file must not merge
    just because they were handed over adjacent in the list."""
    segments = [
        seg("S", 7.0, 9.0, "terza"),
        seg("S", 0.0, 2.0, "prima"),
        seg("S", 2.5, 4.0, "seconda"),
    ]
    turns = diarize.to_turns(segments, max_gap=2.0)
    assert [t["text"] for t in turns] == ["prima seconda", "terza"]
    assert turns[0]["start"] == 0.0 and turns[0]["end"] == 4.0
    assert turns[1]["start"] == 7.0


def test_to_turns_sorted_input_is_left_alone():
    """Sorting is stable, so segments already in order come out exactly as before,
    ties included."""
    segments = [
        seg("A", 0.0, 2.0, "uno"),
        seg("B", 2.0, 4.0, "due"),
        seg("B", 4.0, 6.0, "tre"),
    ]
    assert diarize.to_turns(segments) == [
        {"speaker": "A", "start": 0.0, "end": 2.0, "text": "uno", "confidence": 1.0},
        {"speaker": "B", "start": 2.0, "end": 6.0, "text": "due tre", "confidence": 1.0},
    ]


def test_to_turns_returns_the_five_expected_keys():
    turns = diarize.to_turns([seg("S", 0.0, 1.0, "Frase.")])
    assert set(turns[0]) == {"speaker", "start", "end", "text", "confidence"}


def test_to_turns_does_not_mutate_the_segments_it_was_given():
    segments = [seg("S", 0.0, 4.0, "Prima."), seg("S", 4.2, 8.0, "Seconda.")]
    before = [dict(s) for s in segments]
    diarize.to_turns(segments)
    assert segments == before


def test_to_turns_numbers_come_out_as_floats():
    turns = diarize.to_turns([{"speaker": "S", "start": 0, "end": 4,
                               "text": "Interi.", "speaker_confidence": 1}])
    assert isinstance(turns[0]["start"], float)
    assert isinstance(turns[0]["end"], float)
    assert isinstance(turns[0]["confidence"], float)


def test_to_turns_feeds_build_profiles_without_translation():
    """The two modules meet here in pipeline.run, so the shapes have to line up."""
    segments = [
        seg("SPEAKER_00", 0.0, 4.0, "Buongiorno Marilena, iniziamo?"),
        seg("SPEAKER_01", 4.5, 9.0, "Sì. Mi chiamo Marilena."),
    ]
    turns = diarize.to_turns(segments)
    profiles = naming.build_profiles(turns, {"SPEAKER_00": 4.0, "SPEAKER_01": 4.5}, "it")
    assert {p.label for p in profiles} == {"SPEAKER_00", "SPEAKER_01"}
    assert all(p.name_cues for p in profiles)


# --------------------------------------------------------------------------- #
# diarize._unpack: one code path against three pyannote return shapes
# --------------------------------------------------------------------------- #

class FakeAnnotation:
    """Just enough of a pyannote Annotation for _to_turns and _unpack."""

    def __init__(self, tracks):
        self._tracks = list(tracks)   # (start, end, label)

    def itertracks(self, yield_label=False):
        for i, (start, end, label) in enumerate(self._tracks):
            segment = SimpleNamespace(start=start, end=end)
            yield (segment, f"track{i}", label) if yield_label else (segment, f"track{i}")

    def labels(self):
        return sorted({label for _, _, label in self._tracks})


class FakeDiarizeOutput:
    """pyannote 4: a dataclass with named fields."""

    def __init__(self, annotation, embeddings=None, exclusive=None):
        self.speaker_diarization = annotation
        self.speaker_embeddings = embeddings
        self.exclusive_speaker_diarization = exclusive


def an_annotation():
    return FakeAnnotation([(0.0, 3.5, "SPEAKER_00"),
                           (3.6, 8.0, "SPEAKER_01"),
                           (8.1, 12.0, "SPEAKER_00")])


def some_centroids():
    return np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)


def test_unpack_pyannote4_dataclass():
    ann, centroids = an_annotation(), some_centroids()
    exclusive = FakeAnnotation([(0.0, 3.4, "SPEAKER_00")])
    got_ann, got_emb, got_exc = diarize._unpack(
        FakeDiarizeOutput(ann, centroids, exclusive))
    assert got_ann is ann
    assert got_emb is centroids
    assert got_exc is exclusive


def test_unpack_pyannote3_tuple():
    ann, centroids = an_annotation(), some_centroids()
    got_ann, got_emb, got_exc = diarize._unpack((ann, centroids))
    assert got_ann is ann
    assert got_emb is centroids
    assert got_exc is None


def test_unpack_bare_annotation():
    ann = an_annotation()
    assert diarize._unpack(ann) == (ann, None, None)


def test_unpack_the_three_shapes_agree():
    """The whole point of the function: whichever pyannote is installed, the rest of
    the module sees one thing."""
    ann, centroids = an_annotation(), some_centroids()
    shapes = [
        FakeDiarizeOutput(ann, centroids, None),   # pyannote 4
        (ann, centroids),                          # pyannote 3, return_embeddings
    ]
    results = [diarize._unpack(s) for s in shapes]
    assert results[0] == results[1] == (ann, centroids, None)

    ann_only = diarize._unpack(ann)
    assert ann_only[0] is ann and ann_only[1] is None and ann_only[2] is None


def test_unpack_the_three_shapes_give_the_same_turns():
    ann, centroids = an_annotation(), some_centroids()
    turns = [diarize._to_turns(diarize._unpack(shape)[0])
             for shape in (FakeDiarizeOutput(ann, centroids, None),
                           (ann, centroids),
                           ann)]
    assert turns[0] == turns[1] == turns[2]
    assert [t.speaker for t in turns[0]] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]


def test_unpack_dataclass_without_embeddings_or_exclusive():
    """pyannote 4 asked for a diarization it could not embed: the fields are absent
    or empty, and the caller has to receive None rather than an exception."""
    ann = an_annotation()
    bare = SimpleNamespace(speaker_diarization=ann)
    assert diarize._unpack(bare) == (ann, None, None)


def test_unpack_prefers_the_named_field_over_tuple_unpacking():
    """A dataclass that also behaves like a tuple must take the pyannote 4 path,
    otherwise the exclusive segmentation is silently dropped."""

    class TupleLikeOutput(tuple):
        def __new__(cls, annotation, embeddings, exclusive):
            self = super().__new__(cls, (annotation, embeddings, exclusive))
            self.speaker_diarization = annotation
            self.speaker_embeddings = embeddings
            self.exclusive_speaker_diarization = exclusive
            return self

    ann, centroids = an_annotation(), some_centroids()
    exclusive = FakeAnnotation([(0.0, 3.4, "SPEAKER_00")])
    got = diarize._unpack(TupleLikeOutput(ann, centroids, exclusive))
    assert got == (ann, centroids, exclusive)


def test_unpack_leaves_the_centroids_untouched():
    """They become voice prints in the registry, so no reshaping or casting may
    happen on the way out."""
    centroids = some_centroids()
    _, got, _ = diarize._unpack((an_annotation(), centroids))
    assert got is centroids
    assert got.dtype == np.float32
    assert got.shape == (2, 3)


# --------------------------------------------------------------------------- #
# diarize._to_turns, the other half of the unpacking path
# --------------------------------------------------------------------------- #

def test_to_turns_from_annotation_sorts_by_start():
    ann = FakeAnnotation([(8.0, 9.0, "SPEAKER_01"),
                          (0.0, 3.0, "SPEAKER_00"),
                          (3.5, 7.0, "SPEAKER_01")])
    turns = diarize._to_turns(ann)
    assert [t.start for t in turns] == [0.0, 3.5, 8.0]
    assert turns[0].duration == pytest.approx(3.0)


def test_to_turns_from_an_empty_annotation():
    assert diarize._to_turns(FakeAnnotation([])) == []


def test_to_turns_from_annotation_casts_to_float_and_str():
    ann = FakeAnnotation([(np.float64(1.0), np.float64(2.0), 0)])
    turn_ = diarize._to_turns(ann)[0]
    assert type(turn_.start) is float
    assert type(turn_.end) is float
    assert turn_.speaker == "0"


def test_diarization_speech_time_and_longest_turns():
    dia = diarize.Diarization(turns=diarize._to_turns(an_annotation()))
    assert dia.speakers() == ["SPEAKER_00", "SPEAKER_01"]
    assert dia.speech_time()["SPEAKER_00"] == pytest.approx(3.5 + 3.9)
    longest = dia.longest_turns("SPEAKER_00", n=1)
    assert len(longest) == 1
    assert longest[0].start == 8.1
