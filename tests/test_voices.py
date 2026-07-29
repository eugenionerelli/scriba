"""Tests for `scriba.voices`: the voice print registry and its matching policy.

Everything here is synthetic. No audio, no models, no network: embeddings are unit
vectors built by hand so that every cosine similarity in a test is a number chosen in
advance, not something that came out of a neural network and has to be taken on faith.

The construction: fix a probe vector on the first axis, e0. Then

    v(s, k) = s * e0 + sqrt(1 - s**2) * e_k        (k != 0)

is a unit vector whose cosine with the probe is exactly `s`, and picking a different
`k` for each person makes their similarities to the probe independent of each other.
That is what lets the three decision zones and the runner-up margin be tested at the
boundary instead of "somewhere in the right neighbourhood".

All state is redirected into tmp_path by an autouse fixture. The real ~/.scriba is
never read and never written.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from scriba import voices as V

DIM = 256  # wespeaker-voxceleb-resnet34-LM, the size the module documents


# --------------------------------------------------------------------------- #
# isolation
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def isolated_voices_dir(tmp_path, monkeypatch):
    """Point every path voices.py touches at tmp_path.

    REGISTRY_PATH and EMB_PATH are module globals computed at import time from
    config.VOICES_DIR, so setting SCRIBA_HOME after the import would be too late.
    Rebinding the names inside the voices module is what actually works, and it is
    what the module reads at call time.
    """
    home = tmp_path / "scriba-home"
    vdir = home / "voices"
    monkeypatch.setenv("SCRIBA_HOME", str(home))
    monkeypatch.setattr(V, "VOICES_DIR", vdir)
    monkeypatch.setattr(V, "REGISTRY_PATH", vdir / "registry.json")
    monkeypatch.setattr(V, "EMB_PATH", vdir / "embeddings.npy")

    # A test suite that writes into the user's own registry would be worse than no
    # test suite, so this is checked rather than assumed.
    for p in (V.VOICES_DIR, V.REGISTRY_PATH, V.EMB_PATH):
        assert Path(p).is_relative_to(tmp_path)
    yield vdir


@pytest.fixture
def reg():
    return V.VoiceRegistry()


# --------------------------------------------------------------------------- #
# embedding helpers
# --------------------------------------------------------------------------- #

def vec(similarity: float, axis: int = 1) -> np.ndarray:
    """A unit vector whose cosine with PROBE is exactly `similarity`."""
    assert axis != 0, "axis 0 is the probe direction"
    v = np.zeros(DIM, dtype=np.float64)
    v[0] = similarity
    v[axis] = math.sqrt(max(0.0, 1.0 - similarity * similarity))
    return v.astype(np.float32)


PROBE = vec(1.0)


def score_of(registry: V.VoiceRegistry, embedding: np.ndarray) -> V.Match:
    """A match with every gate wide open: used to read the raw scores."""
    return registry.match(embedding, threshold=0.0, suggest_threshold=0.0, margin=0.0)


# --------------------------------------------------------------------------- #
# the maths underneath
# --------------------------------------------------------------------------- #

def test_l2norm_returns_unit_vectors_and_survives_a_zero_vector():
    x = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    n = V.l2norm(x)
    assert float(np.linalg.norm(n[0])) == pytest.approx(1.0, abs=1e-6)
    # A silent voice print must not become NaN and poison every later comparison.
    assert np.all(np.isfinite(n[1]))
    assert float(np.linalg.norm(n[1])) == pytest.approx(0.0, abs=1e-6)


def test_cosine_matches_the_similarity_we_built_in():
    for target in (1.0, 0.92, 0.75, 0.55, 0.2, 0.0):
        got = float(V.cosine(PROBE, vec(target)[None, :]).reshape(-1)[0])
        assert got == pytest.approx(target, abs=1e-6)


def test_cosine_against_a_batch_returns_one_score_per_row():
    bank = np.stack([vec(0.9, 1), vec(0.5, 2), vec(0.1, 3)])
    got = V.cosine(PROBE, bank).reshape(-1)
    assert got.shape == (3,)
    assert list(got) == pytest.approx([0.9, 0.5, 0.1], abs=1e-6)


# --------------------------------------------------------------------------- #
# enrolment
# --------------------------------------------------------------------------- #

def test_enroll_creates_the_person_and_one_row(reg):
    p = reg.enroll("Alba Verzieri", vec(0.9), source="lunedi.m4a", note="collega")
    assert p.name == "Alba Verzieri"
    assert p.rows == [0]
    assert p.sources == ["lunedi.m4a"]
    assert p.note == "collega"
    assert reg.emb.shape == (1, DIM)
    assert reg.by_name("Alba Verzieri") is p


def test_enroll_trims_the_name_and_by_name_ignores_case_and_padding(reg):
    p = reg.enroll("  Alba Verzieri  ", vec(0.9))
    assert p.name == "Alba Verzieri"
    assert reg.by_name("alba verzieri") is p
    assert reg.by_name("  ALBA VERZIERI ") is p
    assert reg.by_name("Bruno Meltrame") is None


def test_enroll_rejects_a_non_finite_embedding(reg):
    bad = vec(0.9)
    bad[7] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        reg.enroll("Alba Verzieri", bad)
    worse = vec(0.9)
    worse[7] = np.inf
    with pytest.raises(ValueError):
        reg.enroll("Alba Verzieri", worse)
    assert reg.people == {}
    assert reg.emb.shape[0] == 0


def test_enroll_rejects_a_different_embedding_dimension(reg):
    reg.enroll("Alba Verzieri", vec(0.9))
    with pytest.raises(ValueError, match="embedding model"):
        reg.enroll("Bruno Meltrame", np.ones(128, dtype=np.float32))
    # The rejection must leave nothing half-written behind.
    assert reg.emb.shape == (1, DIM)
    assert reg.by_name("Bruno Meltrame") is None


def test_the_same_person_from_two_recordings_keeps_both_prints(reg):
    """Different room, different mic: both prints stay, neither overwrites the other."""
    a = vec(0.9, axis=1)      # one afternoon
    b = vec(0.4, axis=2)      # a phone call, same person, very different acoustics
    p1 = reg.enroll("Alba Verzieri", a, source="ufficio.m4a")
    p2 = reg.enroll("Alba Verzieri", b, source="telefono.m4a")

    assert p1 is p2, "the second recording must attach to the same person"
    assert len(reg.people) == 1
    assert p1.rows == [0, 1], "both voice prints kept, nothing overwritten"
    assert reg.emb.shape == (2, DIM)
    assert p1.sources == ["ufficio.m4a", "telefono.m4a"]
    stored = reg.vectors_of(p1)
    assert np.allclose(stored[0], a)
    assert np.allclose(stored[1], b)


def test_re_enrolling_the_identical_print_does_not_pile_up_copies(reg):
    p = reg.enroll("Alba Verzieri", vec(0.9), source="lunedi.m4a")
    again = reg.enroll("Alba Verzieri", vec(0.9), source="lunedi.m4a")
    assert again is p
    assert p.rows == [0]
    assert reg.emb.shape == (1, DIM)
    assert p.sources == ["lunedi.m4a"], "the same source is not appended twice"


def test_a_near_identical_print_is_deduplicated_but_a_merely_similar_one_is_kept(reg):
    p = reg.enroll("Alba Verzieri", vec(1.0))
    # Above the 0.999 guard: the same audio re-processed, dropped.
    reg.enroll("Alba Verzieri", vec(0.9999, axis=4))
    assert p.rows == [0]
    # Below it: a genuinely different recording of the same person, kept.
    reg.enroll("Alba Verzieri", vec(0.99, axis=5))
    assert p.rows == [0, 1]


def test_deduplicated_enrolment_still_records_the_new_source(reg):
    p = reg.enroll("Alba Verzieri", vec(0.9), source="lunedi.m4a")
    reg.enroll("Alba Verzieri", vec(0.9), source="martedi.m4a")
    assert p.rows == [0]
    assert p.sources == ["lunedi.m4a", "martedi.m4a"]


# --------------------------------------------------------------------------- #
# aliases
# --------------------------------------------------------------------------- #

def test_an_alias_resolves_to_the_person(reg):
    p = reg.enroll("Alba Verzieri", vec(0.9), aliases=["Albi", "la Verzieri"])
    assert reg.by_name("Albi") is p
    assert reg.by_name("ALBI") is p
    assert reg.by_name("la verzieri") is p


def test_enrolling_under_an_alias_adds_a_print_instead_of_a_second_person(reg):
    p = reg.enroll("Alba Verzieri", vec(0.9), aliases=["Albi"])
    same = reg.enroll("Albi", vec(0.4, axis=2), source="cena.m4a")
    assert same is p
    assert len(reg.people) == 1, "an alias must not spawn a duplicate person"
    assert p.rows == [0, 1]
    assert p.name == "Alba Verzieri", "the canonical name is not replaced by the alias"


# Regression: this used to fail.
def test_enroll_can_add_an_alias_to_someone_already_in_the_registry(reg):
    reg.enroll("Alba Verzieri", vec(0.9))
    reg.enroll("Alba Verzieri", vec(0.4, axis=2), aliases=["Albi"])
    assert reg.by_name("Albi") is not None


# --------------------------------------------------------------------------- #
# matching: the empty and degenerate registries
# --------------------------------------------------------------------------- #

def test_an_empty_registry_decides_nothing(reg):
    m = reg.match(PROBE)
    assert m.accepted is False
    assert m.person is None
    assert m.candidate is None, "no name, not even as a suggestion"
    assert m.score == 0.0
    assert m.runner_up is None
    assert m.runner_up_score == 0.0
    assert "empty registry" in m.reason


def test_a_person_with_no_voice_prints_is_skipped(reg):
    orphan = V.Person(id="deadbeef0001", name="Bruno Meltrame")
    reg.people[orphan.id] = orphan
    m = reg.match(PROBE)
    assert m.accepted is False
    assert m.candidate is None
    assert "no voice prints" in m.reason


def test_a_non_finite_probe_is_refused_rather_than_matched(reg):
    reg.enroll("Alba Verzieri", vec(0.99))
    bad = PROBE.copy()
    bad[3] = np.nan
    m = reg.match(bad)
    assert m.accepted is False
    assert m.candidate is None
    assert m.reason == "invalid embedding"


def test_a_silent_probe_scores_zero_and_is_treated_as_a_new_voice(reg):
    reg.enroll("Alba Verzieri", vec(0.99))
    m = reg.match(np.zeros(DIM, dtype=np.float32))
    assert m.accepted is False
    assert m.candidate is None
    assert m.score == pytest.approx(0.0, abs=1e-6)
    assert "new voice" in m.reason


# --------------------------------------------------------------------------- #
# matching: the three zones
# --------------------------------------------------------------------------- #

def test_above_the_match_threshold_the_name_is_applied(reg):
    p = reg.enroll("Alba Verzieri", vec(0.92))
    m = reg.match(PROBE, threshold=0.75, suggest_threshold=0.55, margin=0.05)
    assert m.accepted is True
    assert m.person is p
    assert m.score == pytest.approx(0.92, abs=1e-6)
    assert m.reason == "accepted"


def test_between_the_thresholds_it_only_suggests_and_never_applies_the_name(reg):
    """The zone that exists so a wrong name cannot arrive in silence."""
    p = reg.enroll("Alba Verzieri", vec(0.62))
    m = reg.match(PROBE, threshold=0.75, suggest_threshold=0.55, margin=0.05)
    assert m.accepted is False
    assert m.person is None, "a suggestion must not be applied"
    assert m.candidate is p, "but it is handed to the user to confirm"
    assert m.score == pytest.approx(0.62, abs=1e-6)
    assert "Alba Verzieri" in m.reason and "confirm" in m.reason


def test_below_the_suggest_threshold_it_is_a_new_voice(reg):
    reg.enroll("Alba Verzieri", vec(0.30))
    m = reg.match(PROBE, threshold=0.75, suggest_threshold=0.55, margin=0.05)
    assert m.accepted is False
    assert m.person is None
    assert m.candidate is None, "too far to be worth even suggesting"
    assert "new voice" in m.reason


def test_the_match_threshold_is_inclusive_at_its_exact_value(reg):
    reg.enroll("Alba Verzieri", vec(0.8))
    s = score_of(reg, PROBE).score
    at = reg.match(PROBE, threshold=s, suggest_threshold=0.1, margin=0.0)
    assert at.accepted is True, "'at or above the threshold' includes 'at'"
    above = reg.match(PROBE, threshold=math.nextafter(s, 1.0),
                      suggest_threshold=0.1, margin=0.0)
    assert above.accepted is False
    assert above.candidate is not None, "a hair under the bar is a suggestion"


def test_the_suggest_threshold_is_inclusive_at_its_exact_value(reg):
    reg.enroll("Alba Verzieri", vec(0.6))
    s = score_of(reg, PROBE).score
    at = reg.match(PROBE, threshold=0.99, suggest_threshold=s, margin=0.0)
    assert at.candidate is not None, "at the suggest threshold, still a suggestion"
    below = reg.match(PROBE, threshold=0.99,
                      suggest_threshold=math.nextafter(s, 1.0), margin=0.0)
    assert below.candidate is None, "a hair under it, a new voice"
    assert "new voice" in below.reason


# --------------------------------------------------------------------------- #
# matching: the runner-up margin
# --------------------------------------------------------------------------- #

def test_two_close_candidates_above_the_threshold_decide_nothing(reg):
    """The property that stops a confident wrong name."""
    alba = reg.enroll("Alba Verzieri", vec(0.90, axis=1))
    reg.enroll("Bruno Meltrame", vec(0.88, axis=2))
    m = reg.match(PROBE, threshold=0.75, suggest_threshold=0.55, margin=0.05)
    assert m.accepted is False, "both are over the bar, so neither may be applied"
    assert m.person is None
    assert m.candidate is alba
    assert m.runner_up == "Bruno Meltrame"
    assert m.runner_up_score == pytest.approx(0.88, abs=1e-6)
    assert "ambiguous" in m.reason
    assert "Alba Verzieri" in m.reason and "Bruno Meltrame" in m.reason


def test_two_identical_voice_prints_under_two_names_decide_nothing(reg):
    """The worst case: the registry cannot tell them apart at all."""
    reg.enroll("Alba Verzieri", vec(1.0))
    reg.enroll("Bruno Meltrame", vec(1.0))
    m = reg.match(PROBE, threshold=0.75, suggest_threshold=0.55, margin=0.05)
    assert m.accepted is False
    assert m.score == pytest.approx(1.0, abs=1e-6)
    assert m.runner_up_score == pytest.approx(1.0, abs=1e-6)
    assert {m.candidate.name, m.runner_up} == {"Alba Verzieri", "Bruno Meltrame"}
    assert "ambiguous" in m.reason


def test_a_distant_runner_up_does_not_block_the_match(reg):
    alba = reg.enroll("Alba Verzieri", vec(0.95, axis=1))
    reg.enroll("Bruno Meltrame", vec(0.30, axis=2))
    m = reg.match(PROBE, threshold=0.75, suggest_threshold=0.55, margin=0.05)
    assert m.accepted is True
    assert m.person is alba
    assert m.runner_up == "Bruno Meltrame"
    assert m.runner_up_score == pytest.approx(0.30, abs=1e-6)


def test_a_runner_up_that_is_also_over_the_threshold_is_fine_if_it_is_far_enough(reg):
    alba = reg.enroll("Alba Verzieri", vec(0.95, axis=1))
    reg.enroll("Bruno Meltrame", vec(0.79, axis=2))
    m = reg.match(PROBE, threshold=0.75, suggest_threshold=0.55, margin=0.05)
    assert m.accepted is True and m.person is alba


def test_the_margin_boundary_a_gap_equal_to_the_margin_is_accepted(reg):
    reg.enroll("Alba Verzieri", vec(0.90, axis=1))
    reg.enroll("Bruno Meltrame", vec(0.85, axis=2))
    loose = score_of(reg, PROBE)
    gap = loose.score - loose.runner_up_score
    at = reg.match(PROBE, threshold=0.75, suggest_threshold=0.55, margin=gap)
    assert at.accepted is True, "the comparison is strict: gap == margin still decides"
    over = reg.match(PROBE, threshold=0.75, suggest_threshold=0.55,
                     margin=math.nextafter(gap, 1.0))
    assert over.accepted is False
    assert "ambiguous" in over.reason


def test_a_single_person_registry_can_never_be_blocked_by_the_margin(reg):
    """With nobody to be confused with, there is nothing ambiguous about it."""
    p = reg.enroll("Alba Verzieri", vec(0.80))
    m = reg.match(PROBE, threshold=0.75, suggest_threshold=0.55, margin=0.99)
    assert m.accepted is True
    assert m.person is p
    assert m.runner_up is None
    assert m.runner_up_score == 0.0


def test_the_runner_up_is_the_second_best_of_everyone_not_just_the_last_enrolled(reg):
    reg.enroll("Alba Verzieri", vec(0.40, axis=1))
    reg.enroll("Bruno Meltrame", vec(0.93, axis=2))
    reg.enroll("Cora Ripaldi", vec(0.70, axis=3))
    reg.enroll("Dario Nusco", vec(0.10, axis=4))
    m = reg.match(PROBE, threshold=0.75, suggest_threshold=0.55, margin=0.05)
    assert m.accepted is True
    assert m.person.name == "Bruno Meltrame"
    assert m.runner_up == "Cora Ripaldi"
    assert m.runner_up_score == pytest.approx(0.70, abs=1e-6)


def test_in_the_suggest_zone_the_margin_is_not_consulted(reg):
    """Documents the current order of the checks in `match`.

    Two people are equally plausible and both sit under the certainty threshold. The
    threshold branch returns first, so the answer names one of them as "could be X"
    and never mentions that a second person scores the same. Nothing is applied, so
    this is not a wrong name, but the explanation handed to the user is missing the
    fact that would change their answer.
    """
    reg.enroll("Alba Verzieri", vec(0.60, axis=1))
    reg.enroll("Bruno Meltrame", vec(0.60, axis=2))
    m = reg.match(PROBE, threshold=0.75, suggest_threshold=0.55, margin=0.05)
    assert m.accepted is False, "the important half: no name is applied"
    assert m.candidate is not None
    assert "ambiguous" not in m.reason
    assert m.runner_up_score == pytest.approx(m.score, abs=1e-6)


# --------------------------------------------------------------------------- #
# matching: centroid versus single prints
# --------------------------------------------------------------------------- #

def test_the_score_is_the_best_of_the_centroid_and_the_single_prints(reg):
    """Two prints either side of the probe: the average lands on it, neither print does."""
    a = math.radians(60)
    v1 = np.zeros(DIM, dtype=np.float32); v1[0] = math.cos(a); v1[1] = math.sin(a)
    v2 = np.zeros(DIM, dtype=np.float32); v2[0] = math.cos(a); v2[1] = -math.sin(a)
    p = reg.enroll("Alba Verzieri", v1)
    reg.enroll("Alba Verzieri", v2)

    per_sample = V.cosine(PROBE, reg.vectors_of(p)).reshape(-1)
    assert list(per_sample) == pytest.approx([0.5, 0.5], abs=1e-6)
    centroid = float(V.cosine(PROBE, reg.centroid_of(p)[None, :]).reshape(-1)[0])
    assert centroid == pytest.approx(1.0, abs=1e-6)
    assert reg.match(PROBE).score == pytest.approx(1.0, abs=1e-6)


def test_a_single_print_scores_the_same_through_the_centroid(reg):
    p = reg.enroll("Alba Verzieri", vec(0.9))
    assert reg.centroid_of(p) == pytest.approx(vec(0.9), abs=1e-6)
    assert reg.match(PROBE).score == pytest.approx(0.9, abs=1e-6)


def test_centroid_and_vectors_of_are_empty_for_a_person_without_prints(reg):
    orphan = V.Person(id="deadbeef0002", name="Bruno Meltrame")
    reg.people[orphan.id] = orphan
    assert reg.centroid_of(orphan) is None
    assert len(reg.vectors_of(orphan)) == 0


# --------------------------------------------------------------------------- #
# rename / forget
# --------------------------------------------------------------------------- #

def test_rename_keeps_the_voice_prints(reg):
    p = reg.enroll("Alba Verzieri", vec(0.92))
    renamed = reg.rename("alba verzieri", "  Alba Ferrero  ")
    assert renamed is p
    assert p.name == "Alba Ferrero"
    assert p.rows == [0]
    assert reg.match(PROBE).person is p
    assert reg.rename("nobody at all", "x") is None


def test_forget_removes_the_person_and_leaves_the_others_matchable(reg):
    reg.enroll("Alba Verzieri", vec(0.95, axis=1))
    bruno = reg.enroll("Bruno Meltrame", vec(0.20, axis=2))
    assert reg.forget("ALBA VERZIERI") is True
    assert reg.by_name("Alba Verzieri") is None
    assert reg.forget("Alba Verzieri") is False
    # Row 0 is orphaned on purpose; Bruno's index must still point at Bruno.
    assert bruno.rows == [1]
    assert np.allclose(reg.vectors_of(bruno)[0], vec(0.20, axis=2))


# Regression: this used to fail.
def test_rename_onto_an_existing_name_does_not_create_two_of_the_same_person(reg):
    reg.enroll("Alba Verzieri", vec(0.90, axis=1))
    reg.enroll("Bruno Meltrame", vec(0.88, axis=2))

    # Refused, and said out loud. Merging two people is a decision, and doing it
    # silently leaves a registry where nothing can be told apart afterwards.
    with pytest.raises(ValueError, match="already in the registry"):
        reg.rename("Bruno Meltrame", "Alba Verzieri")

    names = [p.name.casefold() for p in reg.people.values()]
    assert len(set(names)) == len(names), f"duplicate names in the registry: {names}"
    assert reg.by_name("Bruno Meltrame") is not None, "the rename did not half-happen"


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #

def test_save_and_reload_round_trip(isolated_voices_dir):
    reg1 = V.VoiceRegistry()
    alba = reg1.enroll("Alba Verzieri", vec(0.95, axis=1),
                       source="ufficio.m4a", aliases=["Albi"], note="collega")
    reg1.enroll("Alba Verzieri", vec(0.40, axis=2), source="telefono.m4a")
    reg1.enroll("Bruno Meltrame", vec(0.20, axis=3), source="cena.m4a")
    reg1.save()

    assert V.REGISTRY_PATH.exists() and V.EMB_PATH.exists()
    assert V.REGISTRY_PATH.parent == isolated_voices_dir

    reg2 = V.VoiceRegistry()
    assert set(reg2.people) == set(reg1.people)
    assert np.array_equal(reg2.emb, reg1.emb)

    alba2 = reg2.by_name("Alba Verzieri")
    assert alba2.id == alba.id
    assert alba2.rows == [0, 1]
    assert alba2.aliases == ["Albi"]
    assert alba2.note == "collega"
    assert alba2.sources == ["ufficio.m4a", "telefono.m4a"]
    assert alba2.created == alba.created and alba2.updated == alba.updated
    assert reg2.by_name("Albi") is alba2, "aliases survive the round trip"

    # And the whole point: the names still attach on the next recording.
    m = reg2.match(PROBE, threshold=0.75, suggest_threshold=0.55, margin=0.05)
    assert m.accepted is True and m.person.name == "Alba Verzieri"


def test_the_saved_registry_is_readable_json_with_a_version(reg):
    reg.enroll("Alba Verzieri", vec(0.9), source="ufficio.m4a")
    reg.save()
    raw = json.loads(V.REGISTRY_PATH.read_text())
    assert raw["version"] == 1
    assert [p["name"] for p in raw["people"]] == ["Alba Verzieri"]
    assert raw["people"][0]["rows"] == [0]


def test_an_empty_registry_round_trips_without_inventing_anything(reg):
    reg.save()
    reloaded = V.VoiceRegistry()
    assert reloaded.people == {}
    assert reloaded.emb.shape[0] == 0
    assert reloaded.match(PROBE).accepted is False


def test_a_reloaded_registry_keeps_appending_at_the_right_row(reg):
    reg.enroll("Alba Verzieri", vec(0.95, axis=1))
    reg.save()

    reg2 = V.VoiceRegistry()
    bruno = reg2.enroll("Bruno Meltrame", vec(0.20, axis=2))
    assert bruno.rows == [1], "a new print must not land on somebody else's row"
    assert reg2.emb.shape == (2, DIM)
    assert np.allclose(reg2.vectors_of(reg2.by_name("Alba Verzieri"))[0], vec(0.95, axis=1))
    assert np.allclose(reg2.vectors_of(bruno)[0], vec(0.20, axis=2))


def test_forget_survives_a_save_and_reload_with_the_indices_intact(reg):
    reg.enroll("Alba Verzieri", vec(0.95, axis=1))
    reg.enroll("Bruno Meltrame", vec(0.20, axis=2))
    reg.forget("Alba Verzieri")
    reg.save()

    reg2 = V.VoiceRegistry()
    assert reg2.by_name("Alba Verzieri") is None
    bruno = reg2.by_name("Bruno Meltrame")
    assert bruno.rows == [1]
    assert np.allclose(reg2.vectors_of(bruno)[0], vec(0.20, axis=2))
    assert reg2.emb.shape == (2, DIM), "orphaned rows are kept so indices stay valid"


def _snapshot(directory: Path):
    if not directory.exists():
        return None
    return sorted((str(p), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in directory.rglob("*") if p.is_file())


def test_the_registry_writes_only_under_the_isolated_home(reg, tmp_path):
    real = Path.home() / ".scriba" / "voices"
    before = _snapshot(real)

    reg.enroll("Alba Verzieri", vec(0.9), source="ufficio.m4a")
    reg.save()

    assert V.REGISTRY_PATH.is_relative_to(tmp_path)
    assert V.EMB_PATH.is_relative_to(tmp_path)
    assert V.REGISTRY_PATH.exists() and V.EMB_PATH.exists()
    assert _snapshot(real) == before, "the user's own voice registry was touched"


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #

def test_summary_counts_prints_and_recordings_and_sorts_by_name(reg):
    reg.enroll("bruno meltrame", vec(0.20, axis=2), source="cena.m4a")
    reg.enroll("Alba Verzieri", vec(0.95, axis=1), source="ufficio.m4a", aliases=["Albi"])
    reg.enroll("Alba Verzieri", vec(0.40, axis=3), source="telefono.m4a")

    rows = reg.summary()
    assert [r["name"] for r in rows] == ["Alba Verzieri", "bruno meltrame"]
    assert rows[0]["prints"] == 2 and rows[0]["recordings"] == 2
    assert rows[0]["aliases"] == "Albi"
    assert rows[1]["prints"] == 1


# --------------------------------------------------------------------------- #
# the minimum-speech rule
# --------------------------------------------------------------------------- #

def test_voices_py_has_no_minimum_speech_gate_of_its_own():
    """Where the `voice_min_speech_sec` rule actually lives.

    It is not in this module: neither `enroll` nor `match` is told how many seconds of
    speech an embedding came from, so a voice print built from two seconds of "mhm" is
    enrolled and matched exactly like one built from ten minutes. The gate is applied
    once by the caller, in pipeline.Job.identify (scriba/pipeline.py:324), which skips
    the registry entirely for thin speakers.

    That makes the rule easy to bypass: `scriba whoami` (cli.py:234) and
    recurring.scan carry their own separate `--min-speech` default of 20.0s, and any
    other caller of VoiceRegistry.enroll gets no gate at all. Pinning it here so that
    moving the rule into the registry, or forgetting it in a new caller, is visible.
    """
    import inspect

    for fn in (V.VoiceRegistry.enroll, V.VoiceRegistry.match):
        params = set(inspect.signature(fn).parameters)
        assert not {"speech_sec", "duration", "min_speech", "min_speech_sec"} & params

    src = inspect.getsource(V)
    assert "min_speech" not in src and "voice_min_speech_sec" not in src


# --------------------------------------------------------------------------- #
# crash consistency: registry.json and embeddings.npy can disagree
# --------------------------------------------------------------------------- #

# Regression: this used to fail.
def test_a_lost_embeddings_file_must_not_hand_one_persons_rows_to_another(reg):
    reg.enroll("Alba Verzieri", vec(0.95, axis=1))
    reg.save()
    V.EMB_PATH.unlink()  # the half of save() that did not survive

    reg2 = V.VoiceRegistry()
    alba = reg2.by_name("Alba Verzieri")
    assert alba.rows == [0] and reg2.emb.size == 0

    bruno = reg2.enroll("Bruno Meltrame", vec(0.20, axis=2))
    assert np.allclose(reg2.vectors_of(bruno)[0], vec(0.20, axis=2))
    assert len(reg2.vectors_of(alba)) == 0, (
        "Alba has no voice print any more; she must not inherit Bruno's"
    )


# Regression: this used to fail.
def test_a_lost_embeddings_file_must_not_make_match_raise(reg):
    reg.enroll("Alba Verzieri", vec(0.95, axis=1))
    reg.enroll("Alba Verzieri", vec(0.40, axis=3))
    reg.save()
    V.EMB_PATH.unlink()

    reg2 = V.VoiceRegistry()
    reg2.enroll("Bruno Meltrame", vec(0.20, axis=2))
    reg2.match(PROBE)  # IndexError: index 1 is out of bounds for axis 0 with size 1


# Regression: this used to fail.
def test_match_reports_a_dimension_change_instead_of_crashing_in_numpy(reg):
    reg.enroll("Alba Verzieri", vec(0.9))
    with pytest.raises(ValueError, match="embedding model"):
        reg.match(np.ones(192, dtype=np.float32))
