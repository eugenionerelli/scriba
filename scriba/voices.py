"""Voice registry: voice prints that survive across separate recordings.

This is the piece whisperx and every menu bar app are missing. Diarization gives
you SPEAKER_00 / SPEAKER_01, but those labels are arbitrary and start over on every
file: the same person is SPEAKER_00 on Monday and SPEAKER_02 on Tuesday.

Here we keep an archive of embeddings (256-dim, wespeaker-voxceleb-resnet34-LM, the
same ones pyannote already computes during diarization) tied to a name. On the next
recording we compare the new centroids against the archive and the names attach
themselves.

Matching policy, conservative on purpose: an "I don't know" beats a wrong name,
because a wrong name quietly poisons every downstream transcript and then lands in
a summary as if it were a fact.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import VOICES_DIR

REGISTRY_PATH = VOICES_DIR / "registry.json"
EMB_PATH = VOICES_DIR / "embeddings.npy"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between a vector (or batch) and a matrix."""
    return l2norm(a) @ l2norm(b).T


@dataclass
class Person:
    id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    note: str = ""
    rows: list[int] = field(default_factory=list)   # rows in the embeddings matrix
    sources: list[str] = field(default_factory=list)
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)


@dataclass
class Match:
    person: Person | None
    score: float
    runner_up: str | None
    runner_up_score: float
    reason: str
    # Borderline: close to someone. Not close enough to decide on its own, so we
    # only hand it to the user as a suggestion. That is the difference between an
    # assistant that gets it wrong in silence and one that says "might be her, check".
    candidate: Person | None = None

    @property
    def accepted(self) -> bool:
        return self.person is not None


class VoiceRegistry:
    """Append-only archive of voice prints.

    The `emb` matrix grows by one row per enrollment; `Person.rows` indexes the rows
    that belong to that person. We never delete rows (the indices would end up
    misaligned): `forget` drops the person and leaves their rows orphaned, harmless.
    """

    def __init__(self) -> None:
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        self.people: dict[str, Person] = {}
        self.emb: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._load()

    # ------------------------------------------------------------------ I/O
    def _load(self) -> None:
        if REGISTRY_PATH.exists():
            raw = json.loads(REGISTRY_PATH.read_text())
            self.people = {p["id"]: Person(**p) for p in raw.get("people", [])}
        if EMB_PATH.exists():
            self.emb = np.load(EMB_PATH)

    def save(self) -> None:
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        REGISTRY_PATH.write_text(json.dumps(
            {"version": 1, "people": [p.__dict__ for p in self.people.values()]},
            indent=2, ensure_ascii=False,
        ))
        np.save(EMB_PATH, self.emb)

    # ---------------------------------------------------------------- reads
    def by_name(self, name: str) -> Person | None:
        key = name.strip().casefold()
        for p in self.people.values():
            if p.name.casefold() == key or key in {a.casefold() for a in p.aliases}:
                return p
        return None

    def vectors_of(self, person: Person) -> np.ndarray:
        if not person.rows or self.emb.size == 0:
            return np.zeros((0, self.emb.shape[1] if self.emb.ndim == 2 else 0), dtype=np.float32)
        return self.emb[person.rows]

    def centroid_of(self, person: Person) -> np.ndarray | None:
        v = self.vectors_of(person)
        if len(v) == 0:
            return None
        return l2norm(l2norm(v).mean(axis=0))

    # -------------------------------------------------------------- writes
    def enroll(
        self,
        name: str,
        embedding: np.ndarray,
        *,
        source: str = "",
        aliases: list[str] | None = None,
        note: str = "",
    ) -> Person:
        """Add a voice print to a person, creating the person if they do not exist."""
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(embedding)):
            raise ValueError("invalid embedding (contains NaN/inf)")

        if self.emb.size == 0:
            self.emb = np.zeros((0, embedding.shape[0]), dtype=np.float32)
        elif self.emb.shape[1] != embedding.shape[0]:
            raise ValueError(
                f"embedding dimension {embedding.shape[0]} != registry {self.emb.shape[1]}; "
                "you switched embedding model: start from a fresh registry"
            )

        person = self.by_name(name)
        if person is not None:
            # Re-running `scriba name` on the same file is routine (fixing a typo,
            # adding a person). Without this check, each run would pile up an
            # identical copy of the same voice print, which then outweighs the
            # others in the comparison and skews later matches.
            existing = self.vectors_of(person)
            if len(existing) and float(cosine(embedding, existing).max()) > 0.999:
                person.updated = _now()
                if source and source not in person.sources:
                    person.sources.append(source)
                return person
        if person is None:
            person = Person(id=uuid.uuid4().hex[:12], name=name.strip(),
                            aliases=aliases or [], note=note)
            self.people[person.id] = person

        self.emb = np.vstack([self.emb, embedding[None, :]])
        person.rows.append(self.emb.shape[0] - 1)
        if source and source not in person.sources:
            person.sources.append(source)
        person.updated = _now()
        return person

    def rename(self, old: str, new: str) -> Person | None:
        p = self.by_name(old)
        if p:
            p.name = new.strip()
            p.updated = _now()
        return p

    def forget(self, name: str) -> bool:
        p = self.by_name(name)
        if not p:
            return False
        del self.people[p.id]
        return True

    # ----------------------------------------------------------- matching
    def match(
        self,
        embedding: np.ndarray,
        *,
        threshold: float = 0.75,
        suggest_threshold: float = 0.55,
        margin: float = 0.05,
    ) -> Match:
        """Who is this voice?

        For each person we take the larger of (a) the similarity to the centroid and
        (b) the best similarity to a single voice print. This matters for people
        enrolled in different acoustic settings (phone, room, outdoors), where the
        averaged centroid ends up resembling none of the recordings.
        """
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(embedding)) or not self.people:
            return Match(None, 0.0, None, 0.0,
                         "empty registry" if not self.people else "invalid embedding")

        scores: list[tuple[float, Person]] = []
        for p in self.people.values():
            vecs = self.vectors_of(p)
            if len(vecs) == 0:
                continue
            per_sample = cosine(embedding, vecs).reshape(-1)
            centroid = self.centroid_of(p)
            best = float(per_sample.max())
            if centroid is not None:
                best = max(best, float(cosine(embedding, centroid[None, :]).reshape(-1)[0]))
            scores.append((best, p))

        if not scores:
            return Match(None, 0.0, None, 0.0, "no voice prints in the registry")

        scores.sort(key=lambda t: t[0], reverse=True)
        top_score, top = scores[0]
        second_score, second = (scores[1] if len(scores) > 1 else (0.0, None))
        second_name = second.name if second else None

        if top_score < suggest_threshold:
            return Match(None, top_score, second_name, second_score,
                         f"no useful similarity (the closest is {top.name} at "
                         f"{top_score:.3f}, below {suggest_threshold:.2f}): "
                         "most likely a new voice")
        if top_score < threshold:
            return Match(None, top_score, second_name, second_score,
                         f"could be {top.name} ({top_score:.3f}). That is under the "
                         f"certainty threshold ({threshold:.2f}), so confirm it yourself",
                         candidate=top)
        if top_score - second_score < margin and second is not None:
            return Match(None, top_score, second_name, second_score,
                         f"ambiguous: {top.name} {top_score:.3f} against {second.name} "
                         f"{second_score:.3f}, they are too close to each other "
                         f"(margin < {margin:.2f})",
                         candidate=top)
        return Match(top, top_score, second_name, second_score, "accepted")

    def summary(self) -> list[dict]:
        out = []
        for p in self.people.values():
            out.append({
                "name": p.name,
                "aliases": ", ".join(p.aliases),
                "prints": len(p.rows),
                "recordings": len(p.sources),
                "updated": p.updated,
            })
        return sorted(out, key=lambda d: d["name"].casefold())
