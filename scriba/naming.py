"""Putting names to voices.

Three routes, in order of reliability:

1. **Voice print** (voices.py). Deterministic and reusable: if the person is already
   enrolled, the name attaches on its own. It is the only route that gets better over time.
2. **Cues in the text**. In real conversations names get said out loud ("Good morning
   Ada", "I am Rafiq"). Here we pull out the spots where that happens, with enough
   context around them to be judged.
3. **A human or an LLM judging** the briefing produced by (2). We don't do this
   automatically: we produce the briefing and whoever decides, decides.

None of these invents a name. When the name isn't known the voice stays "Voice 2", and
the source document says plainly that nobody identified it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# The phrasings people use to name someone out loud. Deliberately broad: they are here
# to surface *candidates to read*, not to decide anything.
CUE_PATTERNS = {
    "it": [
        r"\b(?:mi chiamo|sono io,? |sono)\s+([A-Z][\wàèéìòù'-]+)",
        r"\b(?:ciao|buongiorno|buonasera|salve|grazie|senti|scusa|allora)[,]?\s+([A-Z][\wàèéìòù'-]+)",
        r"\b(?:dice|dico a|chiedi a|secondo)\s+([A-Z][\wàèéìòù'-]+)",
    ],
    "es": [
        r"\b(?:me llamo|soy)\s+([A-Z][\wáéíóúñ'-]+)",
        r"\b(?:hola|buenos días|buenas tardes|gracias|oye|perdona|mira|vale)[,]?\s+([A-Z][\wáéíóúñ'-]+)",
        r"\b(?:dice|dile a|pregúntale a|según)\s+([A-Z][\wáéíóúñ'-]+)",
    ],
    "en": [
        r"\b(?:my name is|i'm|i am|this is)\s+([A-Z][\w'-]+)",
        r"\b(?:hi|hello|hey|thanks|thank you|sorry)[,]?\s+([A-Z][\w'-]+)",
    ],
}

# Words that start with a capital letter and are not people's names.
STOPWORDS = {
    "il", "la", "lo", "un", "una", "e", "ma", "però", "quindi", "allora", "sì", "no",
    "el", "los", "las", "una", "pero", "entonces", "vale", "bueno", "sí", "que",
    "the", "and", "but", "so", "yes", "ok", "okay", "università", "universidad",
}


@dataclass
class SpeakerProfile:
    label: str
    speech_seconds: float
    turn_count: int
    first_seen: float
    samples: list[str] = field(default_factory=list)
    registry_name: str | None = None       # certain: applies itself
    registry_candidate: str | None = None  # borderline: only proposed, never applied
    registry_score: float = 0.0
    registry_note: str = ""
    name_cues: list[str] = field(default_factory=list)


def extract_cues(turns: list[dict], language: str) -> dict[str, list[str]]:
    """Sentences where a personal name is spoken, per speaker.

    Returns the whole sentence, not the name alone: the reader has to be able to tell
    whether the name belongs to *the person speaking* or *the person being spoken to*,
    which is the distinction automatic attribution always gets wrong.
    """
    patterns = CUE_PATTERNS.get(language, CUE_PATTERNS["en"])
    out: dict[str, list[str]] = {}
    for t in turns:
        text = t.get("text", "")
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                candidate = m.group(1)
                if candidate.casefold() in STOPWORDS or len(candidate) < 3:
                    continue
                if not candidate[0].isupper():
                    continue
                spk = t.get("speaker") or "?"
                snippet = text.strip()
                if len(snippet) > 240:
                    start = max(m.start() - 100, 0)
                    snippet = "…" + text[start:m.end() + 100].strip() + "…"
                out.setdefault(spk, [])
                if snippet not in out[spk]:
                    out[spk].append(snippet)
    return out


def build_profiles(
    turns: list[dict],
    speech_time: dict[str, float],
    language: str,
    matches: dict[str, dict] | None = None,
    *,
    n_samples: int = 6,
) -> list[SpeakerProfile]:
    matches = matches or {}
    cues = extract_cues(turns, language)
    by_speaker: dict[str, list[dict]] = {}
    for t in turns:
        by_speaker.setdefault(t.get("speaker") or "?", []).append(t)

    profiles: list[SpeakerProfile] = []
    for label, items in by_speaker.items():
        longest = sorted(items, key=lambda t: len(t["text"]), reverse=True)[:n_samples]
        longest.sort(key=lambda t: t["start"])
        m = matches.get(label, {})
        profiles.append(SpeakerProfile(
            label=label,
            speech_seconds=speech_time.get(label, 0.0),
            turn_count=len(items),
            first_seen=min(t["start"] for t in items),
            samples=[t["text"][:400] for t in longest],
            registry_name=m.get("name"),
            registry_candidate=m.get("candidate"),
            registry_score=float(m.get("score", 0.0)),
            registry_note=m.get("reason", ""),
            name_cues=cues.get(label, [])[:8],
        ))
    profiles.sort(key=lambda p: p.speech_seconds, reverse=True)
    return profiles


def dossier(profiles: list[SpeakerProfile], *, language: str, title: str) -> str:
    """The document to put in front of a human (or an LLM) so the names can be decided."""
    lines = [
        f"# Who is who: {title}",
        "",
        f"Language: {language}. Distinct voices found: {len(profiles)}.",
        "",
        "For each voice: how much it speaks, how it introduces itself, and the spots "
        "where a personal name is spoken. Careful: a name spoken by one voice is "
        "usually the name of the *other* one, not its own.",
        "",
    ]
    for p in profiles:
        mm, ss = divmod(int(p.speech_seconds), 60)
        lines.append(f"## {p.label}")
        lines.append("")
        lines.append(f"- Speech: {mm}m {ss:02d}s across {p.turn_count} turns "
                     f"(first turn at {int(p.first_seen//60)}:{int(p.first_seen%60):02d})")
        if p.registry_name:
            lines.append(f"- **Voice registry**: matches **{p.registry_name}** "
                         f"(similarity {p.registry_score:.3f})")
        elif p.registry_candidate:
            lines.append(f"- **Voice registry**: resembles **{p.registry_candidate}** "
                         f"({p.registry_score:.3f}). That score is too low to decide by "
                         "itself. Listen again and confirm yourself.")
        elif p.registry_note:
            lines.append(f"- Voice registry: no match ({p.registry_note})")
        if p.name_cues:
            lines.append("- Sentences containing personal names:")
            for c in p.name_cues:
                lines.append(f"  - «{c}»")
        lines.append("- Longest turns:")
        for s in p.samples:
            lines.append(f"  - «{s.strip()}»")
        lines.append("")
    return "\n".join(lines)


def apply_names(mapping: dict[str, str], profiles: list[SpeakerProfile]) -> dict[str, str]:
    """Merges the manually decided names with the ones already certain from the voice registry."""
    names: dict[str, str] = {}
    for p in profiles:
        if p.registry_name:
            names[p.label] = p.registry_name
    names.update({k: v for k, v in mapping.items() if v})
    return names
