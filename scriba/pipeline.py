"""The orchestrator: from audio file to NotebookLM source.

A job is a folder under ~/.scriba/jobs/<slug>/ holding all intermediate state.
That is deliberate: the phases cost minutes of CPU, and redoing only the name
attribution without re-transcribing is the normal case, not the exception.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import asr, audio, diarize, export, lang, naming
from .config import JOBS_DIR, Settings, ensure_dirs, hf_token
from .voices import VoiceRegistry

Reporter = Callable[[str], None]

# Machine-readable error markers.
#
# The macOS app has to tell "you never configured a token" apart from any other
# failure, so it can say something useful instead of dumping a stack trace. It used
# to do that by matching a phrase from the human message. That broke silently the
# moment the sentence was reworded, on an error path no test exercises. A marker is
# ugly in a terminal and impossible to break by accident; prose is the opposite.
ERR_NO_TOKEN = "[scriba:error:hf-token]"


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s-]", "", text).strip()
    text = re.sub(r"[\s_-]+", "-", text)
    return text.lower() or "recording"


def job_slug(source: Path) -> str:
    """Working folder name: readable, and still unique per path.

    With the file name alone, a `Recording 38.m4a` on the Desktop and another one
    under ~/Library/CloudStorage would land in the same job folder, and the second
    would silently reuse the transcript of the first. The suffix fingerprints the
    containing folder, so two files with the same name in different folders stay
    apart.

    Move the source file to another folder and the slug changes with it: the old
    job is orphaned and the work is redone. That is the honest trade for having
    the folder in the name at all.
    """
    digest = hashlib.sha1(str(source.parent).encode()).hexdigest()[:6]
    return f"{slugify(source.stem)}-{digest}"


@dataclass
class JobResult:
    job_dir: Path
    outputs: list[Path]
    language: str
    speakers: list[str]
    names: dict[str, str]
    unresolved: list[str]
    dossier_path: Path
    duration: float


class Job:
    def __init__(self, source: Path, settings: Settings | None = None,
                 *, report: Reporter | None = None):
        ensure_dirs()
        self.source = Path(source).expanduser().resolve()
        if not self.source.exists():
            raise FileNotFoundError(self.source)
        self.s = settings or Settings.load()
        self.report: Reporter = report or (lambda m: print(f"[scriba] {m}"))
        self.dir = JOBS_DIR / job_slug(self.source)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.wav = self.dir / "audio16k.wav"
        self.state_path = self.dir / "state.json"
        self.state: dict[str, Any] = (
            json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        )

    # ------------------------------------------------------------------ util
    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False,
                                              default=str))

    def _cache(self, key: str, path: Path, produce: Callable[[], Any],
               load: Callable[[Path], Any], dump: Callable[[Any, Path], None],
               *, force: bool) -> Any:
        if path.exists() and not force:
            self.report(f"{key}: reusing the cached result")
            return load(path)
        value = produce()
        dump(value, path)
        return value

    # ---------------------------------------------------------------- phases
    def prepare_audio(self, *, force: bool = False) -> Path:
        if self.wav.exists() and not force:
            self.report("audio: already prepared")
            return self.wav
        self.report("audio: converting to 16 kHz mono WAV, normalizing the level")
        audio.prepare(self.source, self.wav)
        info = audio.probe(self.source)
        self.state["duration"] = info.duration
        self.state["source"] = str(self.source)
        self.state["codec"] = info.codec
        self._save_state()
        return self.wav

    def detect_language(self, *, force: bool = False) -> str:
        if self.s.language and self.s.language != "auto":
            self.state["language"] = self.s.language
            self.state["language_note"] = "set by hand"
            self._save_state()
            return self.s.language
        if self.state.get("language") and not force:
            return self.state["language"]

        self.report("language: sampling the file at 5 points")
        guess = lang.detect(self.wav)
        self.state["language"] = guess.language
        self.state["language_confidence"] = guess.confidence
        self.state["language_note"] = guess.note
        self.state["language_samples"] = guess.samples
        self._save_state()
        flag = "" if guess.reliable else "  ⚠️  "
        # The note can itself contain commas, so it goes in brackets. A second colon
        # would read badly against the "language:" prefix the app matches on.
        self.report(f"language:{flag} {guess.language} ({guess.confidence:.0%}) [{guess.note}]")
        return guess.language

    def transcribe(self, *, force: bool = False) -> list[dict]:
        path = self.dir / "transcript.json"
        if path.exists() and not force:
            self.report("transcription: reusing the cache")
            data = json.loads(path.read_text())
            self.state["word_level"] = data.get("word_level", False)
            return data["segments"]

        s = Settings(**{**asdict(self.s), "language": self.state.get("language", self.s.language)})
        self.report(f"transcription: {s.model} in {s.language} (CPU, {s.compute_type})")
        t = asr.transcribe(self.wav, s)
        path.write_text(json.dumps(
            {"segments": t.segments, "language": t.language, "word_level": t.word_level},
            ensure_ascii=False,
        ))
        self.state["word_level"] = t.word_level
        self._save_state()
        return t.segments

    def diarize(self, *, force: bool = False) -> diarize.Diarization:
        turns_path = self.dir / "diarization.json"
        emb_path = self.dir / "embeddings.npz"
        if turns_path.exists() and emb_path.exists() and not force:
            self.report("diarization: reusing the cache")
            raw = json.loads(turns_path.read_text())
            emb = dict(np.load(emb_path))
            return diarize.Diarization(
                turns=[diarize.SpeakerTurn(**t) for t in raw],
                embeddings={k: np.asarray(v) for k, v in emb.items()},
            )

        token = hf_token()
        if not token:
            raise RuntimeError(
                f"{ERR_NO_TOKEN} The Hugging Face token for pyannote is missing.\n"
                "  scriba token hf_xxxxxxxx   # stores it in the Keychain\n"
                "and accept the terms at hf.co/pyannote/speaker-diarization-3.1"
            )
        # Naming the model and the device on every run is deliberate. Which of the two
        # models is in use depends on the installed pyannote, and Metal is picked
        # automatically. Both belong in the log rather than in someone's memory.
        dev = diarize._pick_device(self.s.diarize_device)
        self.report(f"diarization: {diarize.DIARIZATION_MODEL.split('/')[-1]} "
                    f"on {dev}, with voice print extraction")
        dia = diarize.run(
            self.wav, hf_token=token,
            min_speakers=self.s.min_speakers, max_speakers=self.s.max_speakers,
            device=self.s.diarize_device,
        )
        turns_path.write_text(json.dumps([asdict(t) for t in dia.turns], ensure_ascii=False))
        np.savez(emb_path, **dia.embeddings)
        self.report(f"diarization: {len(dia.speakers())} distinct voices, "
                    f"{len(dia.embeddings)} usable voice prints")
        return dia

    def identify(self, dia: diarize.Diarization) -> dict[str, dict]:
        """Match the voice prints against the voice registry."""
        reg = VoiceRegistry()
        speech = dia.speech_time()
        matches: dict[str, dict] = {}
        for label, vec in dia.embeddings.items():
            if speech.get(label, 0.0) < self.s.voice_min_speech_sec:
                # "candidate" has to be present even here. The reporting loop below
                # reads that field on every entry, and a dict built without it raised
                # KeyError on exactly the speakers this branch exists to handle.
                matches[label] = {
                    "name": None, "candidate": None, "score": 0.0,
                    "reason": f"only {speech.get(label, 0):.0f}s of speech: "
                              "voice print too thin to trust",
                }
                continue
            m = reg.match(vec,
                          threshold=self.s.voice_match_threshold,
                          suggest_threshold=self.s.voice_suggest_threshold,
                          margin=self.s.voice_match_margin)
            matches[label] = {
                "name": m.person.name if m.person else None,
                "candidate": m.candidate.name if m.candidate else None,
                "score": m.score, "reason": m.reason,
            }
        for label, info in matches.items():
            if info.get("name"):
                self.report(f"voice {label} → {info['name']} ({info['score']:.3f})")
            elif info.get("candidate"):
                self.report(f"voice {label} ≈ {info['candidate']}? "
                            f"({info['score']:.3f}), to be confirmed")
        return matches

    # ------------------------------------------------------------------- run
    def run(self, *, force: str | None = None) -> JobResult:
        f_all = force == "all"
        self.prepare_audio(force=f_all)
        self.detect_language(force=f_all or force == "lang")
        segments = self.transcribe(force=f_all or force == "asr")

        dia = None
        matches: dict[str, dict] = {}
        if self.s.diarize:
            dia = self.diarize(force=f_all or force == "diar")
            segments = diarize.assign(segments, dia,
                                      word_level=self.state.get("word_level", False))
            matches = self.identify(dia)

        turns = diarize.to_turns(segments)
        (self.dir / "turns.json").write_text(json.dumps(turns, ensure_ascii=False, indent=1))

        language = self.state.get("language", "it")
        speech = dia.speech_time() if dia else {}
        profiles = naming.build_profiles(turns, speech, language, matches)

        dossier_path = self.dir / "who-is-who.md"
        dossier_path.write_text(naming.dossier(
            profiles, language=language, title=self.source.stem))

        # Names already settled in an earlier pass stay put.
        saved: dict[str, str] = self.state.get("names", {})
        names = naming.apply_names(saved, profiles)
        self.state["names"] = names
        self.state["matches"] = matches
        self._save_state()

        unresolved = [export.display(p.label, names) for p in profiles
                      if p.label not in names]

        meta = {
            "title": self.source.stem,
            "recorded": datetime.fromtimestamp(self.source.stat().st_mtime),
            "duration": self.state.get("duration", 0.0),
            "language": language,
            "source_file": self.source.name,
            "speaker_stats": speech,
            "unresolved": unresolved,
        }
        outputs = export.write_all(
            self.dir / "output", slugify(self.source.stem), self.s.output_formats,
            turns=turns, segments=segments, names=names, meta=meta,
            matches=[{"speaker": k, **v} for k, v in matches.items()],
        )
        self.report(f"wrote {len(outputs)} files to {self.dir / 'output'}")

        return JobResult(
            job_dir=self.dir, outputs=outputs, language=language,
            speakers=[p.label for p in profiles], names=names,
            unresolved=unresolved, dossier_path=dossier_path,
            duration=self.state.get("duration", 0.0),
        )

    # ---------------------------------------------------------------- rename
    def set_names(self, mapping: dict[str, str], *, enroll: bool = True) -> JobResult:
        """Assign the names, then learn them for next time. That is the whole point."""
        self.state["names"] = {**self.state.get("names", {}), **mapping}
        self._save_state()

        if enroll:
            emb_path = self.dir / "embeddings.npz"
            if emb_path.exists():
                reg = VoiceRegistry()
                data = np.load(emb_path)
                added = 0
                for label, name in mapping.items():
                    if label in data and name:
                        reg.enroll(name, data[label], source=self.source.name)
                        added += 1
                reg.save()
                self.report(f"voice registry: {added} voice prints enrolled. "
                            "On the next recording these names attach on their own")
        return self.run()
