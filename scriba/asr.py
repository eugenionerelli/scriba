"""Transcription. The backend is abstracted; today it is whisperx/faster-whisper.

The settings are not the whisperx CLI defaults: they are tuned for real Italian
conversations recorded with a phone lying on the table, which is the use case
here. Every deviation is commented, so it can be checked instead of taken on faith.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings

# Priming prompt: it shows whisper an example of the punctuation and accents we want.
# This is not "prompt magic". It conditions the decoder on the right register
# (spoken Italian, punctuated, with capital letters) and cuts down on the
# all-lowercase, comma-free transcripts that make the markdown unreadable.
DEFAULT_INITIAL_PROMPTS = {
    "it": ("Registrazione di una conversazione in italiano fra più persone. "
           "Trascrizione fedele, con punteggiatura, maiuscole e accenti corretti. "
           "Ecco, allora, secondo me la questione è un'altra: perché no?"),
    "es": ("Grabación de una conversación en español entre varias personas. "
           "Transcripción fiel, con puntuación, mayúsculas y acentos correctos. "
           "Bueno, entonces, a ver: ¿cómo lo hacemos? Vale, perfecto."),
    "en": ("Recording of a conversation in English between several people. "
           "Faithful transcription with punctuation and capitalisation. "
           "So, right, the way I see it is: why not?"),
}


@dataclass
class Transcript:
    segments: list[dict[str, Any]]
    language: str
    word_level: bool


def _cpu_threads(requested: int) -> int:
    if requested > 0:
        return requested
    # ctranslate2 has no Metal backend: on Apple Silicon everything runs on CPU.
    # Only the P cores are worth using. Throwing the E cores in makes it slower, not
    # faster: they run about 4x behind, and the sync barrier waits for the slowest
    # thread. That is why the sysctl below asks for perflevel0, the P cluster.
    try:
        import subprocess
        perf = int(subprocess.run(["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                                  capture_output=True, text=True).stdout.strip())
        return max(perf, 1)
    except Exception:
        return max((os.cpu_count() or 4) // 2, 1)


def build_asr_options(s: Settings) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "beam_size": s.beam_size,
        "best_of": s.beam_size,
        # The temperature fallback is the safety net against loops: if greedy
        # decoding produces text that is too compressible (repetitions) or too
        # unlikely, it retries hotter. Removing it makes the output unstable.
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if s.temperature_fallback else [0.0],
        "compression_ratio_threshold": s.compression_ratio_threshold,
        "log_prob_threshold": s.log_prob_threshold,
        "no_speech_threshold": s.no_speech_threshold,
        # False: every window is independent. With True, whisper latches onto its
        # own previous output and falls into repetition loops during long pauses.
        # That is the classic failure on voice memos full of silences.
        "condition_on_previous_text": s.condition_on_previous_text,
        "initial_prompt": (s.initial_prompt if s.initial_prompt is not None
                           else DEFAULT_INITIAL_PROMPTS.get(s.language)),
        "hotwords": ", ".join(s.hotwords) if s.hotwords else None,
        "suppress_numerals": False,
    }
    return opts


def transcribe(
    wav: Path,
    s: Settings,
    *,
    progress: bool = True,
) -> Transcript:
    import whisperx

    device = "cpu"  # ctranslate2 runs on CPU on a Mac: say so instead of pretending
    threads = _cpu_threads(s.threads)
    # "auto" is a scriba value, not a whisper one: by this point it must already be
    # resolved to an ISO code. Passing it through would make whisper look for a
    # language called "auto" and fail with an opaque error halfway through loading.
    language = None if s.language in ("", "auto", None) else s.language

    model = whisperx.load_model(
        s.model,
        device=device,
        compute_type=s.compute_type,
        language=language,
        asr_options=build_asr_options(s),
        vad_options={"vad_onset": s.vad_onset, "vad_offset": s.vad_offset},
        threads=threads,
    )

    audio = whisperx.load_audio(str(wav))
    result = model.transcribe(audio, batch_size=s.batch_size, print_progress=progress)
    language = result.get("language", s.language)

    # Free the memory right away: large-v3 int8 takes ~1.6 GB, the alignment model
    # ~1.3 GB and pyannote another GB. On a 16 GB machine it is better not to keep
    # all three alive at once.
    del model
    import gc
    gc.collect()

    word_level = False
    if s.align:
        try:
            # The trailing colon matters: the app detects this phase from the
            # "alignment:" prefix, and the failure message below deliberately does
            # not carry it. Without that distinction the app would announce
            # "aligning" at the exact moment alignment gave up.
            print(f"[scriba] alignment: word-level timestamps ({language})")
            align_model, metadata = whisperx.load_align_model(
                language_code=language, device=device
            )
            result = whisperx.align(
                result["segments"], align_model, metadata, audio, device,
                return_char_alignments=False, print_progress=progress,
            )
            word_level = True
            del align_model
            gc.collect()
        except Exception as exc:  # pragma: no cover - depends on the network
            # Alignment is an improvement, not a requirement: without it the
            # timestamps stay whisper's (coarser) ones and diarization works
            # per segment instead of per word.
            print(f"[scriba] alignment failed ({exc}); continuing without word-level timestamps")

    return Transcript(segments=result["segments"], language=language, word_level=word_level)
