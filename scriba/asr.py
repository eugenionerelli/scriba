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


def mps_available() -> tuple[bool, str]:
    """Whether this install of ctranslate2 can use the GPU, and why not when it cannot.

    Upstream ctranslate2 has no Metal backend, which is the single reason a
    seven-minute recording used to cost seven minutes of CPU. OpenNMT/CTranslate2#2077
    adds one, and it is not in any released wheel: a build from that branch has to be
    installed on purpose. So this asks the library rather than the operating system.
    A Mac with a perfectly good GPU and a stock wheel must come back False.
    """
    try:
        import ctranslate2
    except ImportError:
        return False, "ctranslate2 is not installed"
    if not hasattr(ctranslate2, "get_mps_device_count"):
        return False, ("this ctranslate2 has no Metal backend: it is the released "
                       "wheel, which is CPU-only on Apple Silicon")
    try:
        if ctranslate2.get_mps_device_count() < 1:
            return False, "no Metal device"
    except Exception as exc:  # pragma: no cover - depends on the build
        return False, f"Metal unavailable: {exc}"
    return True, ""


def _asr_device(s) -> tuple[str, str]:
    """The device to decode on, and the numeric type to decode in.

    They are decided together because the sensible type differs per device: int8 is
    what makes the CPU bearable and is the slowest thing the GPU can be asked to do.
    Measured on an M4 with the branch build, batch of one: MPS int8 57.6 s against
    MPS float16 24.9 s and CPU int8 50.5 s. Asking for int8 on Metal would look like
    asking for the fast path and get the slowest one in the build.

    "auto" prefers Metal when the installed ctranslate2 has it. On the reference
    recording that is 80 s against 443 s, with 724 words against 725 and the same
    text; the whole difference is where the matrices are multiplied.
    """
    wanted = getattr(s, "asr_device", "auto")
    if wanted == "cpu":
        return "cpu", s.compute_type

    ok, why = mps_available()
    if wanted == "mps":
        if not ok:
            raise RuntimeError(
                f"asr_device is set to mps but {why}. Either install a ctranslate2 "
                "built with -DWITH_MPS=ON, or set asr_device back to auto.")
        return "mps", _mps_compute_type(s.compute_type)
    return ("mps", _mps_compute_type(s.compute_type)) if ok else ("cpu", s.compute_type)


def _mps_compute_type(requested: str) -> str:
    """float16 on the GPU unless somebody insisted on something else.

    int8 is the scriba default because of the CPU, and carrying it over to Metal
    would silently pick the slowest configuration this build offers. A person who
    writes float32 in the settings meant it and gets it.
    """
    return "float16" if requested in ("int8", "int8_float32", "auto", "default") else requested


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

    # The words can come from two places. The boundaries only ever come from one.
    #
    # Apple's transcriber runs on the Neural Engine and returns text in about a
    # hundredth of the time whisper takes on this hardware, but its word ranges
    # are contiguous: each word starts where the last one ended, so the silence
    # between them ends up inside them. Speaker attribution is decided by overlap
    # with the diarization, so that would cost exactly what this tool is for.
    #
    # Whichever engine writes the text, wav2vec2 places the words. That is the
    # same alignment stage as before, and on the reference recording the pair
    # measured better than the pipeline it replaces: 0.223 s average word against
    # 0.282 s, and 221 s of silence recognised against 182 s.
    if s.backend == "apple":
        return _transcribe_apple(wav, s, progress=progress)

    device, compute_type = _asr_device(s)
    threads = _cpu_threads(s.threads)
    # "auto" is a scriba value, not a whisper one: by this point it must already be
    # resolved to an ISO code. Passing it through would make whisper look for a
    # language called "auto" and fail with an opaque error halfway through loading.
    language = None if s.language in ("", "auto", None) else s.language

    if device == "mps":
        print(f"[scriba] transcription: {s.model} on Metal ({compute_type})")

    model = whisperx.load_model(
        s.model,
        device=device,
        compute_type=compute_type,
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


def _transcribe_apple(wav: Path, s: Settings, *, progress: bool = True) -> Transcript:
    import gc

    import whisperx

    from . import apple

    language = s.language if s.language not in ("", "auto", None) else "en"
    print(f"[scriba] transcription: Apple on-device model in {language} (Neural Engine)")
    segments = apple.transcribe(wav, language)
    if not segments:
        raise RuntimeError(
            f"the Apple model produced nothing for {wav.name}. Either the recording "
            "holds no speech it recognises, or the language is wrong."
        )

    word_level = False
    if s.align:
        try:
            print(f"[scriba] alignment: word-level timestamps ({language})")
            audio = whisperx.load_audio(str(wav))
            align_model, metadata = whisperx.load_align_model(
                language_code=language, device="cpu"
            )
            aligned = whisperx.align(
                [{k: v for k, v in seg.items() if k != "confidence"} for seg in segments],
                align_model, metadata, audio, "cpu",
                return_char_alignments=False, print_progress=progress,
            )
            # Carry the confidence across. The aligner knows nothing about it, and
            # it is the one thing this engine gives that whisper does not.
            by_start = {round(float(seg["start"]), 2): seg.get("confidence")
                        for seg in segments}
            for seg in aligned["segments"]:
                conf = by_start.get(round(float(seg.get("start", -1)), 2))
                if conf is not None:
                    seg["confidence"] = conf
            segments = aligned["segments"]
            word_level = True
            del align_model
            gc.collect()
        except Exception as exc:
            print(f"[scriba] alignment failed ({exc}); keeping the model's own "
                  "timings, which run word into word")

    return Transcript(segments=segments, language=language, word_level=word_level)
