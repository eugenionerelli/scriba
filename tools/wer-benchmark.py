#!/usr/bin/env python3
"""Word error rate for each backend, against text a human wrote.

The blind listening test in tools/compare-engines.py asks a person to judge the
places where two engines disagree. It is honest and it is slow, and it only ever
looks at the contested spots, which is a biased sample of the recording by
construction. This does the other half: a public dataset where somebody already
wrote down what was said, so the score is a number and nobody's ear is involved.

FLEURS is read speech in 102 languages, which is the right shape for a
multilingual comparison and the wrong shape for predicting a meeting. Read speech
is clean, one speaker, no crosstalk, no chairs. Take the numbers here as an upper
bound on both engines and do not carry them over to a conversation without
checking; the ranking is more transferable than the absolute figure.

Normalisation follows Whisper's own basic normaliser: case, punctuation and
accents removed, digits left as digits. Without it half the measured error is one
engine writing "260 €" where the reference says "doscientos sesenta euros",
which is not a mistake anybody cares about.

    python tools/wer-benchmark.py --langs it es en --samples 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# FLEURS names its configurations by locale. Whisper and Apple both take the
# bare language code, so the mapping stays here rather than in either backend.
FLEURS = {"it": "it_it", "es": "es_419", "en": "en_us", "fr": "fr_fr",
          "de": "de_de", "pt": "pt_br", "nl": "nl_nl", "ca": "ca_es"}


def normalise(text: str, lang: str):
    from transformers.models.whisper.english_normalizer import (
        BasicTextNormalizer, EnglishTextNormalizer)
    if lang == "en":
        # The English normaliser also folds contractions and spelled-out numbers,
        # which is the standard for every published English WER.
        return EnglishTextNormalizer({})(text)
    return BasicTextNormalizer()(text)


def load_samples(lang: str, n: int, cache: Path) -> list[dict]:
    """n samples of FLEURS test, written to disk as wav so any engine can read them.

    Straight from the Hub's parquet export rather than through `datasets`. The
    library's streaming path sat for twenty minutes on this dataset without
    yielding a first row, and a benchmark that cannot be started is not a
    benchmark. One HTTP request per language, cached on disk after that.
    """
    import io

    import pyarrow.parquet as pq
    import soundfile as sf

    out_dir = cache / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.json"
    if manifest.exists():
        done = json.loads(manifest.read_text())
        if len(done) >= n:
            return done[:n]

    shard = out_dir / "test.parquet"
    if not shard.exists():
        import urllib.request
        url = (f"https://huggingface.co/api/datasets/google/fleurs/parquet/"
               f"{FLEURS[lang]}/test/0.parquet")
        print(f"  fetching {url}", flush=True)
        with urllib.request.urlopen(url) as r, open(shard, "wb") as f:
            f.write(r.read())

    table = pq.read_table(shard, columns=["audio", "transcription"])
    rows = []
    for i in range(min(n, table.num_rows)):
        audio = table["audio"][i].as_py()
        text = table["transcription"][i].as_py()
        samples, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        wav = out_dir / f"{i:04d}.wav"
        sf.write(wav, samples, rate)
        rows.append({"wav": str(wav), "reference": text,
                     "seconds": round(len(samples) / rate, 2)})
    manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    return rows


def run_apple(rows: list[dict], lang: str) -> tuple[list[str], float]:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scriba import apple

    said, t0 = [], time.perf_counter()
    for r in rows:
        try:
            segs = apple.transcribe(Path(r["wav"]), lang)
            said.append(" ".join(s["text"] for s in segs))
        except Exception as exc:
            print(f"    apple failed on {Path(r['wav']).name}: {exc}", file=sys.stderr)
            said.append("")
    return said, time.perf_counter() - t0


def run_whisperx(rows: list[dict], lang: str, model_name: str) -> tuple[list[str], float]:
    """scriba's own settings, not a bare whisperX.

    This used to spell the options out here, which quietly measured a different
    tool. The difference that showed it: scriba primes the decoder with a short
    sentence in the language of the recording, and on one benchmark clip that
    prompt is the whole ball game. Without it, large-v3 answered a 21-second
    passage with two words and stopped; with it, all 51 words came out. A
    benchmark of a configuration nobody runs is worse than no benchmark, because
    the number looks like it means something.
    """
    import whisperx

    from scriba.asr import build_asr_options
    from scriba.config import Settings

    s = Settings()
    s.language = lang
    s.model = model_name
    model = whisperx.load_model(
        model_name, device="cpu", compute_type="int8", language=lang,
        asr_options=build_asr_options(s),
        vad_options={"vad_onset": s.vad_onset, "vad_offset": s.vad_offset})
    said, t0 = [], time.perf_counter()
    for r in rows:
        audio = whisperx.load_audio(r["wav"])
        res = model.transcribe(audio, batch_size=s.batch_size, print_progress=False)
        said.append(" ".join(seg["text"] for seg in res["segments"]))
    return said, time.perf_counter() - t0


def per_clip(said: list[str], rows: list[dict], lang: str) -> list[dict]:
    """One row per sample, so a bad total can be pointed at rather than described.

    The totals hide the failure that matters. Spanish came out at 7.13% and it was
    not a bad transcript: it was thirty-nine good ones and a single 21-second clip
    where the model produced two words. Averaged, that reads as a mediocre engine;
    looked at per clip, it is one catastrophic dropout and nothing else.
    """
    import jiwer

    out = []
    for i, (row, hyp) in enumerate(zip(rows, said)):
        ref = normalise(row["reference"], lang)
        if not ref.strip():
            continue
        m = jiwer.process_words([ref], [normalise(hyp, lang)])
        out.append({"i": i, "wer": round(m.wer, 3), "sub": m.substitutions,
                    "del": m.deletions, "ins": m.insertions,
                    "ref_words": len(ref.split()),
                    "hyp_words": len(normalise(hyp, lang).split()),
                    "seconds": row["seconds"]})
    return out


def score(said: list[str], rows: list[dict], lang: str) -> dict:
    # jiwer.process_words, not compute_measures: the latter was removed in jiwer 4
    # and the call raised AttributeError only after the transcription had already
    # run, which on large-v3 is eight minutes of CPU thrown away per language.
    import jiwer

    refs = [normalise(r["reference"], lang) for r in rows]
    hyps = [normalise(s, lang) for s in said]
    keep = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    refs, hyps = [r for r, _ in keep], [h for _, h in keep]
    m = jiwer.process_words(refs, hyps)
    return {"wer": round(m.wer, 4), "substitutions": m.substitutions,
            "deletions": m.deletions, "insertions": m.insertions,
            "reference_words": sum(len(r.split()) for r in refs)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--langs", nargs="+", default=["it", "es", "en"])
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--engines", nargs="+", default=["apple", "whisperx"])
    ap.add_argument("--cache", type=Path, default=Path.home() / ".scriba" / "fleurs")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    results: dict[str, dict] = {}
    for lang in args.langs:
        if lang not in FLEURS:
            print(f"no FLEURS configuration mapped for {lang}", file=sys.stderr)
            continue
        print(f"\n=== {lang} ===", flush=True)
        rows = load_samples(lang, args.samples, args.cache)
        audio_seconds = sum(r["seconds"] for r in rows)
        print(f"{len(rows)} samples, {audio_seconds/60:.1f} minutes of audio", flush=True)
        results[lang] = {"samples": len(rows), "audio_seconds": round(audio_seconds, 1)}

        for engine in args.engines:
            print(f"  {engine} ...", flush=True)
            said, secs = (run_apple(rows, lang) if engine == "apple"
                          else run_whisperx(rows, lang, args.model))
            s = score(said, rows, lang)
            s["seconds"] = round(secs, 1)
            s["realtime_factor"] = round(audio_seconds / secs, 1) if secs else None
            s["per_clip"] = per_clip(said, rows, lang)
            worst = max(s["per_clip"], key=lambda c: c["del"], default=None)
            if worst and worst["del"] >= 10:
                # Name it on the spot. A number that comes from one broken clip
                # should never be read as a property of the engine.
                print(f"    one clip carries it: #{worst['i']} lost {worst['del']} "
                      f"of {worst['ref_words']} words in {worst['seconds']}s", flush=True)
            results[lang][engine] = s
            print(f"    WER {s['wer']:.1%}  ({secs:.0f}s, "
                  f"{s['realtime_factor']}x realtime)", flush=True)
            (args.out or Path("wer-results.json")).write_text(
                json.dumps(results, indent=1, ensure_ascii=False))

    print("\n" + "=" * 58)
    print(f"{'language':10s} {'engine':12s} {'WER':>8s} {'sub':>6s} {'del':>6s} {'ins':>6s}")
    for lang, r in results.items():
        for engine in args.engines:
            if engine in r:
                e = r[engine]
                print(f"{lang:10s} {engine:12s} {e['wer']:8.1%} {e['substitutions']:6d} "
                      f"{e['deletions']:6d} {e['insertions']:6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
