#!/usr/bin/env python3
"""Build a blind listening test out of the places where two engines disagree.

The question this exists to answer is which engine gets the words right, and it
is not a question either engine can answer about itself. Confidence scores do not
help: a recogniser is confident about a word it invented. Comparing to a third
model does not help either, because the obvious third model is the same weights
in a different wrapper and votes with its twin.

So the only reference worth having is a person listening to the audio. What can
be automated is everything around that: find the places where the two engines
produced different words, cut those seconds out of the recording, hide which
engine said what, and count at the end.

The output is a folder with the clips and one HTML file. It opens in a browser
with no server, keeps the answers in local storage so it can be done in two
sittings, and only reveals which engine is which once every case is decided.

Nothing here leaves the machine. The clips are pieces of the recording and the
page is a local file; it is deliberately not something to send anybody.

    python tools/compare-engines.py ~/.scriba/jobs/<job>
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

PAD = 1.2          # seconds of context on each side of a disagreement
MERGE_WITHIN = 2.0  # two disagreements closer than this become one clip
MIN_TOKENS = 1      # a one-word difference is still a word


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]", " ", text).split()


def timed(segments: list[dict]) -> list[tuple[str, str, float]]:
    """(comparable token, word as written, start) for every word with a time."""
    out = []
    for seg in segments:
        for w in (seg.get("words") or []):
            if w.get("start") is None:
                continue
            raw = str(w.get("word", ""))
            for t in tokens(raw):
                out.append((t, raw.strip(), float(w["start"])))
    return out


def disagreements(a: list, b: list) -> list[dict]:
    """Every stretch where the two word sequences differ, with a time from each."""
    sm = SequenceMatcher(None, [t for t, _, _ in a], [t for t, _, _ in b], autojunk=False)
    found = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if (i2 - i1) < MIN_TOKENS and (j2 - j1) < MIN_TOKENS:
            continue
        # A time for the clip. An insertion on one side has no time of its own,
        # so fall back to the neighbouring word on the other side.
        times = [t for _, _, t in a[i1:i2]] + [t for _, _, t in b[j1:j2]]
        if not times:
            near = a[max(i1 - 1, 0):i1] or b[max(j1 - 1, 0):j1]
            if not near:
                continue
            times = [near[-1][2]]
        found.append({
            "start": min(times), "end": max(times),
            "a": " ".join(w for _, w, _ in a[i1:i2]),
            "b": " ".join(w for _, w, _ in b[j1:j2]),
        })

    merged: list[dict] = []
    for d in sorted(found, key=lambda x: x["start"]):
        if merged and d["start"] - merged[-1]["end"] <= MERGE_WITHIN:
            m = merged[-1]
            m["end"] = max(m["end"], d["end"])
            m["a"] = f'{m["a"]} {d["a"]}'.strip()
            m["b"] = f'{m["b"]} {d["b"]}'.strip()
        else:
            merged.append(dict(d))
    return merged


def context(words: list, start: float, end: float, pad: float = 6.0) -> str:
    """What one engine wrote around a disagreement, so the clip can be followed."""
    return " ".join(w for _, w, t in words if start - pad <= t <= end + pad)


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>%(title)s</title>
<style>
 body{font:16px/1.6 -apple-system,system-ui,sans-serif;max-width:44rem;margin:2rem auto;padding:0 1rem;
      color:#111;background:#fff}
 @media(prefers-color-scheme:dark){body{color:#eee;background:#151515}
   .case{background:#1e1e1e;border-color:#333}button{background:#2a2a2a;color:#eee;border-color:#444}
   .ctx{color:#999}}
 h1{font-size:1.3rem}
 .case{border:1px solid #ddd;border-radius:10px;padding:1rem 1.2rem;margin:1.2rem 0;background:#fafafa}
 .case.done{opacity:.5}
 audio{width:100%%;margin:.4rem 0 .8rem}
 .ctx{color:#666;font-size:.85rem;margin-bottom:.6rem}
 button{display:block;width:100%%;text-align:left;font:inherit;padding:.6rem .8rem;margin:.35rem 0;
        border:1px solid #ccc;border-radius:8px;background:#fff;cursor:pointer}
 button:hover{border-color:#888}
 button.picked{border-color:#2a7;border-width:2px}
 .bar{position:sticky;top:0;background:inherit;padding:.6rem 0;border-bottom:1px solid #ddd;
      margin-bottom:1rem}
 .verdict{font-size:1.05rem;padding:1rem;border:2px solid #2a7;border-radius:10px;margin:1.5rem 0}
 small{color:#666}
</style>
<h1>%(title)s</h1>
<p>%(intro)s</p>
<div class="bar"><span id="count">0</span> di %(n)d decisi
 &nbsp;<button style="display:inline;width:auto" onclick="reveal()">Mostra il risultato</button>
 &nbsp;<button style="display:inline;width:auto" onclick="if(confirm('Ricominciare?')){localStorage.removeItem(KEY);location.reload()}">Azzera</button>
</div>
<div id="verdict"></div>
%(cases)s
<script>
const KEY = %(key)s;
const TRUTH = %(truth)s;   // per ogni caso: quale opzione appartiene a quale motore
let picks = JSON.parse(localStorage.getItem(KEY) || "{}");

function pick(i, opt){
  picks[i] = opt;
  localStorage.setItem(KEY, JSON.stringify(picks));
  paint();
}
function paint(){
  document.getElementById("count").textContent = Object.keys(picks).length;
  document.querySelectorAll(".case").forEach(c => {
    const i = c.dataset.i;
    c.classList.toggle("done", picks[i] !== undefined);
    c.querySelectorAll("button").forEach(b =>
      b.classList.toggle("picked", picks[i] === b.dataset.opt));
  });
}
function reveal(){
  const tally = {};
  let decided = 0;
  for (const [i, opt] of Object.entries(picks)){
    const who = TRUTH[i][opt];
    if (!who) continue;
    tally[who] = (tally[who] || 0) + 1;
    decided++;
  }
  const rows = Object.entries(tally).sort((a,b) => b[1]-a[1])
    .map(([who, n]) => `<div><b>${who}</b>: ${n} volte su ${decided} (${Math.round(100*n/decided)}%%)</div>`)
    .join("");
  document.getElementById("verdict").innerHTML =
    `<div class="verdict"><b>Risultato su ${decided} casi decisi di %(n)d</b>${rows}
     <div><small>"nessuno dei due" non conta per nessuno dei motori.</small></div></div>`;
  window.scrollTo(0,0);
}
paint();
</script>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("job", type=Path, help="a job folder under ~/.scriba/jobs")
    ap.add_argument("--second", type=Path, default=None,
                    help="the other engine's aligned segments (default: second-opinion-*.json)")
    ap.add_argument("--out", type=Path, default=None, help="where to write the test")
    ap.add_argument("--max", type=int, default=40, help="how many cases at most")
    ap.add_argument("--names", default="whisperX large-v3,Apple on-device",
                    help="what to call the two engines, transcript first")
    args = ap.parse_args()

    job = args.job.expanduser()
    wav = job / "audio16k.wav"
    tpath = job / "transcript.json"
    if not wav.exists() or not tpath.exists():
        print(f"{job} has no transcript to compare", file=sys.stderr)
        return 1

    second_path = args.second or next(iter(sorted(job.glob("second-opinion-*.json"))), None)
    if second_path is None or not Path(second_path).exists():
        print("no second opinion on disk. Run `scriba verify` on this recording first.",
              file=sys.stderr)
        return 1

    cached = json.loads(tpath.read_text())
    first = timed(cached["segments"] if isinstance(cached, dict) else cached)
    second = timed(json.loads(Path(second_path).read_text()))
    if not first or not second:
        print("one of the two transcripts has no per-word times", file=sys.stderr)
        return 1

    name_a, name_b = [n.strip() for n in args.names.split(",")]
    cases = disagreements(first, second)
    # Longest first: a disagreement over six words tells you more about an engine
    # than one over a comma, and forty cases is already twenty minutes of listening.
    cases.sort(key=lambda d: -(len(d["a"].split()) + len(d["b"].split())))
    cases = cases[:args.max]
    cases.sort(key=lambda d: d["start"])
    if not cases:
        print("the two engines produced the same words. Nothing to listen to.")
        return 0

    out = (args.out or job / "compare").expanduser()
    (out / "clips").mkdir(parents=True, exist_ok=True)

    blocks, truth = [], {}
    for i, d in enumerate(cases):
        a0 = max(0.0, d["start"] - PAD)
        dur = (d["end"] - d["start"]) + 2 * PAD
        clip = out / "clips" / f"{i:03d}.m4a"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a0:.2f}",
                        "-t", f"{dur:.2f}", "-i", str(wav), "-c:a", "aac", "-b:a", "64k",
                        str(clip)], check=True)

        # Which engine is shown first is decided by the case number, not by chance:
        # the page has to be reproducible, and a fixed alternation is as blind as a
        # coin toss when nobody is told the rule.
        swap = (i % 2 == 1)
        left = d["b"] if swap else d["a"]
        right = d["a"] if swap else d["b"]
        truth[str(i)] = {"1": name_b if swap else name_a,
                         "2": name_a if swap else name_b,
                         "0": ""}
        ctx = context(first, d["start"], d["end"])
        mm, ss = divmod(int(d["start"]), 60)
        blocks.append(f"""<div class="case" data-i="{i}">
 <b>{i+1}.</b> <small>a {mm:02d}:{ss:02d} della registrazione</small>
 <audio controls preload="none" src="clips/{i:03d}.m4a"></audio>
 <div class="ctx">intorno: ...{html.escape(ctx)}...</div>
 <button data-opt="1" onclick="pick({i},'1')">{html.escape(left) or "(niente)"}</button>
 <button data-opt="2" onclick="pick({i},'2')">{html.escape(right) or "(niente)"}</button>
 <button data-opt="0" onclick="pick({i},'0')"><small>nessuno dei due / non si capisce</small></button>
</div>""")

    page = PAGE % {
        "title": f"Quale dei due ha sentito giusto? · {job.name}",
        "intro": ("Ogni caso è un punto dove i due motori hanno scritto parole diverse. "
                  "Ascolta e scegli quella che è stata detta davvero. Non è scritto quale "
                  "motore ha detto cosa: si vede solo alla fine. Le risposte restano in "
                  "questo browser, si può interrompere e riprendere."),
        "n": len(cases),
        "cases": "\n".join(blocks),
        "truth": json.dumps(truth, ensure_ascii=False),
        "key": json.dumps(f"scriba-compare-{job.name}"),
    }
    (out / "index.html").write_text(page)
    print(f"{len(cases)} casi da ascoltare: {out / 'index.html'}")
    print(f"  open {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
