<div align="center">

# scriba

**Recorded conversations become a document that says who said what.**

Transcribes, separates the voices, and remembers a voice once you have named it.
Everything runs on your own machine.

[![macOS](https://img.shields.io/badge/macOS-14%2B-black?logo=apple&logoColor=white)](#the-app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#install)
[![Licence](https://img.shields.io/badge/licence-MIT-green)](#licence)
[![Built on](https://img.shields.io/badge/built%20on-whisperX%20%2B%20pyannote-orange)](#speed-and-why-it-is-what-it-is)

<img src="docs/img/04-who-is-speaking.png" width="860" alt="Scriba showing two speakers, both recognised from the voice registry, with confidence scores and a quote from each">

</div>

---

Diarization tells you there were two people. It will not tell you that the second one
is Dana, and the next recording will number them differently anyway. scriba keeps the
voice prints that pyannote computes and throws away, files them under the name you
give, and matches them the next time.

```bash
scriba run meeting.m4a
scriba name meeting.m4a SPEAKER_00=Ada SPEAKER_01=Rafiq
scriba run tomorrow.m4a          # Ada and Rafiq are picked out on their own
```

The screenshots below use two recordings that are entirely synthetic: the dialogue is
invented and the voices are macOS speech synthesis. `python tools/make-demo.py`
rebuilds them, so you can reproduce the whole flow without a real conversation.

## What it looks like

Drop a file in and nothing happens until you say so. Transcription runs for about as
long as the recording lasts, which is not something to start by accident.

<img src="docs/img/02-before-you-start.png" width="800" alt="A queued recording with the language and speaker-count options, and a Transcribe button">

The two settings above the button are the ones worth touching. Language is left to
detection unless you know better, and the number of people in the room changes the
result more than anything else here.

<img src="docs/img/03-running.png" width="800" alt="The seven stages of a run, with the finished ones ticked and the engine's own log underneath">

Seven stages, each ticked as it finishes, with the engine's own output underneath. The
window can be closed and the work carries on.

<img src="docs/img/01-empty.png" width="800" alt="The empty window listing previously processed recordings in the sidebar">

## Why this exists

Record a conversation, transcribe it, hand the text to whatever you use for reading
and searching. The flow is obvious. Doing it for real breaks in four places.

### 1. The wrong language raises no error, it invents a text

The bug this project was born from. A script with a hardcoded `--language it`, handed a
recording that was actually in Spanish. Whisper did not fail. It guessed, and produced
fluent, punctuated, entirely made-up Italian.

The two languages are close enough that guessing produces real words. Spanish and
Italian are full of pairs that sound alike and mean different things, so the output
lands on the wrong one and reads perfectly:

| Spanish | Heard as Italian | Which means |
|---|---|---|
| `salir` (to leave) | `salire` | to go up |
| `carta` (letter) | `carta` | paper |
| `burro` (donkey) | `burro` | butter |
| `éxito` (success) | `esito` | outcome |

Those four are the textbook pairs, put here to show the shape of the failure. What it
does to a real recording is worse, because it does it to every sentence at once and
each one comes out plausible.

Nothing on screen says anything is wrong. The text reads as correct, and you then
reason on top of it for weeks.

Left alone, Whisper picks the language from the first 30 seconds, which in a real
conversation is small talk. [`lang.py`](scriba/lang.py) samples five windows across the
whole file and votes, weighting each window by confidence, and says so when the vote is
close instead of choosing in silence.

It also says so when the windows agree and none of them was sure. Agreement on its own
used to be the whole confidence, which is a share: five windows that all say Galician
at 30% agree perfectly, and the file came out reported at 100% confidence. Now the
strength of the winning windows is part of the number, and a unanimous shrug is called
one.

### 2. Speaker labels restart from zero every time

Diarization hands you `SPEAKER_00`, `SPEAKER_01`. Those labels are arbitrary, and they
are reassigned on every file: the same person is `SPEAKER_00` on Monday and
`SPEAKER_02` on Tuesday. So you rename them by hand, every time, forever.

pyannote computes a 256-dimensional embedding per speaker anyway, since that is how it
decides who is who. whisperX discards it by default; since 3.8.6, the version pinned
here, `return_embeddings=True` hands it back. So the embeddings on their own no longer
argue for going around whisperX.

[`diarize.py`](scriba/diarize.py) calls pyannote directly all the same, and now for a
different reason: whisperX's diarization wrapper runs against pyannote 4 and nothing
else, and it drops the overlap-free segmentation pyannote 4 also returns, with no flag
to ask for it back. Calling pyannote directly keeps one file working on pyannote 3 and 4
alike. The vectors survive either way, and [`voices.py`](scriba/voices.py) files them
under the name you assign. From then on the name attaches itself.

There are three zones rather than two:

| Cosine similarity | What happens |
|---|---|
| 0.75 and above | the name is applied |
| 0.55 to 0.75 | "Maybe *Ada*?" A suggestion. Confirming stays your job. |
| below 0.55 | new voice |

There is also a margin of 0.05 over the runner-up: when two people on file come out
almost equally close, nothing is chosen. A wrong name raises no error either, and it
quietly poisons every transcript downstream, so an unanswered question is the better
failure.

The thresholds are a starting point to calibrate, not a published constant. To set them
properly: hand-label three or four conversations, find where the same-speaker and
different-speaker similarity distributions cross, put the threshold there. Twenty
minutes of annotated audio tells you more than any benchmark, and for diarization in
Italian there is no benchmark to argue with.

### 3. Whisper's "segments" are not conversational turns

Whisper cuts roughly every 30 seconds no matter who is speaking. On a seven-minute test
file that gave 16 blocks, each one holding more than one speaker. scriba realigns at
word level and reassembles real turns: same file, 41 turns. That is the difference
between a list of fragments and something a person, or a model, can read.

### 4. What reads the transcript reads prose

SRT and VTT shatter every sentence into two-second blocks with sequence numbers and
timecodes. All of that is noise that eats context and makes retrieval worse. JSON is
worse still.

The default output is one Markdown document: a header with participants, duration and
language, then turns with the name in bold and a timestamp at the start of each.

Voices that could not be identified are marked as unidentified inside the document
itself, on every line and not only in the header. Leave that out and "Voice 2" gets
read as a person somebody knows, and confident answers get built on top of it.

```markdown
# Team sync, 18 July

## Overview
- **Duration**: 00:47:12
- **Participants**: Ada (18:22 of speech), Rafiq (14:05 of speech)
- **Unidentified voices**: Voice 3. A distinct person, but their name is
  never spoken in the recording. Do not guess who they are.

## Transcript

**Ada** [00:12]: Right, so where did we land on the migration?

**Rafiq** [00:31]: Staging is done. Production waits for the backup window.
```

Five other formats come out of the same run for the tools that want them: plain
Markdown, text, SRT, VTT and JSON.

## Install

macOS with Apple Silicon, Python 3.10 or newer, `ffmpeg`, and a Hugging Face account
for the pyannote models. Set aside a few gigabytes and a few minutes: the
dependencies pull in torch, and the first run downloads the `large-v3` weights.

```bash
brew install ffmpeg
```

The Python side wants an environment of its own, because whisperX and pyannote pin a
torch that has no business on the system interpreter. Conda if you have it:

```bash
conda create -n scriba python=3.11 -y && conda activate scriba
pip install git+https://github.com/eugenionerelli/scriba
```

A plain virtual environment does the same job:

```bash
python3 -m venv ~/.venvs/scriba && source ~/.venvs/scriba/bin/activate
pip install git+https://github.com/eugenionerelli/scriba
```

Either way the `scriba` command lives inside that environment and nowhere else, so
every use starts by activating it. Forget that step and the shell says
`command not found`, which is the least informative thing it could say. A launcher on
the PATH is worth the thirty seconds:

```bash
mkdir -p ~/.local/bin
cat > ~/.local/bin/scriba <<'EOF'
#!/bin/zsh
# Set SCRIBA_BIN to the scriba inside your environment if none of these is right.
ENV_NAME="${SCRIBA_ENV:-scriba}"
for candidate in "$SCRIBA_BIN" \
                 "$HOME/.venvs/$ENV_NAME/bin/scriba" \
                 "$HOME/miniforge3/envs/$ENV_NAME/bin/scriba" \
                 "$HOME/miniconda3/envs/$ENV_NAME/bin/scriba" \
                 "$HOME/anaconda3/envs/$ENV_NAME/bin/scriba" \
                 "/opt/homebrew/Caskroom/miniforge/base/envs/$ENV_NAME/bin/scriba"; do
    [[ -x "$candidate" ]] && exec "$candidate" "$@"
done
print -u2 "scriba: nothing called '$ENV_NAME' in any of the usual places."
print -u2 "        Set SCRIBA_BIN to the scriba inside your environment."
exit 1
EOF
chmod +x ~/.local/bin/scriba
```

Check that `~/.local/bin` is on your `PATH`, and `scriba` works from any shell. The
loop matters more than it looks: the path to a conda environment depends on how conda
was installed, and a launcher that hardcodes one of them fails with a message that
never mentions scriba.

The pyannote token goes into the Keychain rather than into a file:

```bash
scriba token hf_xxxxxxxx
```

Then accept the conditions on
[speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1),
[segmentation-3.0](https://hf.co/pyannote/segmentation-3.0) and
[speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1),
using the same account as the token. Skip that and the download fails with a bare 403
that explains nothing. The third one matters even if you never ask for community-1:
under pyannote 4, loading 3.1 pulls one of its files from that repository, so a single
unaccepted licence blocks both models.

### A note on telemetry

pyannote 4 ships a telemetry module. Its `config.yaml` sets `metrics_enabled: true`,
and it reports to an endpoint at pyannote.ai on every model load and every file
processed. scriba turns it off in `scriba/__init__.py`, before any import path can
reach pyannote, because the flag is read once at import time and setting it afterwards
does nothing. If you use pyannote directly for recordings of private conversations, set
`PYANNOTE_METRICS_ENABLED=false` yourself.

## Use

```bash
# the language is worked out on its own
scriba run recording.m4a

# if you know how many people are in the room, saying so helps a lot
scriba run meeting.m4a --lang es --min-speakers 2 --max-speakers 2

# who is who: how much each one talks, their longest turns,
# and the moments where somebody says a name out loud
scriba dossier meeting.m4a

# assign the names, and from here on they are remembered
scriba name meeting.m4a SPEAKER_00=Ada SPEAKER_01=Rafiq

scriba show meeting.m4a               # where the document went, and what else was written
scriba voices list                    # who is on file
scriba watch ~/Memos                  # transcribes every audio file that lands there
scriba jobs list                      # what has been processed, and how far it got
```

`show --reveal` opens the folder in the Finder. It exists because a run prints its
paths and then scrolls away, and a week later the job folder is a slug with a hash
on the end.

`watch` keeps its bookkeeping in a `.scriba-done` folder inside the folder it is
watching, and a file that failed is recorded there too, so it is not retried on the
next pass. Delete its entry to try it again.

`jobs list` exists because the job folder is not somewhere anybody browses. The names
are slugs with a hash on the end, and "did I ever transcribe that one" otherwise has no
answer short of opening files. It also shows what the disk is holding: each job keeps a
full 16 kHz copy of the audio, about 2 MB a minute, which `jobs prune --audio` gives
back. Transcripts and voice prints cost CPU to make and are never touched by it.

Every stage is cached under `~/.scriba/jobs/<name>/`, so redoing just the names
re-transcribes nothing. Pass `--force asr|diar|lang|all` to rerun the stage you want.

### Finding your own voice

Most of your own recordings have you in them, and that fact does two jobs at once.

```bash
scriba whoami ~/Recordings
```

This diarizes every recording in the folder and transcribes none of them, which is what
makes it practical: working out who is present needs embeddings rather than words, and
on Metal that runs at about a tenth of realtime. Hours of audio take minutes.

It then compares speakers *across* recordings, ignoring pairs from inside the same one.
Two people in the same room were separated by the diarizer because they sound
different, so those pairs say nothing about whether one person sounds like themselves
on another day, which is the question that matters here.

The voice present in the most recordings is whoever owns the microphone. You get told
which one that is, with a timestamp in each recording to listen to, and nothing is
written until you confirm:

```bash
scriba whoami ~/Recordings --name "Your name"
```

That enrolls every print of that voice at once, from different days and different
rooms, which is what makes later matching hold up. A registry built from one recording
only ever proves that a file matches itself.

The same scan answers the calibration question from earlier. Comparing two recordings
of the same person made on different days is the honest way to set a threshold, and a
folder of recordings contains hundreds of those comparisons already. `whoami` prints
where the two populations separate in your own material, so the number is measured
rather than assumed.

## The app

```bash
cd macapp && ./build.sh && open Scriba.app
```

SwiftUI with no outside dependencies and no `.xcodeproj`. SwiftPM builds the executable
and the script assembles the bundle around it, so the whole thing stays in git as text
and rebuilds from a terminal.

The app implements none of the work itself. It runs the same `scriba` and reads its
JSON state, which keeps the engine usable on its own and leaves no second copy of the
logic to drift apart a day later.

One rule for the interface: every row in the speaker table has a play button next to
the name field. If assigning a name costs less than checking it, people will assign
without checking. Voices with very little speech are flagged as probable artefacts,
which is usually what they are, somebody else's murmured agreement promoted to a
person.

On first run, open Settings. The app needs the Python of the environment where you
installed `scriba`, and both fields say whether what you typed is there. The second
field, the package folder, only matters when you are running from a clone; a pip
install can leave it alone.

A run can be stopped from the strip under the sidebar or from the Transcribe menu,
and quitting the app stops the engine with it. The keys are the ones a Mac
application puts them on: `cmd O` to add recordings, `cmd T` to start the queue,
`cmd .` to stop, `cmd R` to re-read the list.

The window stays usable while a transcription runs. That took saying twice, because
the first version handed the whole detail pane to the progress display, and a
transcription lasts about as long as the recording does. The strip says what is
happening from wherever you are in the app; the pane is yours.

## Speed, and why it is what it is

`ctranslate2`, the engine under faster-whisper, has no Metal backend, so on Apple
Silicon it runs entirely on the CPU. Measured on an M4 with 16 GB, the same 6:45 of
Spanish audio through both versions, same settings, warm model cache:

| Stage | whisperX 3.3.1 | whisperX 3.8.6 |
|---|---|---|
| Load `large-v3` int8 | 9.1s | 34.7s |
| Transcription, 8 threads | 601.8s | 443.1s |
| Load the aligner | 0.5s | 2.8s |
| Forced alignment, word level | 13.3s | 14.1s |
| Diarization, pyannote 3.1 on CPU | 372.3s | 211.0s |
| Diarization, community-1 on CPU | | 225.9s |
| Diarization, community-1 on Metal | | **35.7s** |
| **Total, best of each** | **16m 37s** | **8m 13s** |

Transcription and diarization are where the time goes, and no setting changes that: it
is the library. The job cache exists for this reason, so you pay it once. Alignment,
the stage that sounds expensive, costs 14 seconds.

The newer whisperX is 26% faster on the transcription, which is not the reason to move
to it. In 3.3.1 every token containing digits came out of the aligner with no timestamp
at all, 5 out of 5 on the test file, three of them amounts of money. In 3.8.6 it is 0
out of 7, and two words the old version dropped from the transcript altogether now
appear. On a recording where the numbers are the content, those are the words you least
want to lose.

For contrast, Apple's own on-device recogniser via
[yap](https://github.com/finnvoor/yap) does the same file in 24 seconds, about 17 times
realtime. It picks up the opening small talk that Whisper's VAD skips over. It garbles
surnames more, splitting one of them into two words that are not words. And it has no
diarization at all, so on its own it does not cover this job.

What helps, in order:

| Move | Effect | Risk |
|---|---|---|
| `--min-speakers` / `--max-speakers` when you know them | No speed-up. It is the one parameter that moves correctness the most | none |
| `large-v3-turbo` in place of `large-v3` | About twice as fast | loses a little on noisy audio |
| Metal for the diarization, which is now the default | 226s to 36s, so 6.3x | Verified identical to the CPU output before the default changed: same 165 turns, same labels, every boundary matching to the millisecond. [#1337](https://github.com/pyannote/pyannote-audio/issues/1337) reports wrong timestamps under Metal and was closed with no fix, so it is worth knowing the shape of that failure. If one speaker ever owns the whole recording, set `diarize_device=cpu` and compare |

Things that are not worth it, checked rather than assumed: `mlx-whisper` implements no
beam search, so moving to it trades quality away against `beam_size=5`.
`lightning-whisper-mlx` and `stable-ts` are both stalled. Apple's `SpeechAnalyzer` API
is fast and exposes no diarization whatsoever, so it solves half the problem at best.

## Layout

```
scriba/
  lang.py       language detection over several samples
  asr.py        whisper, tuned for conversational speech
  diarize.py    pyannote called directly, on either major version
  voices.py     the voice-print registry
  naming.py     the "who is who" briefing: textual cues plus registry matches
  export.py     output formats, the readable document first
  pipeline.py   the orchestrator, cached per stage
  watch.py      watched folder
macapp/         the SwiftUI app
tools/          stylecheck.py, make-demo.py, and the writing rules this repo is held to
```

## Notes

Comments in the code say why, not what. Where there is a magic number or a choice that
reads oddly, the reason sits next to it, usually a bug already paid for.

The tests live in `tests/` and run in about eight seconds, because none of them
load a model:

```bash
pytest
```

They were written by a set of agents pointed at one module each and told to report
defects rather than fix them, which turned up around twenty. The interesting ones
are in the commit log: a registry save that could hand one person's voice print to
another, a cache that kept the previous recording's speaker names, a language vote
that reported a file nobody could identify as its most confident result.

`tools/stylecheck.py` holds the writing rules for this repo, prose and comments alike.
Run both checks before a commit:

```bash
python tools/stylecheck.py --code README.md scriba/*.py macapp/Sources/Scriba/*.swift
python tools/check-output-style.py                   # what the source produces
```

`--code` restricts the check to comments, docstrings and strings shown to a person.
Without it the banned word list is applied to identifiers as well, and every sort
in the codebase comes back as a finding for the argument it sorts by.

The second one exists because the first is not enough. Every document scriba writes is
assembled from string fragments, so each line can pass on its own and the finished file
still be wrong. `check-output-style.py` renders all six output formats plus the
briefing from stand-in data and checks those instead.

Written in one sitting, paired with Claude, starting from a real problem: a Spanish
recording transcribed into Italian that looked flawless and was not. The code was read,
run against real audio, and corrected. The numbers here come from measurements. It is
still an afternoon's worth of code, so take it for what it is.

## Licence

MIT.
