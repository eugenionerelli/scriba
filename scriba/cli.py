"""scriba command-line interface."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import Settings, keychain_set, hf_token, DATA_DIR
from .pipeline import Job
from .voices import VoiceRegistry

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="From voice memo to NotebookLM source: transcribes, separates the "
                       "voices, learns who is who.")
console = Console()


class Stage(str, Enum):
    """Stages `--force` can redo.

    An Enum rather than a string so typer rejects a typo. `--force asrr` used to
    exit 0 having forced nothing: the run reused every cached stage and looked
    exactly like a successful re-run.
    """
    all = "all"
    lang = "lang"
    asr = "asr"
    diar = "diar"


def _settings(language: str | None, model: str | None, min_s: int | None,
              max_s: int | None, no_diarize: bool) -> Settings:
    s = Settings.load()
    if language:
        s.language = language
    if model:
        s.model = model
    if min_s:
        s.min_speakers = min_s
    if max_s:
        s.max_speakers = max_s
    if no_diarize:
        s.diarize = False
    return s


@app.command()
def run(
    files: list[Path] = typer.Argument(..., help="Audio or video files to transcribe"),
    language: str = typer.Option("auto", "--lang", "-l",
                                 help="Language code (it, es, en…) or 'auto'"),
    model: str = typer.Option(None, "--model", "-m", help="large-v3, medium, small…"),
    min_speakers: int = typer.Option(None, "--min-speakers"),
    max_speakers: int = typer.Option(None, "--max-speakers"),
    no_diarize: bool = typer.Option(False, "--no-diarize", help="Transcription only"),
    force: Stage = typer.Option(None, "--force", case_sensitive=False,
                                help="redo one stage, ignoring the cache"),
):
    """Transcribe and diarize one or more files."""
    s = _settings(language, model, min_speakers, max_speakers, no_diarize)
    for f in files:
        console.rule(f"[bold]{f.name}")
        # markup=False: the engine writes notes like "[bilingual, check by hand]" and
        # Rich reads square brackets as a style tag, so the note vanished and left
        # a bare warning symbol. The language warning is the project's main safety
        # net, and it was the one line that never reached the screen.
        job = Job(f, s, report=lambda m: console.print(f"  {m}", style="dim",
                                                      markup=False, highlight=False))
        res = job.run(force=force.value if force else None)

        table = Table(show_header=True, header_style="bold")
        table.add_column("voice")
        table.add_column("name")
        table.add_column("source")
        for label in res.speakers:
            name = res.names.get(label)
            match = job.state.get("matches", {}).get(label, {})
            origin = ("voice registry" if match.get("name") == name and name
                      else "set manually" if name else "-")
            table.add_row(label, name or "[yellow]unassigned[/yellow]", origin)
        console.print(table)

        if res.unresolved:
            console.print(f"\n  Unidentified voices. Read the briefing:\n"
                          f"  [cyan]{res.dossier_path}[/cyan]\n"
                          f"  then: [bold]scriba name \"{f.name}\" SPEAKER_00=Name ...[/bold]\n")
        for o in res.outputs:
            console.print(f"  → {o}")


@app.command()
def name(
    file: Path = typer.Argument(..., help="The same audio file already processed"),
    assignments: list[str] = typer.Argument(..., help="SPEAKER_00=Marco SPEAKER_01=Anna"),
    no_enroll: bool = typer.Option(False, "--no-enroll",
                                   help="Do not save the voice print in the voice registry"),
):
    """Assign names to the voices and learn them for the next recordings."""
    mapping: dict[str, str] = {}
    for a in assignments:
        if "=" not in a:
            console.print(f"[red]invalid format: {a} (expected SPEAKER_00=Name)[/red]")
            raise typer.Exit(1)
        k, v = a.split("=", 1)
        mapping[k.strip()] = v.strip()

    job = Job(file, report=lambda m: console.print(f"  {m}", style="dim",
                                                  markup=False, highlight=False))

    # Check the labels against the ones this job actually has. Getting a label wrong
    # used to write the pair into the job state, report "0 voice prints enrolled. On
    # the next recording these names attach on their own", print Done in green and
    # exit 0. Every part of that was false.
    known = job.speaker_labels()
    if known:
        unknown = [k for k in mapping if k not in known]
        if unknown:
            console.print(f"[red]This recording has no {', '.join(unknown)}.[/red]")
            console.print(f"Voices found: {', '.join(sorted(known))}")
            raise typer.Exit(1)
    else:
        console.print("[red]This file has not been diarized yet. Run `scriba run` first.[/red]")
        raise typer.Exit(1)

    res = job.set_names(mapping, enroll=not no_enroll)
    console.print(f"\n[green]Done.[/green] {res.job_dir / 'output'}")


@app.command()
def watch(
    folder: Path = typer.Argument(..., help="Folder to watch"),
    language: str = typer.Option("auto", "--lang", "-l"),
    max_speakers: int = typer.Option(None, "--max-speakers"),
):
    """Watch a folder: every audio file that lands there is transcribed on its own."""
    from .watch import watch as _watch
    s = _settings(language, None, None, max_speakers, False)
    _watch(folder, s, report=lambda m: console.print(m, style="dim",
                                                    markup=False, highlight=False))


@app.command()
def whoami(
    folder: Path = typer.Argument(..., help="Folder of recordings to scan"),
    name: str = typer.Option(None, "--name", "-n",
                             help="Enroll the recurring voice under this name"),
    min_speech: float = typer.Option(20.0, "--min-speech",
                                     help="Ignore speakers with less speech than this"),
    include_video: bool = typer.Option(False, "--include-video",
                                       help="Also scan video files, not just voice recordings"),
):
    """Find the voice that recurs across your recordings, and calibrate on it.

    Most of your own recordings have you in them. The voice present in the most of
    them is you, and nobody has to label anything for that to work.

    Only diarization runs, never transcription, so scanning hours of audio takes
    minutes. Recordings already processed are read from the cache.
    """
    from . import recurring

    s = Settings.load()
    samples = recurring.scan(folder, s, min_speech=min_speech,
                             include_video=include_video,
                             report=lambda m: console.print(m, style="dim",
                                                            markup=False, highlight=False))
    if len(samples) < 2:
        console.print("[red]Not enough speakers found to compare.[/red]")
        raise typer.Exit(1)

    sims = recurring.cross_file_similarities(samples)
    cut, note = recurring.suggest_threshold(sims)

    console.print(f"\n[bold]{len(samples)} speakers across {len({x.file for x in samples})} "
                  f"recordings, {len(sims)} cross-recording comparisons[/bold]")
    console.print(f"  similarity: lowest {sims[0]:.3f}, median {sims[len(sims)//2]:.3f}, "
                  f"highest {sims[-1]:.3f}")
    if cut:
        console.print(f"  suggested threshold: [bold]{cut:.2f}[/bold]  ({note})")
        console.print(f"  currently set to {s.voice_match_threshold:.2f}")
    else:
        console.print(f"  {note}")

    threshold = cut or s.voice_match_threshold
    try:
        groups = recurring.cluster(samples, threshold)
    except ValueError as exc:
        console.print(f"\n[yellow]Cannot group these voices: {exc}[/yellow]")
        console.print("Either no voice recurs across these recordings, or they were "
                      "made in conditions too different to link up.")
        raise typer.Exit(0)
    top = groups[0] if groups else None
    if top is None or len(top.files) < 2:
        console.print("\n[yellow]No voice appears in more than one recording.[/yellow]")
        raise typer.Exit(0)

    table = Table(show_header=True, header_style="bold")
    for col in ("voice", "recordings", "speech", "listen to"):
        table.add_column(col)
    for i, g in enumerate(groups[:5], 1):
        if len(g.files) < 2:
            continue
        best = max(g.samples, key=lambda x: x.speech_seconds)
        table.add_row(
            f"{i}" + ("  (most recurring)" if i == 1 else ""),
            str(len(g.files)),
            f"{g.speech_seconds / 60:.0f} min",
            f"{best.file.name} at {int(best.longest_start // 60)}:"
            f"{int(best.longest_start % 60):02d}",
        )
    console.print(table)

    console.print(f"\nThe most recurring voice is in [bold]{len(top.files)} of "
                  f"{len({x.file for x in samples})}[/bold] recordings.")
    for sample in sorted(top.samples, key=lambda x: -x.speech_seconds)[:6]:
        console.print(f"  {sample.file.name}  {sample.label}  "
                      f"{sample.speech_seconds / 60:.0f} min  "
                      f"listen at {int(sample.longest_start // 60)}:"
                      f"{int(sample.longest_start % 60):02d}",
                      style="dim", markup=False, highlight=False)

    if not name:
        console.print("\nListen to a couple of those, and if that voice is you:")
        console.print(f'  [bold]scriba whoami "{folder}" --name "Your name"[/bold]')
        return

    reg = VoiceRegistry()
    added = 0
    for sample in top.samples:
        reg.enroll(name, sample.embedding, source=sample.file.name)
        added += 1
    reg.save()
    console.print(f"\n[green]Enrolled {name}[/green] from {added} recordings. "
                  "Prints from different days and rooms are what make the matching hold up.")


@app.command()
def info(file: Path):
    """Job status as JSON. This is the channel the macOS app talks through."""
    job = Job(file)
    turns_path = job.dir / "turns.json"
    payload = {
        "job_dir": str(job.dir),
        "source": str(job.source),
        "state": job.state,
        "turns": json.loads(turns_path.read_text()) if turns_path.exists() else [],
        "outputs": sorted(str(p) for p in (job.dir / "output").glob("*")) ,
        "dossier": str(job.dir / "who-is-who.md"),
        "audio": str(job.wav),
    }
    print(json.dumps(payload, ensure_ascii=False, default=str))


@app.command()
def dossier(file: Path):
    """Print the "who is who" briefing for a recording that has already been processed."""
    job = Job(file)
    path = job.dir / "who-is-who.md"
    if not path.exists():
        console.print("[red]No briefing yet: run `scriba run` first.[/red]")
        raise typer.Exit(1)
    console.print(path.read_text(), markup=False, highlight=False)


voices_app = typer.Typer(help="The voice print registry.")
app.add_typer(voices_app, name="voices")


@voices_app.command("list")
def voices_list():
    """Who is in the registry."""
    reg = VoiceRegistry()
    rows = reg.summary()
    if not rows:
        console.print("Registry empty. It fills itself as you use [bold]scriba name[/bold].")
        return
    table = Table(show_header=True, header_style="bold")
    for col in ("name", "aliases", "prints", "recordings", "updated"):
        table.add_column(col)
    for r in rows:
        table.add_row(r["name"], r["aliases"], str(r["prints"]),
                      str(r["recordings"]), r["updated"][:10])
    console.print(table)


@voices_app.command("forget")
def voices_forget(name: str):
    """Remove a person from the registry."""
    reg = VoiceRegistry()
    if reg.forget(name):
        reg.save()
        console.print(f"[green]Removed: {name}[/green]")
    else:
        console.print(f"[yellow]Not found: {name}[/yellow]")


@voices_app.command("rename")
def voices_rename(old: str, new: str):
    """Change a person's name without losing the voice prints."""
    reg = VoiceRegistry()
    if reg.rename(old, new):
        reg.save()
        console.print(f"[green]{old} → {new}[/green]")
    else:
        console.print(f"[yellow]Not found: {old}[/yellow]")


@app.command()
def token(value: str = typer.Argument(None, help="Hugging Face token (hf_...)")):
    """Save the pyannote token in the Keychain (never in plain text in a file)."""
    if value is None:
        current = hf_token()
        console.print("Token present." if current else "No token configured.")
        console.print("Usage: [bold]scriba token hf_xxxxxxxx[/bold]")
        return
    keychain_set(value.strip())
    console.print("[green]Saved in the Keychain.[/green]")


@app.command()
def settings(
    show: bool = typer.Option(True, "--show/--no-show"),
    set_: list[str] = typer.Option(None, "--set", help="key=value"),
):
    """Read and write the settings."""
    s = Settings.load()
    for item in (set_ or []):
        k, v = item.split("=", 1)
        if not hasattr(s, k):
            console.print(f"[red]unknown setting: {k}[/red]")
            raise typer.Exit(1)
        cur = getattr(s, k)
        if isinstance(cur, bool):
            v2: object = v.lower() in {"1", "true", "si", "sì", "yes"}
        elif isinstance(cur, int) and not isinstance(cur, bool):
            v2 = int(v)
        elif isinstance(cur, float):
            v2 = float(v)
        elif isinstance(cur, list):
            v2 = [x.strip() for x in v.split(",") if x.strip()]
        else:
            v2 = v
        setattr(s, k, v2)
    if set_:
        s.save()
    if show:
        console.print_json(json.dumps(asdict(s), ensure_ascii=False))
        console.print(f"[dim]{DATA_DIR}[/dim]")


def main() -> None:
    """Run the app, and turn the failures we expect into one readable line.

    Without this, each of these arrives as a forty-line rich traceback with the
    useful sentence somewhere in the middle. The missing-token message especially
    is written to tell you exactly which command to run, and it was landing at the
    bottom of a decorated stack dump where nobody would read it.

    Anything not listed here still raises with its traceback, which is right for a
    bug: an unexpected failure should look unexpected.
    """
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow] Finished stages stay cached.")
        sys.exit(130)
    except FileNotFoundError as exc:
        console.print(f"[red]File not found:[/red] {exc.filename or exc}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        console.print(f"[red]A cached file is corrupted:[/red] {exc}")
        console.print("Delete that job folder under ~/.scriba/jobs and run it again.")
        sys.exit(1)
    except ValueError as exc:
        console.print(str(exc), style="red", markup=False, highlight=False)
        sys.exit(1)
    except RuntimeError as exc:
        # The engine raises RuntimeError for failures it can explain, with the
        # explanation already written. Print it and get out of the way.
        console.print(str(exc), style="red", markup=False, highlight=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
