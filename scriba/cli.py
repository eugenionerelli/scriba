"""scriba command-line interface."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
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
    force: str = typer.Option(None, "--force",
                              help="all | lang | asr | diar: redo one stage, ignoring the cache"),
):
    """Transcribe and diarize one or more files."""
    s = _settings(language, model, min_speakers, max_speakers, no_diarize)
    for f in files:
        console.rule(f"[bold]{f.name}")
        job = Job(f, s, report=lambda m: console.print(f"  [dim]{m}[/dim]"))
        res = job.run(force=force)

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

    job = Job(file, report=lambda m: console.print(f"  [dim]{m}[/dim]"))
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
    _watch(folder, s, report=lambda m: console.print(f"[dim]{m}[/dim]"))


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
    console.print(path.read_text())


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
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
