"""`orchestrator serve` — run local inference against a trained model."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from ..soup.adapter import SoupAdapter

app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def serve(
    model_dir: Path,
    prompt: str = typer.Option("Hello", "--prompt"),
) -> None:
    adapter = SoupAdapter()
    if not adapter.available():
        rprint("[yellow]soup CLI not found on PATH[/yellow]")
        raise typer.Exit(1)
    rprint(adapter.infer(model_dir, prompt))
