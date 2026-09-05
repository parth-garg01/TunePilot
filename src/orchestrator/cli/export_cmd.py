"""`orchestrator export` — export a trained model via the Soup adapter."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from ..soup.adapter import SoupAdapter

app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def export(
    model_dir: Path = typer.Argument(..., help="Trained model directory."),
    dest: Path = typer.Argument(..., help="Export destination."),
    format: str = typer.Option("safetensors"),
) -> None:
    adapter = SoupAdapter()
    if not adapter.available():
        rprint("[yellow]soup CLI not found on PATH; skipping export step[/yellow]")
        raise typer.Exit(1)
    out = adapter.export(model_dir, dest, format=format)
    rprint(f"[green]exported[/green] to {out}")
