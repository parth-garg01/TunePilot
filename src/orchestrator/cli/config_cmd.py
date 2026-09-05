"""`orchestrator config` — show / validate configuration."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from ..config import load_config

app = typer.Typer()


@app.command("show")
def show(config: Path = typer.Option(Path("orchestrator.yaml"))) -> None:
    cfg = load_config(config)
    rprint(cfg.model_dump(mode="json"))


@app.command("validate")
def validate(config: Path = typer.Option(Path("orchestrator.yaml"))) -> None:
    cfg = load_config(config)
    rprint(f"[green]config OK[/green]: {cfg.project.name}")
