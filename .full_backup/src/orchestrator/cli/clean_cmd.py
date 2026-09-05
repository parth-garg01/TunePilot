"""`orchestrator clean` — remove local artifacts. Requires --yes for destructive ops."""

from __future__ import annotations

import shutil
from pathlib import Path

import typer
from rich import print as rprint

from ..artifacts.layout import ProjectLayout
from ..config import load_config

app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def clean(
    config: Path = typer.Option(Path("orchestrator.yaml")),
    what: str = typer.Option("logs", help="One of: logs, checkpoints, models, all"),
    yes: bool = typer.Option(False, "--yes"),
) -> None:
    if not yes:
        rprint("[red]refusing destructive delete without --yes[/red]")
        raise typer.Exit(2)
    cfg = load_config(config)
    layout = ProjectLayout(root=Path(cfg.artifacts.root), name=cfg.project.name)
    targets = {
        "logs": [layout.logs],
        "checkpoints": [layout.checkpoints],
        "models": [layout.models],
        "all": [layout.logs, layout.checkpoints, layout.models, layout.evaluations, layout.jobs],
    }.get(what, [])
    for t in targets:
        if t.exists():
            shutil.rmtree(t)
            t.mkdir(parents=True, exist_ok=True)
            rprint(f"cleared {t}")
