"""`orchestrator experiments` — list experiments."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from ..artifacts.layout import ProjectLayout
from ..config import load_config
from ..jobs.registry import ExperimentRegistry

app = typer.Typer()


@app.command("list")
def list_experiments(config: Path = typer.Option(Path("orchestrator.yaml"))) -> None:
    cfg = load_config(config)
    layout = ProjectLayout(root=Path(cfg.artifacts.root), name=cfg.project.name)
    reg = ExperimentRegistry(layout.db)
    for e in reg.list_experiments():
        rprint(f"[{e.id}] {e.name}  status={e.status}  model={e.model_identifier}  ds={e.dataset_fingerprint[:12]}")
    reg.close()
