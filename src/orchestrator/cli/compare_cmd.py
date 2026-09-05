"""`orchestrator compare` — compare candidates from the registry."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from ..artifacts.layout import ProjectLayout
from ..config import load_config
from ..evaluation.comparison import CandidateComparison, CandidateResult
from ..jobs.registry import ExperimentRegistry

app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def compare(config: Path = typer.Option(Path("orchestrator.yaml"))) -> None:
    cfg = load_config(config)
    layout = ProjectLayout(root=Path(cfg.artifacts.root), name=cfg.project.name)
    reg = ExperimentRegistry(layout.db)
    rows: list[CandidateResult] = []
    for exp in reg.list_experiments():
        evals = reg.evaluations_for(exp.id)
        metrics = {e.metric: e.value for e in evals}
        rows.append(
            CandidateResult(
                identifier=exp.model_identifier,
                val_loss=metrics.get("val_loss"),
                task_score=metrics.get("task"),
                metrics=metrics,
            )
        )
    if not rows:
        rprint("[yellow]no evaluated candidates[/yellow]")
        raise typer.Exit(0)
    report = CandidateComparison(objective=cfg.selection.objective).compare(rows)
    rprint(report.render())
    reg.close()
