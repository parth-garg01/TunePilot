"""`orchestrator jobs` — inspect job state."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from ..artifacts.layout import ProjectLayout
from ..config import load_config
from ..jobs.registry import ExperimentRegistry
from ..reporting.status import format_job_table

app = typer.Typer()


def _registry(config: Path) -> ExperimentRegistry:
    cfg = load_config(config)
    layout = ProjectLayout(root=Path(cfg.artifacts.root), name=cfg.project.name)
    return ExperimentRegistry(layout.db)


@app.command("status")
def status(config: Path = typer.Option(Path("orchestrator.yaml"))) -> None:
    reg = _registry(config)
    jobs = reg.list_jobs()
    rprint(format_job_table(jobs))
    reg.close()


@app.command("cancel")
def cancel(job_id: int, config: Path = typer.Option(Path("orchestrator.yaml"))) -> None:
    reg = _registry(config)
    from ..jobs.records import JobStatus
    reg.set_job_status(job_id, JobStatus.CANCELLED, error="cancelled by user")
    reg.close()
    rprint(f"cancelled job {job_id}")
