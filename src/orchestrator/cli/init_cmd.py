"""`orchestrator init` — bootstrap a project directory + starter config."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from ..artifacts.layout import ProjectLayout
from ..config import (
    ArtifactsConfig,
    ComputeConfig,
    DatasetConfig,
    GoalConfig,
    ModelSelectionConfig,
    OrchestratorConfig,
    ParallelismConfig,
    ProjectConfig,
    SelectionConfig,
    TrainingConfig,
    save_config,
)
from ..jobs.registry import ExperimentRegistry

app = typer.Typer(invoke_without_command=True)


def init_project(
    name: str = "my-llm-project",
    root: Path = Path("./projects"),
    dataset: Path = Path("./data/train.jsonl"),
    goal: str = "Create a helpful assistant",
) -> Path:
    layout = ProjectLayout(root=root, name=name)
    layout.ensure()

    cfg = OrchestratorConfig(
        project=ProjectConfig(name=name, description=goal),
        dataset=DatasetConfig(path=str(dataset)),
        goal=GoalConfig(type="auto", description=goal),
        model_selection=ModelSelectionConfig(),
        training=TrainingConfig(),
        compute=ComputeConfig(),
        parallelism=ParallelismConfig(),
        selection=SelectionConfig(),
        artifacts=ArtifactsConfig(root=str(root)),
    )
    cfg_path = save_config(cfg, layout.project_dir / "orchestrator.yaml")

    registry = ExperimentRegistry(layout.db)
    registry.ensure_project(name, goal)
    registry.close()

    rprint(f"[green]initialized[/green] project '{name}' at {layout.project_dir}")
    rprint(f"config: {cfg_path}")
    return cfg_path


@app.callback(invoke_without_command=True)
def init_main(
    ctx: typer.Context,
    name: str = typer.Argument("my-llm-project", help="Project name."),
    root: Path = typer.Option(Path("./projects"), "--root", help="Artifact root directory."),
    dataset: Path = typer.Option(Path("./data/train.jsonl"), "--dataset"),
    goal: str = typer.Option("Create a helpful assistant", "--goal"),
) -> None:
    if ctx.invoked_subcommand is None:
        init_project(name=name, root=root, dataset=dataset, goal=goal)

