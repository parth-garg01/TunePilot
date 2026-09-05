"""Top-level CLI (PRD section 7).

Uses Typer. Every command is a thin adapter that dispatches into the modules
in `orchestrator.*`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

from .. import __version__
from ..config import default_config_path, load_config
from ..logging_utils import configure, get_logger
from . import (
    init_cmd,
    config_cmd,
    dataset_cmd,
    models_cmd,
    hardware_cmd,
    plan_cmd,
    train_cmd,
    jobs_cmd,
    experiments_cmd,
    evaluate_cmd,
    compare_cmd,
    export_cmd,
    serve_cmd,
    clean_cmd,
    doctor_cmd,
    run_cmd,
)

app = typer.Typer(help="Autonomous LLM Fine-Tuning Orchestrator")
log = get_logger(__name__)


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to orchestrator.yaml"),
    project: str = typer.Option("my-llm-project", "--project", "-p", help="Active project name"),
) -> None:
    configure(level="DEBUG" if verbose else ("WARNING" if quiet else "INFO"))
    ctx.obj = {"config_path": config or default_config_path()}
    if ctx.invoked_subcommand is None:
        from ..terminal.session import TerminalChatSession
        session = TerminalChatSession(project_name=project)
        session.run_loop()


@app.command("chat")
def chat(
    project: str = typer.Option("my-llm-project", "--project", "-p", help="Active project name"),
) -> None:
    """Launch interactive terminal chatbot session."""
    from ..terminal.session import TerminalChatSession
    session = TerminalChatSession(project_name=project)
    session.run_loop()


@app.command()
def version() -> None:
    """Show installed orchestrator version."""
    rprint(f"TunePilot (soup-orchestrator) {__version__}")


@app.command("init")
def init(
    name: str = typer.Argument("my-llm-project", help="Project name."),
    root: Path = typer.Option(Path("./projects"), "--root", help="Artifact root directory."),
    dataset: Path = typer.Option(Path("./data/train.jsonl"), "--dataset"),
    goal: str = typer.Option("Create a helpful assistant", "--goal"),
) -> None:
    """Initialize a project directory + config."""
    init_cmd.init_project(name=name, root=root, dataset=dataset, goal=goal)

app.add_typer(config_cmd.app, name="config", help="Show or validate configuration.")
app.add_typer(dataset_cmd.app, name="dataset", help="Analyze datasets.")
app.add_typer(models_cmd.app, name="models", help="Discover and rank models.")
app.add_typer(hardware_cmd.app, name="hardware", help="Inspect local hardware and remote profiles.")
app.add_typer(plan_cmd.app, name="plan", help="Plan pilot + full experiments.")
app.add_typer(train_cmd.app, name="train", help="Submit training jobs.")
app.add_typer(jobs_cmd.app, name="jobs", help="Inspect jobs.")
app.add_typer(experiments_cmd.app, name="experiments", help="Inspect experiments.")
app.add_typer(evaluate_cmd.app, name="evaluate", help="Run evaluation.")
app.add_typer(compare_cmd.app, name="compare", help="Compare candidates.")
app.add_typer(export_cmd.app, name="export", help="Export a trained model.")
app.add_typer(serve_cmd.app, name="serve", help="Serve a trained model locally.")
app.add_typer(clean_cmd.app, name="clean", help="Clean project artifacts.")
app.add_typer(doctor_cmd.app, name="doctor", help="Run environment diagnostics.")
app.add_typer(run_cmd.app, name="run", help="One-shot end-to-end pipeline.")


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

