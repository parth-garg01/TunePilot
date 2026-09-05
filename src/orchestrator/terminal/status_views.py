"""Rich Status Views, Formatted Tables, and Live Streaming Displays (PRD Section 14 & 17, Feature Update)."""

from __future__ import annotations

import time
from typing import Any, Iterable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


console = Console()


def print_banner(project_name: str, status: str = "Ready") -> None:
    banner_text = Text()
    banner_text.append("TunePilot\n", style="bold cyan")
    banner_text.append("Autonomous LLM Fine-Tuning Assistant", style="dim white")

    panel = Panel(
        banner_text,
        border_style="cyan",
        title="[bold green]⚡ September 5 Edition[/bold green]",
        subtitle=f"Project: [bold]{project_name}[/bold] | Status: [green]{status}[/green]",
    )
    console.print(panel)


def print_welcome_back(project_name: str, active_jobs: list[dict[str, Any]]) -> None:
    console.print(f"\n[bold green]Welcome back to project '{project_name}'.[/bold green]")
    if active_jobs:
        table = Table(title="Active Experiments", border_style="cyan", show_header=True)
        table.add_column("Job ID", style="bold cyan", width=10)
        table.add_column("Experiment", style="white")
        table.add_column("Backend", style="dim")
        table.add_column("Status", style="bold green")

        for j in active_jobs:
            table.add_row(
                f"job-{j.get('job_id', 0):03d}",
                f"exp-{j.get('experiment_id', 0):03d}",
                str(j.get("backend", "kaggle")),
                str(j.get("status", "RUNNING")),
            )
        console.print(table)
    else:
        console.print("[dim]No active background jobs running.[/dim]\n")


def format_models_table(models: list[dict[str, Any]]) -> Table:
    table = Table(title="Discovered Model Candidates", border_style="cyan", show_header=True)
    table.add_column("#", style="bold yellow", width=4)
    table.add_column("Model Identifier", style="bold white")
    table.add_column("Family", style="dim")
    table.add_column("Params", justify="right")
    table.add_column("Context", justify="right")
    table.add_column("Score", justify="right", style="green")
    table.add_column("License", style="dim")

    for i, m in enumerate(models, 1):
        params_str = f"{m.get('parameters', 0) / 1e9:.1f}B" if m.get("parameters") else "-"
        ctx_str = f"{m.get('context_length', 0):,} tokens" if m.get("context_length") else "-"
        table.add_row(
            str(i),
            m.get("identifier", "-"),
            m.get("family", "-"),
            params_str,
            ctx_str,
            f"{m.get('score', 0):.2f}",
            m.get("license", "-"),
        )
    return table


def format_jobs_table(jobs: list[dict[str, Any]]) -> Table:
    table = Table(title="Jobs & Experiments Status", border_style="cyan", show_header=True)
    table.add_column("ID", style="bold cyan", width=8)
    table.add_column("Experiment", style="white")
    table.add_column("Backend", style="dim")
    table.add_column("Status", style="bold")
    table.add_column("Created", style="dim")

    for j in jobs:
        st = str(j.get("status", "QUEUED"))
        st_style = "green" if st in {"RUNNING", "COMPLETED"} else ("red" if st == "FAILED" else "yellow")
        table.add_row(
            f"job-{j.get('job_id', 0):03d}",
            f"exp-{j.get('experiment_id', 0):03d}",
            str(j.get("backend", "kaggle")),
            f"[{st_style}]{st}[/{st_style}]",
            str(j.get("created_at", "-")),
        )
    return table


def stream_assistant_response(text: str, delay: float = 0.005) -> None:
    """Streams response text smoothly to the terminal."""
    console.print("\n[bold cyan]TunePilot>[/bold cyan] ", end="")
    for word in text.split(" "):
        console.print(word + " ", end="", markup=True)
        time.sleep(delay)
    console.print("\n")
