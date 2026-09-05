"""Interactive Help and Command Discoverability System (PRD Section 16, Feature Update)."""

from __future__ import annotations

from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table


def get_help_panel() -> RenderableType:
    """Build a beautifully formatted Rich panel for interactive terminal help."""
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(style="bold cyan", no_wrap=True)
    grid.add_column(style="dim")

    sections = [
        ("Project Commands", [
            ("init <name>", "Initialize project layout, database, and starter config"),
            ("config show", "Inspect active orchestrator configuration"),
            ("status", "Show current project state, active jobs, and experiments"),
        ]),
        ("Dataset Engine", [
            ("dataset analyze <path>", "Inspect format, tokens, distribution, and anomalies"),
            ("dataset inspect", "Preview schema, examples, and task classification"),
        ]),
        ("Model Discovery", [
            ("models discover", "Find, filter, and rank compatible base models"),
            ("models list", "Display currently discovered candidate models"),
        ]),
        ("Compute & Hardware", [
            ("hardware detect", "Inspect local GPUs, CPU, VRAM, and remote profiles"),
            ("compute providers", "List authorized compute backends (Kaggle T4x2, etc.)"),
        ]),
        ("Planning & Training", [
            ("plan", "Generate pilot and full experiment execution plans"),
            ("train", "Submit training jobs to authorized compute backend"),
            ("jobs status", "Monitor queued, running, and completed GPU jobs"),
            ("experiments list", "View detailed experiment metrics and history"),
        ]),
        ("Evaluation & Export", [
            ("evaluate", "Run multi-layer evaluation on trained checkpoints"),
            ("compare", "Rank trained models and select the winning candidate"),
            ("export", "Merge LoRA adapters, quantize, and export artifacts"),
        ]),
        ("Assistant & Navigation", [
            ("help / /help", "Display this commands and help overview"),
            ("clear / /clear", "Clear the terminal screen"),
            ("exit / Ctrl+D", "Exit interactive session (remote jobs continue)"),
        ]),
    ]

    for title, cmds in sections:
        grid.add_row(f"\n[bold yellow]{title}[/bold yellow]", "")
        for name, desc in cmds:
            grid.add_row(f"  {name:<26}", desc)

    grid.add_row(
        "\n[bold green]Natural Language Examples[/bold green]",
        "\n[dim]You can also type plain English instructions anytime:[/dim]",
    )
    grid.add_row('  "Analyze my dataset ./data/train.jsonl"', "Inspect dataset and detect task")
    grid.add_row('  "Find the best 7B models for this task"', "Discover and filter top candidates")
    grid.add_row('  "Why was that model rejected?"', "Inspect architectural or hardware filters")
    grid.add_row('  "Train the top 2 candidates"', "Plan and launch training jobs")
    grid.add_row('  "Resume the failed experiment"', "Safely retry from latest checkpoint")

    return Panel(grid, title="[bold]TunePilot Commands & Conversational Guide[/bold]", border_style="cyan")
