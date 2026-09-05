"""Final training report."""

from __future__ import annotations

from ..evaluation.comparison import ComparisonReport


def render_final_report(comparison: ComparisonReport, *, model_dir: str, training_hours: float) -> str:
    lines = [
        "FINAL MODEL",
        "",
        f"Winner: {comparison.winner}",
        f"Model directory: {model_dir}",
        f"Training time: {training_hours:.1f}h",
        "",
        comparison.render(),
    ]
    return "\n".join(lines)
