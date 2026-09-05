"""Normalized candidate comparison + winner selection (PRD sections 20, 21)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .metrics import normalize_metric


@dataclass
class CandidateResult:
    identifier: str
    val_loss: float | None = None
    task_score: float | None = None
    tokens_per_sec: float | None = None
    vram_gb: float | None = None
    training_hours: float | None = None
    cost: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ComparisonReport:
    rows: list[CandidateResult]
    scores: dict[str, float]
    winner: str | None
    objective: str
    reason: str

    def render(self) -> str:
        headers = ["Candidate", "ValLoss", "Task", "Tok/s", "VRAM", "Time", "Overall"]
        lines = ["MODEL COMPARISON", "", "  ".join(f"{h:<10}" for h in headers), "-" * 66]
        for r in self.rows:
            lines.append(
                "  ".join(
                    f"{r.identifier[:10]:<10}"
                    for r in [r]
                )
                + "  "
                + "  ".join(
                    f"{fmt(v):<10}"
                    for v in [r.val_loss, r.task_score, r.tokens_per_sec, r.vram_gb, r.training_hours, self.scores.get(r.identifier)]
                )
            )
        lines.append("")
        lines.append(f"Winner: {self.winner or 'no candidate'}")
        lines.append(f"Objective: {self.objective}")
        lines.append(f"Reason: {self.reason}")
        return "\n".join(lines)


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


class CandidateComparison:
    """Compute normalized scores across candidates and pick a winner."""

    def __init__(self, objective: str = "balanced") -> None:
        self.objective = objective

    def compare(self, rows: Iterable[CandidateResult]) -> ComparisonReport:
        rows = list(rows)
        if not rows:
            return ComparisonReport(rows=[], scores={}, winner=None, objective=self.objective,
                                     reason="no candidates")

        # Collect ranges for normalization.
        def rng(getter, default: float) -> tuple[float, float]:
            values = [getter(r) for r in rows if getter(r) is not None]
            return (min(values), max(values)) if values else (default, default)

        r_val = rng(lambda r: r.val_loss, 0.0)
        r_task = rng(lambda r: r.task_score, 0.0)
        r_tok = rng(lambda r: r.tokens_per_sec, 0.0)
        r_vram = rng(lambda r: r.vram_gb, 0.0)
        r_time = rng(lambda r: r.training_hours, 0.0)

        scores: dict[str, float] = {}
        for r in rows:
            components = []
            if r.val_loss is not None:
                components.append(("val_loss", 0.25, normalize_metric(r.val_loss, higher_is_better=False, min_v=r_val[0], max_v=r_val[1])))
            if r.task_score is not None:
                components.append(("task", 0.35, normalize_metric(r.task_score, higher_is_better=True, min_v=r_task[0], max_v=r_task[1])))
            if r.tokens_per_sec is not None:
                components.append(("speed", 0.10, normalize_metric(r.tokens_per_sec, higher_is_better=True, min_v=r_tok[0], max_v=r_tok[1])))
            if r.vram_gb is not None:
                components.append(("memory", 0.10, normalize_metric(r.vram_gb, higher_is_better=False, min_v=r_vram[0], max_v=r_vram[1])))
            if r.training_hours is not None:
                components.append(("time", 0.20, normalize_metric(r.training_hours, higher_is_better=False, min_v=r_time[0], max_v=r_time[1])))

            weights = sum(w for _, w, _ in components) or 1
            score = sum(w * v for _, w, v in components) / weights * 100
            scores[r.identifier] = round(score, 2)

        winner = max(scores.items(), key=lambda kv: kv[1])[0]
        reason = f"objective={self.objective}, normalized composite of {len(scores)} candidates"
        return ComparisonReport(rows=rows, scores=scores, winner=winner, objective=self.objective, reason=reason)
