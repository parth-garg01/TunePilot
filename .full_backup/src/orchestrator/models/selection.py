"""Best-model selection policy (PRD sections 11, 21)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from .ranking import RankedCandidate


class Objective(str, Enum):
    QUALITY_FIRST = "quality_first"
    QUALITY = "quality"
    QUALITY_PER_DOLLAR = "quality_per_dollar"
    QUALITY_PER_GPU_HOUR = "quality_per_gpu_hour"
    SPEED = "speed"
    MEMORY_EFFICIENCY = "memory_efficiency"
    BALANCED = "balanced"


@dataclass
class SelectionReport:
    winner: RankedCandidate | None
    alternatives: list[RankedCandidate] = field(default_factory=list)
    objective: Objective = Objective.BALANCED
    reason: str = ""

    def render(self) -> str:
        if not self.winner:
            return "No candidate met the selection criteria."
        lines = [
            f"Recommended: {self.winner.candidate.identifier}",
            "",
            "Alternatives:",
        ]
        for alt in self.alternatives:
            lines.append(f"  {alt.candidate.identifier}  score={alt.score:.1f}")
        lines.append("")
        lines.append(f"Reason: {self.reason}")
        return "\n".join(lines)


class SelectionPolicy:
    """Applies an objective and hard constraints to pick a winner."""

    def __init__(
        self,
        objective: Objective = Objective.BALANCED,
        *,
        max_training_hours: float | None = None,
        max_cost: float | None = None,
        max_model_parameters: int | None = None,
    ) -> None:
        self.objective = objective
        self.max_training_hours = max_training_hours
        self.max_cost = max_cost
        self.max_model_parameters = max_model_parameters

    def select(
        self,
        ranked: Iterable[RankedCandidate],
        *,
        training_hours: dict[str, float] | None = None,
        costs: dict[str, float] | None = None,
    ) -> SelectionReport:
        eligible = [r for r in ranked if r.feasibility is None or r.feasibility.feasible]
        if self.max_model_parameters:
            eligible = [
                r for r in eligible
                if not r.candidate.metadata.parameters or r.candidate.metadata.parameters <= self.max_model_parameters
            ]
        if self.max_training_hours and training_hours:
            eligible = [r for r in eligible if training_hours.get(r.candidate.key, 0) <= self.max_training_hours]
        if self.max_cost and costs:
            eligible = [r for r in eligible if costs.get(r.candidate.key, 0) <= self.max_cost]

        eligible = sorted(eligible, key=self._sort_key(training_hours or {}, costs or {}))
        if not eligible:
            return SelectionReport(winner=None, objective=self.objective, reason="no feasible candidate")

        winner = eligible[0]
        alts = eligible[1:4]
        reason = self._reason(winner)
        return SelectionReport(winner=winner, alternatives=alts, objective=self.objective, reason=reason)

    def _sort_key(self, hours: dict[str, float], costs: dict[str, float]):
        obj = self.objective

        def key(r: RankedCandidate):
            score = r.score
            h = max(0.1, hours.get(r.candidate.key, 1.0))
            c = max(0.01, costs.get(r.candidate.key, 0.01))
            if obj in {Objective.QUALITY, Objective.QUALITY_FIRST}:
                return -score
            if obj == Objective.QUALITY_PER_GPU_HOUR:
                return -(score / h)
            if obj == Objective.QUALITY_PER_DOLLAR:
                return -(score / c)
            if obj == Objective.SPEED:
                return h
            if obj == Objective.MEMORY_EFFICIENCY:
                return -r.subscores.get("efficiency", 0)
            return -(score - 0.5 * h)  # balanced

        return key

    def _reason(self, w: RankedCandidate) -> str:
        obj = self.objective.value
        return (
            f"Objective '{obj}' selected the highest-scoring feasible candidate. "
            f"{w.reason}"
        )
