"""Parallel experiment planner (PRD sections 5, 22)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..models.candidate import ModelCandidate
from ..models.ranking import RankedCandidate


@dataclass
class ExperimentPlan:
    candidates: list[ModelCandidate] = field(default_factory=list)
    hyperparameters: dict[str, list[dict]] = field(default_factory=dict)
    parallel: bool = False
    concurrency: int = 1
    reason: str = ""

    def render(self) -> str:
        lines = ["EXPERIMENT PLAN", ""]
        for i, c in enumerate(self.candidates, 1):
            lines.append(f"{i}. {c.identifier}  ({c.source})")
            for hp in self.hyperparameters.get(c.key, []):
                lines.append(f"     {hp}")
        lines.append("")
        lines.append(f"Parallel: {self.parallel}  Concurrency: {self.concurrency}")
        lines.append(f"Reason: {self.reason}")
        return "\n".join(lines)


class ExperimentPlanner:
    """Decide how many candidates + HP variants to run, and whether in parallel."""

    def __init__(
        self,
        *,
        default_candidates: int = 3,
        hp_variants_per_candidate: int = 1,
        max_concurrent: int = 2,
    ) -> None:
        self.default_candidates = default_candidates
        self.hp_variants_per_candidate = hp_variants_per_candidate
        self.max_concurrent = max_concurrent

    def build(
        self,
        ranked: Iterable[RankedCandidate],
        *,
        available_gpu_hours: float,
        expected_hours_per_pilot: dict[str, float] | None = None,
        allow_parallel: bool = True,
    ) -> ExperimentPlan:
        chosen: list[ModelCandidate] = []
        for r in ranked:
            if r.feasibility and not r.feasibility.feasible:
                continue
            chosen.append(r.candidate)
            if len(chosen) >= self.default_candidates:
                break

        hp = {c.key: self._hp_grid() for c in chosen}
        expected_hours = expected_hours_per_pilot or {}
        total_sequential = sum(expected_hours.get(c.key, 1.0) for c in chosen)
        parallel = allow_parallel and len(chosen) > 1 and total_sequential > available_gpu_hours * 0.5
        concurrency = min(self.max_concurrent, len(chosen))
        reason = (
            f"{len(chosen)} candidates, {sum(len(v) for v in hp.values())} configs, "
            f"expected sequential {total_sequential:.1f}h vs available {available_gpu_hours:.1f}h"
        )
        return ExperimentPlan(
            candidates=chosen,
            hyperparameters=hp,
            parallel=parallel,
            concurrency=concurrency,
            reason=reason,
        )

    def _hp_grid(self) -> list[dict]:
        grids = [{"learning_rate": 2e-5, "epochs": 2}]
        if self.hp_variants_per_candidate >= 2:
            grids.append({"learning_rate": 5e-5, "epochs": 2})
        if self.hp_variants_per_candidate >= 3:
            grids.append({"learning_rate": 1e-5, "epochs": 3})
        return grids
