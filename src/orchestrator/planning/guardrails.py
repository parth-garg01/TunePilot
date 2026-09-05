"""Cost, quota, and duration guardrails (PRD section 25)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GuardrailDecision:
    approved: bool
    reason: str
    action: str = ""

    def render(self) -> str:
        return (
            f"Estimated compute: {self.reason}\n"
            f"Decision: {'PROCEED' if self.approved else 'BLOCK'}\n"
            f"Action: {self.action or 'n/a'}"
        )


class CostGuardrail:
    """Compare estimated vs available compute and produce an actionable decision."""

    def __init__(
        self,
        *,
        available_gpu_hours: float,
        available_budget: float | None = None,
        max_job_hours: float | None = None,
    ) -> None:
        self.available_gpu_hours = available_gpu_hours
        self.available_budget = available_budget
        self.max_job_hours = max_job_hours

    def check(
        self,
        *,
        estimated_gpu_hours: float,
        estimated_cost: float = 0.0,
        approved_by_user: bool = False,
    ) -> GuardrailDecision:
        if self.max_job_hours and estimated_gpu_hours > self.max_job_hours:
            return GuardrailDecision(
                False,
                f"{estimated_gpu_hours:.1f} > per-job cap {self.max_job_hours:.1f}h",
                "reduce scope or split into smaller jobs",
            )
        if estimated_gpu_hours > self.available_gpu_hours:
            return GuardrailDecision(
                False,
                f"{estimated_gpu_hours:.1f}h estimated, {self.available_gpu_hours:.1f}h available",
                "run a reduced pilot instead or wait for quota reset",
            )
        if self.available_budget is not None and estimated_cost > self.available_budget:
            return GuardrailDecision(
                False,
                f"${estimated_cost:.2f} estimated, ${self.available_budget:.2f} available",
                "reduce candidates, use cheaper backend, or increase budget",
            )
        if not approved_by_user and (estimated_gpu_hours > 4 or estimated_cost > 5):
            return GuardrailDecision(
                False,
                f"{estimated_gpu_hours:.1f}h, ${estimated_cost:.2f}",
                "explicit user approval required for this job size",
            )
        return GuardrailDecision(
            True,
            f"{estimated_gpu_hours:.1f}h / ${estimated_cost:.2f} within limits",
            "submit",
        )
