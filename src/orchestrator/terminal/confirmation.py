"""Confirmation Manager for Expensive Operations (PRD Section 12, Feature Update)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class OperationPlan:
    action: str  # "train" | "pilot" | "evaluate_all" | "clean" | "export"
    models: list[str] = field(default_factory=list)
    compute_backend: str = "kaggle_t4x2"
    strategy: str = "Pilot -> Re-rank -> Full Training"
    objective: str = "Quality-first"
    estimated_compute_hours: float = 2.0
    confirmed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def format_plan_card(self) -> str:
        lines = [
            "\n+----------------------------------------------------+",
            "|                   Training Plan                    |",
            "+----------------------------------------------------+",
            "",
            "Models:",
        ]
        if self.models:
            for i, m in enumerate(self.models, 1):
                lines.append(f"  {i}. {m}")
        else:
            lines.append("  (Auto-selected top candidates)")

        lines.extend([
            f"\nCompute:\n  {self.compute_backend}",
            f"\nStrategy:\n  {self.strategy}",
            f"\nObjective:\n  {self.objective}",
            f"\nEstimated compute:\n  ~{self.estimated_compute_hours:.1f} GPU-hours",
            "\n[bold yellow]Proceed? [Y/n][/bold yellow] [dim](or type modification e.g. 'only train top 1')[/dim]",
        ])
        return "\n".join(lines)



class ConfirmationManager:
    """Handles verification, user approval, and prompt-driven parameter alterations."""

    def __init__(self) -> None:
        self.pending_plan: Optional[OperationPlan] = None

    def create_training_plan(
        self,
        models: list[str],
        backend: str = "kaggle_t4x2",
        hours: float = 4.0,
        objective: str = "Quality-first",
    ) -> OperationPlan:
        plan = OperationPlan(
            action="train",
            models=models,
            compute_backend=backend,
            estimated_compute_hours=hours,
            objective=objective,
        )
        self.pending_plan = plan
        return plan

    def evaluate_response(self, user_input: str) -> tuple[bool, Optional[str]]:
        """
        Returns (is_handled, status_message).
        True, "confirmed" -> User accepted.
        True, "cancelled" -> User declined.
        True, "modified" -> User modified parameters.
        False, None -> Not a confirmation response.
        """
        if not self.pending_plan:
            return False, None

        inp = user_input.strip().lower()

        # Direct affirmations
        if inp in {"y", "yes", "proceed", "go", "confirm", "start", "run"}:
            self.pending_plan.confirmed = True
            plan = self.pending_plan
            self.pending_plan = None
            return True, "confirmed"

        # Direct cancellations
        if inp in {"n", "no", "cancel", "stop", "abort"}:
            self.pending_plan = None
            return True, "cancelled"

        # Modifications: e.g. "Only train the first two" / "only 1"
        top_k_match = re.search(r"(?:only\s+train\s+(?:the\s+)?|top\s+)(\d+|first|first\s+two|one|two)", inp)
        if top_k_match:
            k_str = top_k_match.group(1).lower()
            num = 1
            if k_str in {"2", "two", "first two"}:
                num = 2
            elif k_str.isdigit():
                num = int(k_str)

            self.pending_plan.models = self.pending_plan.models[:num]
            self.pending_plan.estimated_compute_hours = max(1.0, num * 1.5)
            return True, "modified"

        # Compute override: "Use local compute instead"
        if "local" in inp:
            self.pending_plan.compute_backend = "local_gpu"
            return True, "modified"

        if "cloud" in inp or "kaggle" in inp:
            self.pending_plan.compute_backend = "kaggle_t4x2"
            return True, "modified"

        return False, None
