"""Compute strategy selection (PRD sections 15, 15A)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..hardware.detect import HardwareProfile


class ComputeStrategy(str, Enum):
    LOCAL_4GB = "local_4gb"
    CLOUD = "cloud"
    AUTO = "auto"


@dataclass
class ComputeOption:
    name: str
    backend: str
    hardware: HardwareProfile
    feasible: bool = True
    estimated_hours: float = 0.0
    estimated_cost: float = 0.0
    expected_quality: float = 0.0
    remaining_quota_hours: float | None = None
    reason: str = ""

    def render(self) -> str:
        lines = [
            f"{self.name}:",
            f"  Feasible: {'YES' if self.feasible else 'NO'}",
            f"  Estimated time: {self.estimated_hours:.1f}h",
            f"  Estimated cost: ${self.estimated_cost:.2f}",
            f"  Expected quality: {self.expected_quality:.0f}",
        ]
        if self.remaining_quota_hours is not None:
            lines.append(f"  Quota remaining: {self.remaining_quota_hours:.1f}h")
        return "\n".join(lines)


class ComputeSelector:
    """Rank compute options based on the objective."""

    def __init__(self, objective: str = "quality_first") -> None:
        self.objective = objective

    def recommend(self, options: list[ComputeOption]) -> tuple[ComputeOption | None, str]:
        eligible = [o for o in options if o.feasible]
        if not eligible:
            return None, "no feasible compute option"

        obj = self.objective
        if obj in {"quality_first", "quality"}:
            eligible.sort(key=lambda o: -o.expected_quality)
            reason = "quality-first objective selected the highest expected-quality feasible option"
        elif obj == "speed":
            eligible.sort(key=lambda o: o.estimated_hours)
            reason = "speed objective selected the fastest feasible option"
        elif obj == "quality_per_dollar":
            eligible.sort(key=lambda o: -(o.expected_quality / max(0.01, o.estimated_cost)))
            reason = "quality-per-dollar objective selected the most cost-efficient option"
        elif obj == "quality_per_gpu_hour":
            eligible.sort(key=lambda o: -(o.expected_quality / max(0.1, o.estimated_hours)))
            reason = "quality-per-gpu-hour objective selected the most time-efficient option"
        else:
            eligible.sort(key=lambda o: -(o.expected_quality - 5 * o.estimated_cost))
            reason = "balanced objective weighed quality against cost"
        return eligible[0], reason
