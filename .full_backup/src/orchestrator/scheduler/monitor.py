"""Runtime monitors used during and after training (PRD sections 5.3, 32)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class GpuMonitor:
    """Verify GPU is actually being used and detect multi-GPU underutilization."""

    expected_gpu_count: int
    min_utilization: float = 0.4

    def evaluate(self, per_gpu_utilization: list[float]) -> dict:
        active = sum(1 for u in per_gpu_utilization if u > 0.05)
        avg = sum(per_gpu_utilization) / max(1, len(per_gpu_utilization))
        underutilized = (
            self.expected_gpu_count > 1
            and active < self.expected_gpu_count
        )
        return {
            "active_gpus": active,
            "avg_utilization": avg,
            "underutilized": underutilized,
            "low_utilization": avg < self.min_utilization,
        }


@dataclass
class TrainingLossMonitor:
    """Detects diverged / non-finite training loss."""

    window: int = 20
    losses: list[float] = field(default_factory=list)

    def record(self, loss: float) -> str:
        self.losses.append(loss)
        if len(self.losses) > self.window:
            self.losses = self.losses[-self.window:]
        if not math.isfinite(loss):
            return "diverged"
        if len(self.losses) >= self.window:
            first = self.losses[0]
            last = self.losses[-1]
            if last > first * 3:
                return "diverging"
        return "ok"

    def summary(self) -> dict:
        if not self.losses:
            return {"count": 0}
        return {
            "count": len(self.losses),
            "min": min(self.losses),
            "max": max(self.losses),
            "last": self.losses[-1],
        }
