"""Pilot training planning and result modeling (PRD section 13, 13A)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..hardware.detect import HardwareProfile
from ..models.candidate import ModelCandidate


@dataclass
class PilotPlan:
    steps: int = 200
    dataset_fraction: float = 0.05
    representative_seq_length: int = 1024
    hardware: HardwareProfile | None = None
    max_wall_minutes: float = 20.0


@dataclass
class PilotResult:
    candidate_key: str
    samples_per_sec: float = 0.0
    tokens_per_sec: float = 0.0
    vram_used_gb: float = 0.0
    loss: float = 0.0
    val_loss: float | None = None
    oom_events: int = 0
    gpu_utilization: float = 0.0
    duration_seconds: float = 0.0
    estimated_full_training_hours: float = 0.0
    warnings: list[str] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)


class PilotPlanner:
    """Build a pilot plan sized to the candidate + dataset."""

    def plan(
        self,
        candidate: ModelCandidate,
        *,
        hardware: HardwareProfile,
        dataset_examples: int,
        dataset_p95_length: int,
    ) -> PilotPlan:
        # Pick a step budget in [100, 500] scaled by dataset size.
        steps = max(100, min(500, dataset_examples // 200))
        frac = min(0.1, max(0.01, 2000 / max(1, dataset_examples)))
        seq = max(512, min(dataset_p95_length, 4096))
        return PilotPlan(
            steps=steps,
            dataset_fraction=frac,
            representative_seq_length=seq,
            hardware=hardware,
            max_wall_minutes=20.0,
        )

    def estimate_full_training(
        self,
        result: PilotResult,
        *,
        total_examples: int,
        epochs: int,
    ) -> float:
        """Extrapolate expected wall-clock hours for full training."""
        if result.samples_per_sec <= 0:
            return 0.0
        total_samples = total_examples * epochs
        seconds = total_samples / result.samples_per_sec
        return round(seconds / 3600.0, 2)
