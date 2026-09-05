"""Adaptive telemetry-based re-ranking (PRD section 13A)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..models.ranking import RankedCandidate


@dataclass
class TelemetryRecord:
    candidate_key: str
    predicted_vram_gb: float
    actual_vram_gb: float
    predicted_hours: float
    actual_hours: float
    samples_per_sec: float
    gpu_utilization: float
    val_loss: float | None = None
    oom_events: int = 0

    @property
    def vram_error(self) -> float:
        if self.predicted_vram_gb <= 0:
            return 0.0
        return (self.actual_vram_gb - self.predicted_vram_gb) / self.predicted_vram_gb

    @property
    def time_error(self) -> float:
        if self.predicted_hours <= 0:
            return 0.0
        return (self.actual_hours - self.predicted_hours) / self.predicted_hours


@dataclass
class TelemetryFeedback:
    updated: list[RankedCandidate] = field(default_factory=list)
    decisions: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def adaptive_rerank(
    ranked: Iterable[RankedCandidate],
    telemetry: Iterable[TelemetryRecord],
    *,
    max_time_hours: float | None = None,
) -> TelemetryFeedback:
    tele = {t.candidate_key: t for t in telemetry}
    result = TelemetryFeedback()

    for r in ranked:
        t = tele.get(r.candidate.key)
        if t is None:
            result.updated.append(r)
            result.decisions[r.candidate.key] = "no telemetry, kept as-is"
            continue

        adj = 0.0
        notes: list[str] = []

        if t.oom_events > 0:
            adj -= 20
            notes.append(f"{t.oom_events} OOM events")
        if t.gpu_utilization < 0.4 and t.gpu_utilization > 0:
            adj -= 5
            notes.append(f"low GPU utilization {t.gpu_utilization:.2f}")
        if t.val_loss is not None and t.val_loss < 1.5:
            adj += 5
            notes.append(f"encouraging val_loss {t.val_loss:.2f}")

        if max_time_hours and t.actual_hours > max_time_hours:
            adj -= 15
            notes.append(f"exceeds time budget {t.actual_hours:.1f}h")

        # Correct predicted VRAM/time errors: penalize large under-estimates.
        if t.vram_error > 0.3:
            adj -= 8
            notes.append(f"VRAM under-estimated by {t.vram_error:.0%}")
        if t.time_error > 0.5:
            adj -= 5
            notes.append(f"time under-estimated by {t.time_error:.0%}")

        r.score = max(0.0, min(100.0, r.score + adj))
        decision = self_decision(r.score, t)
        result.updated.append(r)
        result.decisions[r.candidate.key] = decision
        result.notes.append(f"{r.candidate.identifier}: {', '.join(notes) or 'no adjustment'}")

    result.updated.sort(key=lambda x: -x.score)
    return result


def self_decision(score: float, t: TelemetryRecord) -> str:
    if t.oom_events > 0 and score < 40:
        return "rejected: infeasible"
    if score >= 75:
        return "promoted to full training"
    if score >= 55:
        return "retained for additional pilots"
    if score >= 30:
        return "deprioritized"
    return "rejected: inefficient"
