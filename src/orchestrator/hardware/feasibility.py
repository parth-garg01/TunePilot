"""VRAM estimation and feasibility decisions (PRD sections 10.3, 32)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.candidate import ModelCandidate
from .detect import HardwareProfile


class Strategy(str, Enum):
    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"
    SHARDED = "sharded"
    INFEASIBLE = "infeasible"


@dataclass
class FeasibilityDecision:
    feasible: bool
    strategy: str
    estimated_vram_gb: float
    reason: str
    notes: list[str]


def estimate_vram_gb(candidate: "ModelCandidate", *, dtype_bytes: float = 2.0, overhead: float = 1.4) -> float:
    """Rough VRAM estimate for full-precision fine-tuning of `candidate`."""
    n = candidate.metadata.parameters or 0
    if n <= 0:
        return 0.0
    weight_gb = n * dtype_bytes / (1024 ** 3)
    # Optimizer + activations approximated as ~2x weights for AdamW.
    return round(weight_gb * (1 + 2) * overhead, 2)


def _estimate_lora_vram(candidate: "ModelCandidate") -> float:
    base = estimate_vram_gb(candidate, dtype_bytes=2.0, overhead=1.2)
    return round(base * 0.35, 2)


def _estimate_qlora_vram(candidate: "ModelCandidate") -> float:
    base = estimate_vram_gb(candidate, dtype_bytes=0.5, overhead=1.2)  # 4-bit weights
    return round(max(2.0, base * 0.45), 2)


def feasibility(candidate: "ModelCandidate", hw: HardwareProfile) -> FeasibilityDecision:
    """Determine the cheapest training strategy that fits `hw` for `candidate`."""
    if hw.gpu_count == 0 or hw.total_vram_gb <= 0:
        return FeasibilityDecision(
            feasible=False,
            strategy=Strategy.INFEASIBLE.value,
            estimated_vram_gb=0.0,
            reason="no GPU detected",
            notes=[],
        )

    vram_per_gpu = max(g.memory_gb for g in hw.gpus)
    total_vram = hw.total_vram_gb
    notes: list[str] = []

    full = estimate_vram_gb(candidate)
    if full <= vram_per_gpu * 0.85:
        return FeasibilityDecision(True, Strategy.FULL.value, full,
                                    "full-precision fits on one GPU", notes)

    lora = _estimate_lora_vram(candidate)
    if lora <= vram_per_gpu * 0.85:
        notes.append("prefer LoRA to save memory")
        return FeasibilityDecision(True, Strategy.LORA.value, lora,
                                    "LoRA fits on one GPU", notes)

    qlora = _estimate_qlora_vram(candidate)
    if qlora <= vram_per_gpu * 0.9:
        notes.append("using 4-bit weights with LoRA")
        return FeasibilityDecision(True, Strategy.QLORA.value, qlora,
                                    "QLoRA fits on one GPU", notes)

    if full <= total_vram * 0.85 and hw.gpu_count > 1:
        notes.append("requires distributed sharding")
        return FeasibilityDecision(True, Strategy.SHARDED.value, full,
                                    "fits across all GPUs with sharding", notes)

    if qlora <= total_vram * 0.85 and hw.gpu_count > 1:
        notes.extend(["4-bit weights", "sharded across GPUs"])
        return FeasibilityDecision(True, Strategy.SHARDED.value, qlora,
                                    "QLoRA + sharding fits across GPUs", notes)

    return FeasibilityDecision(
        feasible=False,
        strategy=Strategy.INFEASIBLE.value,
        estimated_vram_gb=full,
        reason=f"needs {full:.1f}GB VRAM, hardware has {total_vram:.1f}GB total",
        notes=notes,
    )
