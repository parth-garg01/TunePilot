"""Hardware-aware model ranking (PRD section 10.3, 11)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..hardware.feasibility import FeasibilityDecision, HardwareProfile, estimate_vram_gb, feasibility
from .candidate import ModelCandidate, TrainingStage


@dataclass
class RankingWeights:
    task_compatibility: float = 0.25
    hardware_feasibility: float = 0.20
    expected_quality: float = 0.15
    dataset_fit: float = 0.10
    context_compatibility: float = 0.10
    training_efficiency: float = 0.10
    ecosystem_maturity: float = 0.05
    license_suitability: float = 0.05

    def normalized(self) -> "RankingWeights":
        total = sum(vars(self).values())
        if total == 0:
            return self
        return RankingWeights(**{k: v / total for k, v in vars(self).items()})


@dataclass
class RankedCandidate:
    candidate: ModelCandidate
    score: float
    subscores: dict[str, float] = field(default_factory=dict)
    feasibility: FeasibilityDecision | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "identifier": self.candidate.identifier,
            "source": self.candidate.source,
            "score": round(self.score, 2),
            "subscores": {k: round(v, 3) for k, v in self.subscores.items()},
            "feasible": self.feasibility.feasible if self.feasibility else True,
            "reason": self.reason,
        }


ACCEPTABLE_LICENSES = {"apache-2.0", "mit", "bsd", "gemma", "llama", "qwen", "nvidia-open-model"}


class ModelRanker:
    """Score candidates in [0, 100]."""

    def __init__(self, weights: RankingWeights | None = None) -> None:
        self.weights = (weights or RankingWeights()).normalized()

    def rank(
        self,
        candidates: Iterable[ModelCandidate],
        *,
        task: str,
        hardware: HardwareProfile,
        dataset_tokens: int | None = None,
        required_context: int | None = None,
        prefer_base: bool = False,
    ) -> list[RankedCandidate]:
        ranked: list[RankedCandidate] = []
        for c in candidates:
            feas = feasibility(c, hardware)
            sub = self._subscores(
                c,
                task=task,
                feas=feas,
                dataset_tokens=dataset_tokens,
                required_context=required_context,
                prefer_base=prefer_base,
            )
            score = 100 * (
                self.weights.task_compatibility * sub["task"]
                + self.weights.hardware_feasibility * sub["hardware"]
                + self.weights.expected_quality * sub["quality"]
                + self.weights.dataset_fit * sub["dataset_fit"]
                + self.weights.context_compatibility * sub["context"]
                + self.weights.training_efficiency * sub["efficiency"]
                + self.weights.ecosystem_maturity * sub["ecosystem"]
                + self.weights.license_suitability * sub["license"]
            )
            reason = self._reason(sub, feas, c)
            ranked.append(RankedCandidate(candidate=c, score=score, subscores=sub, feasibility=feas, reason=reason))
        ranked.sort(key=lambda r: -r.score)
        return ranked

    def _subscores(
        self,
        c: ModelCandidate,
        *,
        task: str,
        feas: FeasibilityDecision,
        dataset_tokens: int | None,
        required_context: int | None,
        prefer_base: bool,
    ) -> dict[str, float]:
        # Task compatibility.
        task_score = 1.0 if task in c.metadata.supported_tasks else 0.5
        if task in {"sft", "chat", "instruction"} and any(
            t in {"sft", "chat", "instruction"} for t in c.metadata.supported_tasks
        ):
            task_score = max(task_score, 0.9)

        # Hardware feasibility. Penalize models that require heavy tricks.
        hw = 1.0 if feas.feasible else 0.0
        if feas.feasible and feas.strategy in {"qlora", "sharded"}:
            hw = 0.7
        if feas.feasible and feas.strategy == "full":
            hw = 1.0

        # Expected quality: parameters as a weak proxy, with diminishing returns.
        n = max(1, c.metadata.parameters)
        import math
        quality = min(1.0, math.log10(max(10, n)) / 12.0)  # ~1.0 near 1e12

        # Dataset fit: larger data + moderate model beats tiny + huge.
        if dataset_tokens:
            dtok = max(1_000, dataset_tokens)
            ratio = dtok / max(1, n / 20)  # rough Chinchilla-ish
            dataset_fit = max(0.2, min(1.0, ratio))
        else:
            dataset_fit = 0.6

        # Context compatibility.
        if required_context and c.metadata.context_length:
            context = min(1.0, c.metadata.context_length / required_context)
        else:
            context = 0.75

        # Training efficiency: smaller VRAM footprint scores higher.
        est = estimate_vram_gb(c) or (0.002 * n / 1_000_000_000)
        efficiency = max(0.1, min(1.0, 8.0 / max(1.0, est)))

        # Ecosystem maturity from source priority + benchmarks.
        eco = {
            "official": 1.0,
            "huggingface": 0.95,
            "modelscope": 0.85,
            "unsloth": 0.85,
            "ngc": 0.85,
            "local": 0.7,
            "user": 0.6,
        }.get(c.source, 0.6)
        if c.metadata.benchmark_evidence:
            eco = min(1.0, eco + 0.05)

        # License.
        lic = c.metadata.license.lower()
        license_score = 1.0 if lic in ACCEPTABLE_LICENSES else (0.6 if lic else 0.4)

        # Base-model preference penalty.
        if prefer_base and c.stage != TrainingStage.BASE:
            task_score *= 0.9
            quality *= 0.95

        return {
            "task": task_score,
            "hardware": hw,
            "quality": quality,
            "dataset_fit": dataset_fit,
            "context": context,
            "efficiency": efficiency,
            "ecosystem": eco,
            "license": license_score,
        }

    def _reason(self, sub: dict[str, float], feas: FeasibilityDecision, c: ModelCandidate) -> str:
        best = sorted(sub.items(), key=lambda kv: -kv[1])[:3]
        parts = [f"{k}={v:.2f}" for k, v in best]
        f = "feasible" if feas.feasible else "infeasible"
        return f"top signals: {', '.join(parts)}; hardware {f} via {feas.strategy}"
