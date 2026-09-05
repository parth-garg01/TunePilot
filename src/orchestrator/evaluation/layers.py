"""Evaluation layers (PRD section 19)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..logging_utils import get_logger
from .metrics import Metric, TrainingMetrics

log = get_logger(__name__)


@dataclass
class EvaluationOutcome:
    layer: str
    metrics: dict[str, float] = field(default_factory=dict)
    normalized: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class EvaluationRunner:
    """Runs the four evaluation layers described in the PRD."""

    def __init__(self, task_metrics: Iterable[Metric]) -> None:
        self.task_metrics = list(task_metrics)

    def layer1(self, tm: TrainingMetrics) -> EvaluationOutcome:
        summary = tm.summary()
        return EvaluationOutcome(layer="training", metrics=summary, normalized={})

    def layer2(self, predictions: list[str], references: list[str]) -> EvaluationOutcome:
        outcome = EvaluationOutcome(layer="task")
        for metric in self.task_metrics:
            outcome.metrics[metric.name] = metric.compute(predictions, references)
        return outcome

    def layer3(self, judge: "LLMJudge", prompts: list[str], responses: list[str]) -> EvaluationOutcome:
        scores = judge.score(prompts, responses)
        avg = sum(scores) / max(1, len(scores))
        return EvaluationOutcome(layer="llm_judge", metrics={"judge_avg": avg}, metadata={"per_item": scores})

    def layer4(self, iface: "HumanReviewInterface") -> EvaluationOutcome:
        scores = iface.collected_scores()
        avg = sum(scores) / max(1, len(scores))
        return EvaluationOutcome(layer="human", metrics={"human_avg": avg}, metadata={"n": len(scores)})


class LLMJudge:
    """Placeholder LLM judge.

    Real implementations should call a *different* model than the one being
    trained (PRD section 19) to avoid self-evaluation bias. The default judge
    uses a scoring function callback provided at construction time.
    """

    def __init__(self, scorer: Callable[[str, str], float] | None = None) -> None:
        self.scorer = scorer or (lambda prompt, response: min(1.0, len(response) / 200))

    def score(self, prompts: list[str], responses: list[str]) -> list[float]:
        return [self.scorer(p, r) for p, r in zip(prompts, responses)]


@dataclass
class HumanReviewInterface:
    """Very small collector for human-in-the-loop scoring."""

    scores: list[float] = field(default_factory=list)

    def record(self, score: float) -> None:
        self.scores.append(float(score))

    def collected_scores(self) -> list[float]:
        return list(self.scores)
