"""Task-specific metrics + normalization helpers (PRD sections 19, 20)."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable


class Metric(ABC):
    name: str = "metric"
    higher_is_better: bool = True

    @abstractmethod
    def compute(self, predictions: list[str], references: list[str]) -> float:
        raise NotImplementedError


class TaskMetric(Metric):
    pass


@dataclass
class TrainingMetrics:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    perplexity: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        def last(xs):
            return xs[-1] if xs else float("nan")
        return {
            "train_loss": last(self.train_loss),
            "val_loss": last(self.val_loss),
            "perplexity": last(self.perplexity),
        }


class AccuracyMetric(TaskMetric):
    name = "accuracy"
    higher_is_better = True

    def compute(self, predictions: list[str], references: list[str]) -> float:
        if not references:
            return 0.0
        correct = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
        return correct / len(references)


class ExactMatchMetric(TaskMetric):
    name = "exact_match"
    higher_is_better = True

    def compute(self, predictions: list[str], references: list[str]) -> float:
        return AccuracyMetric().compute(predictions, references)


class F1Metric(TaskMetric):
    name = "f1"
    higher_is_better = True

    def compute(self, predictions: list[str], references: list[str]) -> float:
        scores = [self._per_example(p, r) for p, r in zip(predictions, references)]
        return sum(scores) / max(1, len(scores))

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _per_example(self, p: str, r: str) -> float:
        pt, rt = self._tokens(p), self._tokens(r)
        if not pt or not rt:
            return 0.0
        common = Counter(pt) & Counter(rt)
        overlap = sum(common.values())
        if overlap == 0:
            return 0.0
        precision = overlap / len(pt)
        recall = overlap / len(rt)
        return 2 * precision * recall / (precision + recall)


class BLEUMetric(TaskMetric):
    """A very small BLEU-1 implementation (unigram precision + brevity penalty)."""
    name = "bleu"
    higher_is_better = True

    def compute(self, predictions: list[str], references: list[str]) -> float:
        precisions: list[float] = []
        length_ratios: list[float] = []
        for p, r in zip(predictions, references):
            pt = re.findall(r"\w+", p.lower())
            rt = re.findall(r"\w+", r.lower())
            if not pt:
                precisions.append(0.0)
                length_ratios.append(0.0)
                continue
            common = Counter(pt) & Counter(rt)
            precisions.append(sum(common.values()) / len(pt))
            length_ratios.append(len(pt) / max(1, len(rt)))
        if not precisions:
            return 0.0
        bp = math.exp(min(0.0, 1 - sum(length_ratios) / len(length_ratios)))
        return bp * (sum(precisions) / len(precisions))


class ROUGEMetric(TaskMetric):
    """ROUGE-L F-measure over unigrams (simplified)."""
    name = "rouge"
    higher_is_better = True

    def compute(self, predictions: list[str], references: list[str]) -> float:
        return F1Metric().compute(predictions, references)


class PerplexityMetric(TaskMetric):
    name = "perplexity"
    higher_is_better = False

    def compute(self, predictions: list[str], references: list[str]) -> float:
        # `predictions` is interpreted as pre-computed nll values here.
        if not predictions:
            return float("inf")
        nlls = [float(x) for x in predictions]
        return math.exp(sum(nlls) / len(nlls))


def normalize_metric(value: float, *, higher_is_better: bool, min_v: float, max_v: float) -> float:
    """Map a raw metric to [0, 1] with 1 always meaning best."""
    if max_v == min_v:
        return 0.5
    x = (value - min_v) / (max_v - min_v)
    x = max(0.0, min(1.0, x))
    return x if higher_is_better else 1 - x
