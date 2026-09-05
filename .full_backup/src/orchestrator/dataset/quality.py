"""Dataset quality checks (PRD section 8).

Detects empty examples, duplicates, conflicting labels, malformed records,
train/eval leakage, excessive duplication, and suspiciously repetitive
samples.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class QualityReport:
    n_examples: int = 0
    empty: int = 0
    duplicates: int = 0
    malformed: int = 0
    missing_fields: dict[str, int] = field(default_factory=dict)
    conflicting_labels: int = 0
    outliers: int = 0
    excessive_repetition: int = 0
    leakage: int = 0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "n_examples": self.n_examples,
            "empty": self.empty,
            "duplicates": self.duplicates,
            "malformed": self.malformed,
            "missing_fields": self.missing_fields,
            "conflicting_labels": self.conflicting_labels,
            "outliers": self.outliers,
            "excessive_repetition": self.excessive_repetition,
            "leakage": self.leakage,
            "warnings": self.warnings,
        }


class QualityChecker:
    """Runs quality checks over a list of examples."""

    def __init__(self, required_fields: Iterable[str] | None = None) -> None:
        self.required_fields = list(required_fields or [])

    def _example_hash(self, ex: dict[str, Any]) -> str:
        text = "".join(f"{k}={ex.get(k)}|" for k in sorted(ex.keys()) if k not in {"__malformed__", "__line__"})
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def check(
        self,
        examples: list[dict[str, Any]],
        *,
        eval_examples: list[dict[str, Any]] | None = None,
    ) -> QualityReport:
        report = QualityReport(n_examples=len(examples))
        hash_counts: Counter[str] = Counter()
        label_map: dict[str, set[str]] = {}
        length_field: str | None = None
        lengths: list[int] = []

        for ex in examples:
            if ex.get("__malformed__"):
                report.malformed += 1
                continue
            if not ex or all(v in (None, "") for v in ex.values()):
                report.empty += 1
                continue
            for f in self.required_fields:
                if f not in ex or ex[f] in (None, ""):
                    report.missing_fields[f] = report.missing_fields.get(f, 0) + 1
            digest = self._example_hash(ex)
            hash_counts[digest] += 1

            # Conflicting labels: same "input" seen with different "label"/"output".
            key = str(ex.get("input") or ex.get("prompt") or ex.get("text") or "")
            label = str(ex.get("label") or ex.get("output") or ex.get("response") or "")
            if key:
                label_map.setdefault(key, set()).add(label)

            # Length distribution for outlier detection.
            for f in ("text", "input", "prompt", "content"):
                if isinstance(ex.get(f), str):
                    length_field = length_field or f
                    lengths.append(len(ex[f]))
                    break

        report.duplicates = sum(1 for c in hash_counts.values() if c > 1)
        report.conflicting_labels = sum(1 for v in label_map.values() if len(v) > 1)

        if lengths:
            lengths.sort()
            p99 = lengths[int(0.99 * (len(lengths) - 1))]
            report.outliers = sum(1 for x in lengths if x > 5 * p99)

        # Excessive repetition: identical example dominates the dataset.
        if hash_counts:
            top_frac = max(hash_counts.values()) / max(1, sum(hash_counts.values()))
            if top_frac > 0.05:
                report.excessive_repetition = max(hash_counts.values())
                report.warnings.append(
                    f"one example accounts for {top_frac:.1%} of the dataset"
                )

        if eval_examples:
            eval_hashes = {self._example_hash(e) for e in eval_examples if not e.get("__malformed__")}
            report.leakage = sum(1 for ex in examples if self._example_hash(ex) in eval_hashes)

        if report.duplicates and report.n_examples:
            report.warnings.append(f"{report.duplicates / report.n_examples:.1%} duplicate examples")
        if report.malformed:
            report.warnings.append(f"{report.malformed / max(1, report.n_examples):.1%} malformed records")
        if report.outliers:
            report.warnings.append(f"{report.outliers} unusually long samples")
        if report.leakage:
            report.warnings.append(f"{report.leakage} train/evaluation leaks detected")

        return report
