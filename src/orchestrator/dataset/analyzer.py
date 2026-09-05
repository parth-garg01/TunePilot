"""Dataset analyzer (PRD section 8).

Produces statistics, length distributions, task hints, and quality warnings
so expensive GPU jobs don't start on obviously broken data.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..logging_utils import get_logger
from .fingerprint import dataset_fingerprint
from .loader import DatasetLoader, LoadedDataset
from .quality import QualityChecker, QualityReport

log = get_logger(__name__)


def _approx_tokens(text: str) -> int:
    """Rough token estimate: 4 characters per token."""
    return max(1, len(text) // 4)


def _example_text(ex: dict[str, Any]) -> str:
    if "messages" in ex and isinstance(ex["messages"], list):
        return "\n".join(str(m.get("content", "")) for m in ex["messages"] if isinstance(m, dict))
    for key in ("text", "content", "prompt", "input"):
        if isinstance(ex.get(key), str):
            base = ex[key]
            for extra in ("response", "output", "answer", "completion"):
                if isinstance(ex.get(extra), str):
                    base += "\n" + ex[extra]
            return base
    return " ".join(str(v) for v in ex.values() if isinstance(v, (str, int, float)))


@dataclass
class DatasetReport:
    path: str
    format: str
    n_examples: int
    size_bytes: int
    tokens_est: int
    lengths: dict[str, float]
    schema: dict[str, int]
    detected_format: str
    fingerprint: str
    languages: dict[str, float] = field(default_factory=dict)
    class_distribution: dict[str, int] = field(default_factory=dict)
    quality: QualityReport = field(default_factory=QualityReport)
    splits_available: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            "DATASET REPORT",
            "",
            f"Examples: {self.n_examples:,}",
            f"Estimated tokens: {self.tokens_est:,}",
            f"Average sequence length: {self.lengths.get('mean', 0):.0f}",
            f"P95 sequence length: {self.lengths.get('p95', 0):.0f}",
            f"Maximum: {self.lengths.get('max', 0):.0f}",
            "",
            f"Detected format: {self.detected_format}",
            "",
        ]
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"- {w}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "format": self.format,
            "n_examples": self.n_examples,
            "size_bytes": self.size_bytes,
            "tokens_est": self.tokens_est,
            "lengths": self.lengths,
            "schema": self.schema,
            "detected_format": self.detected_format,
            "fingerprint": self.fingerprint,
            "languages": self.languages,
            "class_distribution": self.class_distribution,
            "quality": self.quality.summary(),
            "splits_available": self.splits_available,
            "warnings": self.warnings,
        }


class DatasetAnalyzer:
    """Analyze a dataset in one pass."""

    def __init__(self, loader: DatasetLoader | None = None) -> None:
        self.loader = loader or DatasetLoader()

    def analyze_path(self, path: Path | str) -> DatasetReport:
        ds = self.loader.load(path)
        return self.analyze(ds)

    def analyze(self, ds: LoadedDataset) -> DatasetReport:
        examples = ds.all_examples()
        schema = self._schema(examples)
        detected_format = self._detect_format(examples, schema)
        lengths, tokens = self._length_stats(examples)
        classes = self._class_distribution(examples)
        languages = self._language_hints(examples)
        quality = QualityChecker(required_fields=[]).check(examples)
        fingerprint = dataset_fingerprint(examples)

        report = DatasetReport(
            path=ds.path,
            format=ds.format,
            n_examples=len(examples),
            size_bytes=ds.size_bytes,
            tokens_est=tokens,
            lengths=lengths,
            schema=schema,
            detected_format=detected_format,
            fingerprint=fingerprint,
            languages=languages,
            class_distribution=classes,
            quality=quality,
            splits_available=list(ds.splits.keys()),
            warnings=list(quality.warnings),
        )
        return report

    def _schema(self, examples: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ex in examples:
            for k in ex.keys():
                counts[k] = counts.get(k, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def _detect_format(self, examples: list[dict[str, Any]], schema: dict[str, int]) -> str:
        if not examples:
            return "empty"
        keys = set(schema.keys())
        if "messages" in keys:
            return "messages[]"
        if {"prompt", "chosen", "rejected"} <= keys:
            return "preference_pairs"
        if "instruction" in keys and ("response" in keys or "output" in keys):
            return "instruction/response"
        if "text" in keys and "label" in keys:
            return "text_classification"
        if "text" in keys:
            return "raw_text"
        if "prompt" in keys and ("completion" in keys or "response" in keys):
            return "prompt/completion"
        return "generic"

    def _length_stats(self, examples: list[dict[str, Any]]) -> tuple[dict[str, float], int]:
        lens: list[int] = []
        tokens = 0
        for ex in examples:
            text = _example_text(ex)
            n = len(text)
            lens.append(n)
            tokens += _approx_tokens(text)
        if not lens:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}, 0
        lens.sort()

        def pct(p: float) -> float:
            i = int(p * (len(lens) - 1))
            return float(lens[i])

        stats = {
            "mean": statistics.fmean(lens),
            "min": float(lens[0]),
            "max": float(lens[-1]),
            "p50": pct(0.50),
            "p95": pct(0.95),
            "p99": pct(0.99),
        }
        return stats, tokens

    def _class_distribution(self, examples: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ex in examples:
            label = ex.get("label")
            if label is None:
                continue
            key = str(label)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _language_hints(self, examples: list[dict[str, Any]]) -> dict[str, float]:
        # Simple ASCII vs non-ASCII ratio. Real language detection lives behind
        # the huggingface extra and can be added later without changing this API.
        if not examples:
            return {}
        ascii_chars = 0
        total = 0
        for ex in examples[:5000]:
            t = _example_text(ex)
            total += len(t)
            ascii_chars += sum(1 for c in t if ord(c) < 128)
        if total == 0:
            return {}
        ratio = ascii_chars / total
        return {"ascii_ratio": round(ratio, 3), "likely_latin": ratio > 0.9}
