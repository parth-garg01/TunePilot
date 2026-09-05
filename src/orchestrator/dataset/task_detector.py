"""Automatic task detection from dataset schema (PRD section 9)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .analyzer import DatasetReport


@dataclass
class TaskDetection:
    task: str
    confidence: float
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    reason: str = ""

    def render(self) -> str:
        lines = [f"Detected task: {self.task}", f"Confidence: {int(self.confidence * 100)}%"]
        if self.alternatives:
            lines.append("")
            lines.append("Alternative:")
            for name, c in self.alternatives:
                lines.append(f"{name}: {int(c * 100)}%")
        return "\n".join(lines)


class TaskDetector:
    """Heuristic classifier over dataset shape and field distributions."""

    def detect(self, report: DatasetReport) -> TaskDetection:
        scores: dict[str, float] = {}
        keys = set(report.schema.keys())
        fmt = report.detected_format

        if fmt == "messages[]":
            scores["sft"] = 0.9
            scores["chat"] = 0.85
            scores["instruction"] = 0.7
        if fmt == "instruction/response":
            scores["instruction"] = 0.95
            scores["sft"] = 0.85
        if fmt == "preference_pairs":
            scores["dpo"] = 0.9
            scores["orpo"] = 0.5
            scores["kto"] = 0.45
        if fmt == "text_classification":
            scores["classification"] = 0.95
        if fmt == "raw_text":
            scores["continued_pretraining"] = 0.7
            scores["pretraining"] = 0.4
        if fmt == "prompt/completion":
            scores["sft"] = 0.85
            scores["instruction"] = 0.7

        # Fine-grained hints.
        if {"code", "unit_tests"} & keys:
            scores["code"] = max(scores.get("code", 0.0), 0.85)
        if {"summary", "document"} <= keys:
            scores["summarization"] = 0.9
        if {"question", "context", "answer"} <= keys:
            scores["extraction"] = 0.85
        if "reward" in keys or {"score", "response"} <= keys:
            scores["reward"] = max(scores.get("reward", 0.0), 0.7)
        if any("image" in k or "audio" in k for k in keys):
            scores["multimodal"] = 0.75

        if not scores:
            scores["sft"] = 0.5

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top_task, top_conf = ranked[0]
        alts = [(name, c) for name, c in ranked[1:4]]

        reason = f"schema={fmt}, top signals from fields {sorted(keys)[:6]}"
        return TaskDetection(task=top_task, confidence=top_conf, alternatives=alts, reason=reason)
