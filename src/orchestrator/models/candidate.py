"""Common internal model schema (PRD section 10.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrainingStage(str, Enum):
    BASE = "base"
    INSTRUCT = "instruct"
    RL_TUNED = "rl_tuned"
    MERGED = "merged"
    POST_TRAINED = "post_trained"
    UNCERTAIN = "uncertain"


@dataclass
class ModelMetadata:
    architecture: str = ""
    parameters: int = 0
    context_length: int = 0
    tokenizer: str = ""
    license: str = ""
    supported_tasks: list[str] = field(default_factory=list)
    quantization_variants: list[str] = field(default_factory=list)
    download_size_mb: int = 0
    languages: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    model_card: dict[str, Any] = field(default_factory=dict)
    benchmark_evidence: dict[str, float] = field(default_factory=dict)
    checksum: str | None = None


@dataclass
class ModelCandidate:
    source: str
    identifier: str
    revision: str = ""
    metadata: ModelMetadata = field(default_factory=ModelMetadata)
    stage: TrainingStage = TrainingStage.UNCERTAIN
    stage_confidence: float = 0.5
    stage_reason: str = ""
    downloadable: bool = True
    finetuning_supported: bool = True
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.identifier}@{self.revision or 'HEAD'}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "identifier": self.identifier,
            "revision": self.revision,
            "stage": self.stage.value,
            "stage_confidence": self.stage_confidence,
            "stage_reason": self.stage_reason,
            "downloadable": self.downloadable,
            "finetuning_supported": self.finetuning_supported,
            "provenance": self.provenance,
            "metadata": {
                "architecture": self.metadata.architecture,
                "parameters": self.metadata.parameters,
                "context_length": self.metadata.context_length,
                "tokenizer": self.metadata.tokenizer,
                "license": self.metadata.license,
                "supported_tasks": self.metadata.supported_tasks,
                "quantization_variants": self.metadata.quantization_variants,
                "download_size_mb": self.metadata.download_size_mb,
                "languages": self.metadata.languages,
                "tags": self.metadata.tags,
                "benchmark_evidence": self.metadata.benchmark_evidence,
                "checksum": self.metadata.checksum,
            },
        }


def classify_stage(identifier: str, tags: list[str], card: dict[str, Any]) -> tuple[TrainingStage, float, str]:
    """Classify base/instruct/etc from naming + tags + card text.

    Returns (stage, confidence, reason).
    """
    name = identifier.lower()
    tags_l = [t.lower() for t in tags]
    text = " ".join([name] + tags_l + [str(card.get("description", "")).lower()])

    if any(t in text for t in ("dpo", "rlhf", "orpo", "kto", "ppo")):
        return TrainingStage.RL_TUNED, 0.85, "RL/preference tuning markers in name/tags"
    if any(t in text for t in ("instruct", "chat", "-it", "sft", "assistant")):
        return TrainingStage.INSTRUCT, 0.85, "instruction-tuned markers in name/tags"
    if "merged" in text or "merge" in text:
        return TrainingStage.MERGED, 0.7, "merge markers"
    if any(t in text for t in ("base", "pt", "pretrained")):
        return TrainingStage.BASE, 0.85, "base/pretrained markers"
    return TrainingStage.UNCERTAIN, 0.5, "no explicit training-stage markers"
