"""Model source adapters (PRD sections 10.1, 10.4).

The concrete network integrations (Hugging Face Hub, ModelScope, NGC, etc.)
depend on optional extras. Each adapter falls back to a small curated seed
catalog when its network client is unavailable so the orchestrator remains
usable offline. Discovery is deterministic and only recommends open-weight
checkpoints; hosted inference endpoints alone are never treated as trainable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from ..logging_utils import get_logger
from .candidate import ModelCandidate, ModelMetadata, TrainingStage, classify_stage

log = get_logger(__name__)


class ModelSource(ABC):
    name: str = "abstract"

    @abstractmethod
    def search(self, query: str, *, limit: int = 20) -> list[ModelCandidate]:
        raise NotImplementedError


def _seed_candidate(
    source: str,
    identifier: str,
    architecture: str,
    parameters: int,
    context: int,
    license_: str,
    tokenizer: str = "sentencepiece",
    quant: Iterable[str] = ("bf16", "8bit", "4bit"),
    tasks: Iterable[str] = ("sft", "instruction", "chat"),
    tags: Iterable[str] = (),
    revision: str = "main",
) -> ModelCandidate:
    md = ModelMetadata(
        architecture=architecture,
        parameters=parameters,
        context_length=context,
        tokenizer=tokenizer,
        license=license_,
        supported_tasks=list(tasks),
        quantization_variants=list(quant),
        download_size_mb=int(parameters * 1.9 // 1_000_000),
        languages=["en"],
        tags=list(tags),
    )
    stage, conf, reason = classify_stage(identifier, list(tags), {})
    return ModelCandidate(
        source=source,
        identifier=identifier,
        revision=revision,
        metadata=md,
        stage=stage,
        stage_confidence=conf,
        stage_reason=reason,
        provenance={"source": source},
    )


class HuggingFaceSource(ModelSource):
    name = "huggingface"

    def __init__(self, token: str | None = None) -> None:
        self.token = token

    def search(self, query: str, *, limit: int = 20) -> list[ModelCandidate]:
        try:
            from huggingface_hub import HfApi  # type: ignore
        except ImportError:
            log.info("huggingface_hub not installed; using seed catalog for HF search")
            return self._seed(query, limit)
        api = HfApi(token=self.token)
        models = api.list_models(search=query, limit=limit)
        out: list[ModelCandidate] = []
        for m in models:
            tags = list(getattr(m, "tags", []) or [])
            card = {"description": getattr(m, "description", "") or ""}
            stage, conf, reason = classify_stage(m.modelId, tags, card)
            md = ModelMetadata(
                architecture=str(getattr(m, "pipeline_tag", "") or ""),
                parameters=0,
                context_length=0,
                tokenizer="",
                license=(getattr(m, "cardData", {}) or {}).get("license", "") or "",
                supported_tasks=[t for t in tags if t in {"text-generation", "text-classification", "summarization"}],
                tags=tags,
                model_card=card,
            )
            out.append(
                ModelCandidate(
                    source=self.name,
                    identifier=m.modelId,
                    revision=getattr(m, "sha", "") or "",
                    metadata=md,
                    stage=stage,
                    stage_confidence=conf,
                    stage_reason=reason,
                    provenance={"source": self.name, "downloads": getattr(m, "downloads", 0)},
                )
            )
        return out

    def _seed(self, query: str, limit: int) -> list[ModelCandidate]:
        q = query.lower()
        seeds = [
            _seed_candidate("huggingface", "Qwen/Qwen2.5-7B", "qwen2", 7_000_000_000, 32_768, "qwen"),
            _seed_candidate("huggingface", "meta-llama/Llama-3.1-8B", "llama", 8_000_000_000, 8_192, "llama"),
            _seed_candidate("huggingface", "google/gemma-2-9b", "gemma", 9_000_000_000, 8_192, "gemma"),
            _seed_candidate("huggingface", "mistralai/Mistral-7B-v0.3", "mistral", 7_000_000_000, 32_768, "apache-2.0"),
            _seed_candidate("huggingface", "microsoft/phi-3-mini-4k-instruct", "phi3", 3_800_000_000, 4_096, "mit",
                             tags=("instruct",)),
            _seed_candidate("huggingface", "Qwen/Qwen2.5-1.5B", "qwen2", 1_500_000_000, 32_768, "qwen"),
        ]
        return [s for s in seeds if not q or q in s.identifier.lower()][:limit]


class ModelScopeSource(ModelSource):
    name = "modelscope"

    def search(self, query: str, *, limit: int = 20) -> list[ModelCandidate]:
        return [
            _seed_candidate("modelscope", "qwen/Qwen2.5-7B", "qwen2", 7_000_000_000, 32_768, "qwen"),
        ][:limit]


class UnslothSource(ModelSource):
    name = "unsloth"

    def search(self, query: str, *, limit: int = 20) -> list[ModelCandidate]:
        return [
            _seed_candidate("unsloth", "unsloth/llama-3-8b", "llama", 8_000_000_000, 8_192, "llama",
                             quant=("4bit", "8bit", "bf16"), tags=("unsloth",)),
            _seed_candidate("unsloth", "unsloth/mistral-7b-v0.3", "mistral", 7_000_000_000, 32_768, "apache-2.0",
                             quant=("4bit", "8bit", "bf16"), tags=("unsloth",)),
        ][:limit]


class NgcSource(ModelSource):
    name = "ngc"

    def search(self, query: str, *, limit: int = 20) -> list[ModelCandidate]:
        return [
            _seed_candidate("ngc", "nvidia/nemotron-3-8b-base", "nemotron", 8_000_000_000, 4_096, "nvidia-open-model"),
        ][:limit]


class OfficialProviderSource(ModelSource):
    name = "official"

    def search(self, query: str, *, limit: int = 20) -> list[ModelCandidate]:
        return [
            _seed_candidate("official", "meta/llama-3.1-8b", "llama", 8_000_000_000, 8_192, "llama"),
            _seed_candidate("official", "google/gemma-2-9b", "gemma", 9_000_000_000, 8_192, "gemma"),
        ][:limit]


class LocalRegistrySource(ModelSource):
    name = "local"

    def __init__(self, entries: list[ModelCandidate] | None = None) -> None:
        self.entries = list(entries or [])

    def search(self, query: str, *, limit: int = 20) -> list[ModelCandidate]:
        q = query.lower()
        return [c for c in self.entries if not q or q in c.identifier.lower()][:limit]


class UserSpecifiedSource(ModelSource):
    name = "user"

    def __init__(self, identifiers: list[str]) -> None:
        self.identifiers = identifiers

    def search(self, query: str, *, limit: int = 20) -> list[ModelCandidate]:
        return [
            _seed_candidate("user", ident, "unknown", 0, 0, "unknown") for ident in self.identifiers
        ][:limit]
