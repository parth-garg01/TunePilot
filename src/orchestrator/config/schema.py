"""Pydantic models for orchestrator configuration.

Mirrors the YAML schema shown in section 27 of the PRD. Credentials are
never stored here.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


TaskType = Literal[
    "sft",
    "chat",
    "instruction",
    "classification",
    "dpo",
    "kto",
    "orpo",
    "reasoning",
    "code",
    "summarization",
    "extraction",
    "embedding",
    "reward",
    "pretraining",
    "continued_pretraining",
    "multimodal",
    "auto",
]

Objective = Literal[
    "quality_first",
    "quality",
    "quality_per_dollar",
    "quality_per_gpu_hour",
    "speed",
    "memory_efficiency",
    "balanced",
]


class ProjectConfig(BaseModel):
    name: str = Field(min_length=1)
    description: str | None = None


class DatasetConfig(BaseModel):
    path: str
    format: Literal["auto", "json", "jsonl", "csv", "parquet", "hf"] = "auto"
    train_split: str | None = None
    val_split: str | None = None
    test_split: str | None = None


class GoalConfig(BaseModel):
    type: TaskType = "auto"
    description: str = ""


class ModelSelectionConfig(BaseModel):
    automatic: bool = True
    candidates: int = Field(default=3, ge=1, le=10)
    max_parameters: str | None = "14B"
    allow_instruct_tuned: bool = True
    prefer_base: bool = False
    licenses: list[str] = Field(default_factory=lambda: ["apache-2.0", "mit", "llama", "gemma", "qwen"])


class TrainingConfig(BaseModel):
    framework: Literal["soup"] = "soup"
    method: Literal["auto", "full", "lora", "qlora"] = "auto"
    quantization: Literal["auto", "none", "8bit", "4bit"] = "auto"
    pilot: bool = True
    epochs: int = Field(default=2, ge=1)
    max_seq_length: int | None = None
    seed: int = 42


class ComputeConfig(BaseModel):
    preferred: list[str] = Field(default_factory=lambda: ["kaggle"])
    strategy: Literal["local", "cloud", "auto"] = "auto"
    max_gpu_hours: float = Field(default=10.0, ge=0.0)


class ParallelismConfig(BaseModel):
    enabled: bool = True
    max_concurrent_jobs: int = Field(default=2, ge=1)


class SelectionConfig(BaseModel):
    objective: Objective = "balanced"
    max_training_hours: float | None = None
    max_model_size: str | None = None


class ArtifactsConfig(BaseModel):
    backend: Literal["local"] = "local"
    root: str = "./projects"


class OptimizationConfig(BaseModel):
    objective: Objective = "quality_first"


class ConstraintsConfig(BaseModel):
    max_training_hours: float | None = None
    max_cost: float | None = None
    max_model_size: str | None = None


class OrchestratorConfig(BaseModel):
    project: ProjectConfig
    dataset: DatasetConfig
    goal: GoalConfig = Field(default_factory=GoalConfig)
    model_selection: ModelSelectionConfig = Field(default_factory=ModelSelectionConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    constraints: ConstraintsConfig = Field(default_factory=ConstraintsConfig)

    @field_validator("compute")
    @classmethod
    def _validate_preferred(cls, v: ComputeConfig) -> ComputeConfig:
        allowed = {"kaggle", "local", "colab", "runpod", "lambda", "modal", "vast", "lightning"}
        for name in v.preferred:
            if name not in allowed:
                raise ValueError(f"Unknown compute backend: {name}")
        return v
