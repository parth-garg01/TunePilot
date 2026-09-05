"""Soup configuration generator.

Generates a `soup.yaml` matching the schema shown in PRD section 12. The
resulting config is validated separately in `validation.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import yaml

from ..config.schema import OrchestratorConfig
from ..dataset.analyzer import DatasetReport
from ..hardware.feasibility import FeasibilityDecision
from ..models.candidate import ModelCandidate


@dataclass
class SoupTrainingSection:
    method: str = "lora"
    quantization: str = "auto"
    epochs: int = 2
    learning_rate: float = 2e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_seq_length: int | None = None
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    seed: int = 42


@dataclass
class SoupConfig:
    base: str
    data: dict[str, Any]
    training: SoupTrainingSection
    output: dict[str, Any]
    reproducibility: dict[str, Any] = field(default_factory=dict)

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            {
                "base": self.base,
                "data": self.data,
                "training": asdict(self.training),
                "output": self.output,
                "reproducibility": self.reproducibility,
            },
            sort_keys=False,
            indent=2,
        )


class SoupConfigGenerator:
    """Turn (candidate, dataset, feasibility, orchestrator config) -> SoupConfig."""

    def generate(
        self,
        *,
        candidate: ModelCandidate,
        dataset: DatasetReport,
        feasibility: FeasibilityDecision,
        config: OrchestratorConfig,
        output_dir: Path | str = "/workspace/output",
        data_dir: Path | str = "/workspace/data",
    ) -> SoupConfig:
        t = config.training

        method = t.method
        if method == "auto":
            method = feasibility.strategy if feasibility.strategy in {"lora", "qlora", "full"} else "lora"

        quant = t.quantization
        if quant == "auto":
            quant = "4bit" if method == "qlora" else ("8bit" if method == "lora" else "none")

        max_seq = t.max_seq_length or max(512, min(int(dataset.lengths.get("p95", 1024)), 4096))
        training = SoupTrainingSection(
            method=method,
            quantization=quant,
            epochs=t.epochs,
            learning_rate=2e-5,
            batch_size=1,
            gradient_accumulation_steps=16,
            max_seq_length=max_seq,
            gradient_checkpointing=True,
            mixed_precision="bf16",
            seed=t.seed,
        )

        train_file = str(Path(data_dir) / "train.jsonl")
        data = {"train": train_file, "val_split": 0.1}

        output = {"dir": str(output_dir)}
        repro = {
            "dataset_fingerprint": dataset.fingerprint,
            "base_revision": candidate.revision,
            "base_source": candidate.source,
            "seed": t.seed,
        }

        return SoupConfig(
            base=candidate.identifier,
            data=data,
            training=training,
            output=output,
            reproducibility=repro,
        )
