"""Candidate filtering (PRD section 10.2)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .candidate import ModelCandidate, TrainingStage


def parse_size(s: str | None) -> int | None:
    """Parse '7B', '350M', '14b' -> parameter count."""
    if not s:
        return None
    m = re.match(r"^\s*([\d\.]+)\s*([MBmb])\s*$", s)
    if not m:
        try:
            return int(s)
        except ValueError:
            return None
    n = float(m.group(1))
    unit = m.group(2).upper()
    return int(n * (1_000_000 if unit == "M" else 1_000_000_000))


@dataclass
class FilterCriteria:
    task: str | None = None
    architectures: list[str] = field(default_factory=list)
    max_parameters: int | None = None
    min_parameters: int | None = None
    licenses: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    min_context_length: int | None = None
    require_quantization: str | None = None  # "4bit" or "8bit"
    require_downloadable: bool = True
    require_finetunable: bool = True
    prefer_base: bool = False
    allow_instruct: bool = True


class CandidateFilter:
    """Apply hard filters plus optional preferences."""

    def __init__(self, criteria: FilterCriteria) -> None:
        self.criteria = criteria

    def apply(self, candidates: Iterable[ModelCandidate]) -> list[ModelCandidate]:
        c = self.criteria
        out: list[ModelCandidate] = []
        for cand in candidates:
            if c.require_downloadable and not cand.downloadable:
                continue
            if c.require_finetunable and not cand.finetuning_supported:
                continue
            if c.max_parameters and cand.metadata.parameters and cand.metadata.parameters > c.max_parameters:
                continue
            if c.min_parameters and cand.metadata.parameters and cand.metadata.parameters < c.min_parameters:
                continue
            if c.architectures and cand.metadata.architecture and cand.metadata.architecture not in c.architectures:
                continue
            if c.licenses and cand.metadata.license and cand.metadata.license not in c.licenses:
                continue
            if c.min_context_length and cand.metadata.context_length and cand.metadata.context_length < c.min_context_length:
                continue
            if c.require_quantization and c.require_quantization not in cand.metadata.quantization_variants:
                continue
            if c.task and cand.metadata.supported_tasks and c.task not in cand.metadata.supported_tasks:
                # allow overlap for chat/sft/instruction family
                family = {"sft", "chat", "instruction"}
                if not (c.task in family and set(cand.metadata.supported_tasks) & family):
                    continue
            if not c.allow_instruct and cand.stage == TrainingStage.INSTRUCT:
                continue
            out.append(cand)
        return out
