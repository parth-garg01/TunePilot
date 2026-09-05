"""Deduplicate model candidates that appear in multiple sources (PRD 10.6)."""

from __future__ import annotations

from typing import Iterable

from .candidate import ModelCandidate

SOURCE_PRIORITY = {
    "official": 100,
    "huggingface": 90,
    "modelscope": 85,
    "unsloth": 80,
    "ngc": 78,
    "local": 60,
    "user": 55,
}


def _canonical(identifier: str) -> str:
    tail = identifier.rsplit("/", 1)[-1]
    return tail.lower().replace("_", "-").strip()


def _fingerprint(c: ModelCandidate) -> tuple[str, int]:
    return _canonical(c.identifier), c.metadata.parameters


def deduplicate(candidates: Iterable[ModelCandidate]) -> list[ModelCandidate]:
    """Deduplicate by canonical model name + parameter count.

    When two records collide, keep the one from the higher-priority source and
    merge any missing metadata from the losers.
    """
    best: dict[tuple[str, int], ModelCandidate] = {}
    for c in candidates:
        fp = _fingerprint(c)
        current = best.get(fp)
        if current is None:
            best[fp] = c
            continue
        cur_prio = SOURCE_PRIORITY.get(current.source, 0)
        new_prio = SOURCE_PRIORITY.get(c.source, 0)
        if new_prio > cur_prio:
            merged = c
            fallback = current
        else:
            merged = current
            fallback = c
        # Fill any empty fields from the loser.
        if not merged.metadata.license:
            merged.metadata.license = fallback.metadata.license
        if not merged.metadata.architecture:
            merged.metadata.architecture = fallback.metadata.architecture
        if not merged.metadata.context_length:
            merged.metadata.context_length = fallback.metadata.context_length
        if not merged.metadata.parameters:
            merged.metadata.parameters = fallback.metadata.parameters
        if not merged.metadata.quantization_variants:
            merged.metadata.quantization_variants = list(fallback.metadata.quantization_variants)
        merged.provenance.setdefault("alternate_sources", []).append(fallback.source)
        best[fp] = merged
    return list(best.values())
