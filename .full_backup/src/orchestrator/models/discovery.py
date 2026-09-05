"""Multi-source model discovery pipeline (PRD sections 10.1, 10.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..logging_utils import get_logger
from .candidate import ModelCandidate
from .dedupe import SOURCE_PRIORITY, deduplicate
from .sources import ModelSource

log = get_logger(__name__)


@dataclass
class ModelDiscovery:
    sources: list[ModelSource] = field(default_factory=list)

    def add(self, source: ModelSource) -> "ModelDiscovery":
        self.sources.append(source)
        return self

    def discover(self, query: str = "", *, per_source_limit: int = 15) -> list[ModelCandidate]:
        seen: list[ModelCandidate] = []
        ordered = sorted(self.sources, key=lambda s: -SOURCE_PRIORITY.get(getattr(s, "name", ""), 0))
        for src in ordered:
            try:
                found = src.search(query, limit=per_source_limit)
                log.info("source %s returned %s candidates", src.name, len(found))
                seen.extend(found)

            except Exception as e:
                log.warning("source %s failed: %s", src.name, e)
        return deduplicate(seen)
