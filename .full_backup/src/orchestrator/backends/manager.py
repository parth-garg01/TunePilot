"""Register and route across multiple compute backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..errors import BackendError
from ..logging_utils import get_logger
from .base import ComputeBackend

log = get_logger(__name__)


@dataclass
class BackendManager:
    backends: dict[str, ComputeBackend] = field(default_factory=dict)

    def register(self, backend: ComputeBackend) -> "BackendManager":
        self.backends[backend.name] = backend
        return self

    def get(self, name: str) -> ComputeBackend:
        if name not in self.backends:
            raise BackendError(f"Unknown backend: {name}")
        return self.backends[name]

    def preferred(self, order: Iterable[str]) -> ComputeBackend:
        for name in order:
            b = self.backends.get(name)
            if b and b.validate_credentials():
                return b
        raise BackendError(f"No usable backend among preferred: {list(order)}")

    def names(self) -> list[str]:
        return list(self.backends.keys())
