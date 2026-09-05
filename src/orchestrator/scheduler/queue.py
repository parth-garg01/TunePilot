"""Priority queue with dependencies (PRD section 14, 22)."""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class _QueueEntry:
    priority: int
    order: int
    item: Any = field(compare=False)
    dependencies: set[int] = field(default_factory=set, compare=False)
    id: int = field(default=0, compare=False)


class JobQueue:
    """Priority queue where lower `priority` value = higher priority.

    Jobs with pending dependencies are held until their deps have all been
    marked `resolve()`d. Supports cancellation by id.
    """

    def __init__(self) -> None:
        self._heap: list[_QueueEntry] = []
        self._counter = itertools.count()
        self._done: set[int] = set()
        self._pending: dict[int, _QueueEntry] = {}
        self._cancelled: set[int] = set()

    def push(self, item: Any, *, priority: int = 0, depends_on: set[int] | None = None) -> int:
        job_id = next(self._counter) + 1
        entry = _QueueEntry(
            priority=priority,
            order=job_id,
            item=item,
            dependencies=set(depends_on or []),
            id=job_id,
        )
        self._pending[job_id] = entry
        heapq.heappush(self._heap, entry)
        return job_id

    def pop_ready(self) -> Any | None:
        """Pop the highest-priority entry whose dependencies are satisfied."""
        skipped: list[_QueueEntry] = []
        result: _QueueEntry | None = None
        while self._heap:
            entry = heapq.heappop(self._heap)
            if entry.id in self._cancelled:
                continue
            if not entry.dependencies.issubset(self._done):
                skipped.append(entry)
                continue
            result = entry
            break
        for s in skipped:
            heapq.heappush(self._heap, s)
        if result is None:
            return None
        self._pending.pop(result.id, None)
        return result.item

    def resolve(self, job_id: int) -> None:
        self._done.add(job_id)
        self._pending.pop(job_id, None)

    def cancel(self, job_id: int) -> None:
        self._cancelled.add(job_id)
        self._pending.pop(job_id, None)

    def __len__(self) -> int:
        return len(self._pending)
