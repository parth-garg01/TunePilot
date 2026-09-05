"""Local log persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LogPersister:
    root: Path

    def persist(self, name: str, lines: list[str]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{name}.log"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return target


def persist_logs(root: Path | str, name: str, lines: list[str]) -> Path:
    return LogPersister(Path(root)).persist(name, lines)
