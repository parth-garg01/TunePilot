"""Checkpoint discovery, validation, and resume support (PRD section 17)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REQUIRED_FILES = ("training_state.json",)


@dataclass
class Checkpoint:
    path: Path
    step: int
    valid: bool = True
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class CheckpointManager:
    """Locate and validate checkpoints under an experiment directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _parse_step(self, name: str) -> int:
        for token in name.split("-"):
            if token.isdigit():
                return int(token)
        return 0

    def discover(self) -> list[Checkpoint]:
        if not self.root.exists():
            return []
        out: list[Checkpoint] = []
        for p in sorted(self.root.glob("step-*")):
            out.append(self._validate(p))
        for p in sorted(self.root.glob("checkpoint-*")):
            out.append(self._validate(p))
        out.sort(key=lambda c: c.step)
        return out

    def _validate(self, p: Path) -> Checkpoint:
        step = self._parse_step(p.name)
        ck = Checkpoint(path=p, step=step)
        for f in REQUIRED_FILES:
            if not (p / f).exists():
                ck.valid = False
                ck.warnings.append(f"missing {f}")
        metrics_file = p / "metrics.json"
        if metrics_file.exists():
            try:
                ck.metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            except Exception:
                ck.warnings.append("unreadable metrics.json")
        return ck

    def latest_valid(self) -> Checkpoint | None:
        valid = [c for c in self.discover() if c.valid]
        return valid[-1] if valid else None

    def save_training_state(self, step: int, state: dict[str, Any]) -> Path:
        p = self.root / f"step-{step}"
        p.mkdir(parents=True, exist_ok=True)
        (p / "training_state.json").write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        return p
