"""Soup CLI adapter.

Isolates every call to the Soup CLI. The rest of the orchestrator never
imports Soup directly. The adapter records the Soup version used so that
experiments remain reproducible.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

from ..errors import SoupError
from ..logging_utils import get_logger
from .config_generator import SoupConfig
from .validation import validate_soup_config

log = get_logger(__name__)


@dataclass
class SoupExecutionResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    checkpoints: list[Path] = field(default_factory=list)
    model_dir: Path | None = None


class SoupAdapter:
    """Programmatic wrapper around the `soup` CLI."""

    def __init__(self, binary: str = "soup") -> None:
        self.binary = binary

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def version(self) -> str:
        code, out = self._run([self.binary, "--version"])
        if code != 0:
            return "unknown"
        return out.strip().splitlines()[0] if out else "unknown"

    def install_extras(self, extras: Iterable[str]) -> None:
        for extra in extras:
            code, _ = self._run([self.binary, "install", extra])
            if code != 0:
                raise SoupError(f"Failed to install Soup extra: {extra}")

    def write_config(self, config: SoupConfig, dest: Path | str) -> Path:
        p = Path(dest)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = yaml.safe_load(config.to_yaml())
        validate_soup_config(payload)
        p.write_text(config.to_yaml(), encoding="utf-8")
        log.info("wrote Soup config to %s", p)
        return p

    def validate(self, config_path: Path | str) -> None:
        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        validate_soup_config(payload)

    def train(self, config_path: Path | str, *, cwd: Path | str | None = None) -> SoupExecutionResult:
        cmd = [self.binary, "train", "--config", str(config_path)]
        code, out = self._run(cmd, cwd=cwd)
        model_dir = None
        checkpoints: list[Path] = []
        try:
            payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
            outdir = Path(payload["output"]["dir"])
            if outdir.exists():
                model_dir = outdir
                checkpoints = sorted(outdir.glob("checkpoint-*"))
        except Exception:
            pass
        return SoupExecutionResult(exit_code=code, stdout=out, checkpoints=checkpoints, model_dir=model_dir)

    def merge_adapter(self, adapter_dir: Path | str, dest: Path | str) -> Path:
        code, out = self._run([self.binary, "merge", "--adapter", str(adapter_dir), "--output", str(dest)])
        if code != 0:
            raise SoupError(f"Soup merge failed: {out}")
        return Path(dest)

    def export(self, model_dir: Path | str, dest: Path | str, *, format: str = "safetensors") -> Path:
        code, out = self._run([self.binary, "export", "--model", str(model_dir),
                               "--output", str(dest), "--format", format])
        if code != 0:
            raise SoupError(f"Soup export failed: {out}")
        return Path(dest)

    def infer(self, model_dir: Path | str, prompt: str) -> str:
        code, out = self._run([self.binary, "infer", "--model", str(model_dir), "--prompt", prompt])
        if code != 0:
            raise SoupError(f"Soup infer failed: {out}")
        return out

    def _run(self, cmd: list[str], *, cwd: Path | str | None = None) -> tuple[int, str]:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
            log.debug("$ %s -> %d", " ".join(cmd), proc.returncode)
            return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
        except FileNotFoundError:
            return 127, f"binary not found: {self.binary}"
