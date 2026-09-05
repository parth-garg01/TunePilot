"""Load and save orchestrator configurations from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..errors import ConfigError
from .schema import OrchestratorConfig


def default_config_path(project_root: Path | str = ".") -> Path:
    return Path(project_root) / "orchestrator.yaml"


def load_config(path: Path | str | None = None) -> OrchestratorConfig:
    p = Path(path) if path else default_config_path()
    if not p.exists():
        raise ConfigError(f"Configuration file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML at {p}: {e}") from e
    try:
        return OrchestratorConfig.model_validate(raw)
    except Exception as e:
        raise ConfigError(f"Invalid configuration in {p}: {e}") from e


def save_config(config: OrchestratorConfig, path: Path | str | None = None) -> Path:
    p = Path(path) if path else default_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, indent=2),
        encoding="utf-8",
    )
    return p
