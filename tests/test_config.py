from pathlib import Path

import pytest

from orchestrator.config import (
    ArtifactsConfig,
    DatasetConfig,
    OrchestratorConfig,
    ProjectConfig,
    load_config,
    save_config,
)
from orchestrator.errors import ConfigError


def test_roundtrip(tmp_path: Path):
    cfg = OrchestratorConfig(
        project=ProjectConfig(name="demo"),
        dataset=DatasetConfig(path="data/train.jsonl"),
        artifacts=ArtifactsConfig(root=str(tmp_path)),
    )
    path = tmp_path / "orchestrator.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.project.name == "demo"


def test_bad_backend_rejected(tmp_path: Path):
    bad = tmp_path / "orchestrator.yaml"
    bad.write_text(
        "project: {name: demo}\n"
        "dataset: {path: data.jsonl}\n"
        "compute: {preferred: [does-not-exist]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_config(bad)
