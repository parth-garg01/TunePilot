from pathlib import Path

import pytest
import yaml

from orchestrator.config import DatasetConfig, OrchestratorConfig, ProjectConfig
from orchestrator.dataset import DatasetAnalyzer
from orchestrator.hardware.feasibility import feasibility
from orchestrator.hardware.detect import kaggle_t4x2
from orchestrator.models import HuggingFaceSource, ModelDiscovery
from orchestrator.soup import SoupConfigGenerator, validate_soup_config
from orchestrator.soup.validation import SoupValidationError


def _cfg() -> OrchestratorConfig:
    return OrchestratorConfig(
        project=ProjectConfig(name="demo"),
        dataset=DatasetConfig(path="unused"),
    )


def test_generate_soup_config_is_valid(sample_jsonl: Path):
    report = DatasetAnalyzer().analyze_path(sample_jsonl)
    candidate = ModelDiscovery().add(HuggingFaceSource()).discover("")[0]
    feas = feasibility(candidate, kaggle_t4x2())
    soup_cfg = SoupConfigGenerator().generate(
        candidate=candidate, dataset=report, feasibility=feas, config=_cfg(),
    )
    payload = yaml.safe_load(soup_cfg.to_yaml())
    validate_soup_config(payload)


def test_validation_rejects_bad_method():
    with pytest.raises(SoupValidationError):
        validate_soup_config({
            "base": "x",
            "data": {"train": "y"},
            "output": {"dir": "z"},
            "training": {"method": "wrong", "quantization": "none"},
        })
