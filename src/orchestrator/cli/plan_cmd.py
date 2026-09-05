"""`orchestrator plan` — build pilot + full training plan."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from ..config import load_config
from ..dataset import DatasetAnalyzer, TaskDetector
from ..hardware.detect import detect_hardware
from ..models import (
    CandidateFilter,
    FilterCriteria,
    HuggingFaceSource,
    ModelDiscovery,
    ModelRanker,
    OfficialProviderSource,
    UnslothSource,
)
from ..models.filters import parse_size
from ..planning.experiment import ExperimentPlanner
from ..soup.config_generator import SoupConfigGenerator

app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def plan(config: Path = typer.Option(Path("orchestrator.yaml"))) -> None:
    cfg = load_config(config)
    hw = detect_hardware()
    analyzer = DatasetAnalyzer()
    ds_report = analyzer.analyze_path(cfg.dataset.path)
    detection = TaskDetector().detect(ds_report)

    discovery = ModelDiscovery().add(OfficialProviderSource()).add(HuggingFaceSource()).add(UnslothSource())
    candidates = discovery.discover(cfg.goal.description)
    criteria = FilterCriteria(
        task=detection.task,
        max_parameters=parse_size(cfg.model_selection.max_parameters),
        licenses=cfg.model_selection.licenses,
    )
    filtered = CandidateFilter(criteria).apply(candidates)
    ranked = ModelRanker().rank(filtered, task=detection.task, hardware=hw,
                                 dataset_tokens=ds_report.tokens_est)
    plan = ExperimentPlanner(default_candidates=cfg.model_selection.candidates).build(
        ranked,
        available_gpu_hours=cfg.compute.max_gpu_hours,
    )
    rprint(plan.render())

    if plan.candidates:
        top = plan.candidates[0]
        feas = ranked[0].feasibility
        gen = SoupConfigGenerator()
        soup_cfg = gen.generate(candidate=top, dataset=ds_report, feasibility=feas, config=cfg)
        rprint("\nGenerated Soup config for top candidate:\n")
        rprint(soup_cfg.to_yaml())
