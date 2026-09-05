"""`orchestrator train` — submit training jobs."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from ..artifacts.layout import ProjectLayout
from ..backends import BackendManager, KaggleBackend, LocalSoup4GBBackend
from ..backends.base import JobSubmission
from ..config import load_config
from ..dataset import DatasetAnalyzer, TaskDetector
from ..hardware.detect import detect_hardware, kaggle_t4x2
from ..jobs.records import ExperimentRecord
from ..jobs.registry import ExperimentRegistry
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
from ..planning.guardrails import CostGuardrail
from ..scheduler import JobScheduler, RetryPolicy
from ..scheduler.scheduler import ScheduledJob
from ..soup.config_generator import SoupConfigGenerator

app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def train(
    config: Path = typer.Option(Path("orchestrator.yaml")),
    backend: str = typer.Option("kaggle", help="Backend name."),
    approve: bool = typer.Option(False, "--approve", help="Bypass explicit-approval guardrail for medium jobs."),
) -> None:
    cfg = load_config(config)
    layout = ProjectLayout(root=Path(cfg.artifacts.root), name=cfg.project.name)
    layout.ensure()

    analyzer = DatasetAnalyzer()
    ds_report = analyzer.analyze_path(cfg.dataset.path)
    detection = TaskDetector().detect(ds_report)

    hw = kaggle_t4x2() if backend == "kaggle" else detect_hardware()

    discovery = ModelDiscovery().add(OfficialProviderSource()).add(HuggingFaceSource()).add(UnslothSource())
    filtered = CandidateFilter(FilterCriteria(
        task=detection.task,
        max_parameters=parse_size(cfg.model_selection.max_parameters),
        licenses=cfg.model_selection.licenses,
    )).apply(discovery.discover(cfg.goal.description))
    ranked = ModelRanker().rank(filtered, task=detection.task, hardware=hw,
                                 dataset_tokens=ds_report.tokens_est)
    if not ranked:
        rprint("[red]no ranked candidates[/red]")
        raise typer.Exit(1)

    top = ranked[0].candidate
    feas = ranked[0].feasibility
    soup_cfg = SoupConfigGenerator().generate(
        candidate=top, dataset=ds_report, feasibility=feas, config=cfg,
    )
    config_path = layout.configs / f"soup-{top.identifier.replace('/', '_')}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(soup_cfg.to_yaml(), encoding="utf-8")

    guard = CostGuardrail(available_gpu_hours=cfg.compute.max_gpu_hours)
    decision = guard.check(estimated_gpu_hours=1.5, approved_by_user=approve)
    rprint(decision.render())
    if not decision.approved:
        raise typer.Exit(2)

    backends = BackendManager().register(LocalSoup4GBBackend()).register(KaggleBackend())
    registry = ExperimentRegistry(layout.db)
    project = registry.ensure_project(cfg.project.name, cfg.goal.description)
    exp = registry.create_experiment(ExperimentRecord(
        id=None, project_id=project.id, name=f"{top.identifier}",
        model_identifier=top.identifier, dataset_fingerprint=ds_report.fingerprint,
        soup_config=soup_cfg.__dict__ | {"training": soup_cfg.training.__dict__},
    ))
    submission = JobSubmission(
        name=f"{cfg.project.name}-{top.identifier}",
        soup_config_path=config_path,
        dataset_path=Path(cfg.dataset.path),
        hardware=hw,
        max_hours=cfg.compute.max_gpu_hours,
    )
    scheduler = JobScheduler(registry=registry, backends=backends,
                              max_concurrent=cfg.parallelism.max_concurrent_jobs,
                              retry_policy=RetryPolicy())
    scheduler.enqueue(ScheduledJob(experiment_id=exp.id, backend_name=backend, submission=submission))
    scheduler.dispatch_ready()
    rprint(f"[green]submitted[/green] experiment={exp.id} backend={backend} config={config_path}")
