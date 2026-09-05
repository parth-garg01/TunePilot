"""`orchestrator run` — one-shot end-to-end pipeline (PRD section 31)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

from ..artifacts.layout import ProjectLayout
from ..backends import BackendManager, KaggleBackend, LocalSoup4GBBackend
from ..backends.base import JobSubmission
from ..config import (
    DatasetConfig,
    GoalConfig,
    ModelSelectionConfig,
    OrchestratorConfig,
    ProjectConfig,
    load_config,
)
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
    SelectionPolicy,
    UnslothSource,
)
from ..models.filters import parse_size
from ..models.selection import Objective
from ..planning.experiment import ExperimentPlanner
from ..planning.guardrails import CostGuardrail
from ..planning.sanity import (
    after_dataset_analysis,
    before_dataset_processing,
    before_model_selection,
)
from ..reporting.dataset_report import render_dataset_report
from ..reporting.selection_report import render_selection_report
from ..scheduler import JobScheduler, RetryPolicy
from ..scheduler.scheduler import ScheduledJob
from ..soup.config_generator import SoupConfigGenerator

app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def run(
    dataset: Path = typer.Option(..., "--dataset"),
    goal: str = typer.Option("Create a helpful assistant", "--goal"),
    backend: str = typer.Option("kaggle", "--backend"),
    config: Optional[Path] = typer.Option(None, "--config"),
    approve: bool = typer.Option(False, "--approve", help="Skip approval prompt for medium-sized jobs."),
) -> None:
    cfg = load_config(config) if config else _default_config(dataset, goal)
    layout = ProjectLayout(root=Path(cfg.artifacts.root), name=cfg.project.name)
    layout.ensure()

    rprint("[bold]Analyzing dataset...[/bold]\n")
    sanity = before_dataset_processing(cfg.dataset.path, supported={"json", "jsonl", "csv", "parquet"})
    rprint(sanity.render())
    if not sanity.passed:
        raise typer.Exit(1)

    ds_report = DatasetAnalyzer().analyze_path(cfg.dataset.path)
    detection = TaskDetector().detect(ds_report)
    rprint(render_dataset_report(ds_report, detection))

    sanity = after_dataset_analysis(ds_report)
    rprint(sanity.render())
    if not sanity.passed:
        raise typer.Exit(2)

    rprint("\n[bold]Searching compatible models...[/bold]\n")
    hw = kaggle_t4x2() if backend == "kaggle" else detect_hardware()
    discovery = ModelDiscovery().add(OfficialProviderSource()).add(HuggingFaceSource()).add(UnslothSource())
    candidates = discovery.discover(goal)
    filtered = CandidateFilter(FilterCriteria(
        task=detection.task,
        max_parameters=parse_size(cfg.model_selection.max_parameters),
        licenses=cfg.model_selection.licenses,
    )).apply(candidates)
    ranked = ModelRanker().rank(filtered, task=detection.task, hardware=hw,
                                 dataset_tokens=ds_report.tokens_est)
    sanity = before_model_selection(filtered, detection)
    rprint(sanity.render())
    if not sanity.passed:
        raise typer.Exit(3)

    policy = SelectionPolicy(Objective(cfg.selection.objective))
    selection = policy.select(ranked)
    rprint(render_selection_report(ranked, selection))
    if not selection.winner:
        raise typer.Exit(4)

    rprint("\n[bold]Planning pilot experiments...[/bold]\n")
    plan = ExperimentPlanner(default_candidates=cfg.model_selection.candidates).build(
        ranked, available_gpu_hours=cfg.compute.max_gpu_hours,
    )
    rprint(plan.render())

    rprint("\n[bold]Checking guardrails...[/bold]")
    decision = CostGuardrail(available_gpu_hours=cfg.compute.max_gpu_hours).check(
        estimated_gpu_hours=1.5, approved_by_user=approve,
    )
    rprint(decision.render())
    if not decision.approved:
        raise typer.Exit(5)

    rprint("\n[bold]Submitting jobs...[/bold]\n")
    backends = BackendManager().register(LocalSoup4GBBackend()).register(KaggleBackend())
    registry = ExperimentRegistry(layout.db)
    project = registry.ensure_project(cfg.project.name, cfg.goal.description)

    scheduler = JobScheduler(
        registry=registry, backends=backends,
        max_concurrent=cfg.parallelism.max_concurrent_jobs, retry_policy=RetryPolicy(),
    )

    for candidate in plan.candidates:
        r_feas = next((r for r in ranked if r.candidate.key == candidate.key), None)
        feas = r_feas.feasibility if r_feas else None
        soup_cfg = SoupConfigGenerator().generate(
            candidate=candidate, dataset=ds_report, feasibility=feas, config=cfg,
        )
        cfg_path = layout.configs / f"soup-{candidate.identifier.replace('/', '_')}.yaml"
        cfg_path.write_text(soup_cfg.to_yaml(), encoding="utf-8")
        exp = registry.create_experiment(ExperimentRecord(
            id=None, project_id=project.id, name=candidate.identifier,
            model_identifier=candidate.identifier, dataset_fingerprint=ds_report.fingerprint,
            soup_config=soup_cfg.__dict__ | {"training": soup_cfg.training.__dict__},
        ))
        submission = JobSubmission(
            name=f"{cfg.project.name}-{candidate.identifier}",
            soup_config_path=cfg_path,
            dataset_path=Path(cfg.dataset.path),
            hardware=hw,
            max_hours=cfg.compute.max_gpu_hours / max(1, len(plan.candidates)),
        )
        scheduler.enqueue(ScheduledJob(experiment_id=exp.id, backend_name=backend, submission=submission))
    scheduler.dispatch_ready()
    rprint("[green]All pilot jobs submitted.[/green]")
    registry.close()


def _default_config(dataset: Path, goal: str) -> OrchestratorConfig:
    return OrchestratorConfig(
        project=ProjectConfig(name=f"run-{dataset.stem}"[:40]),
        dataset=DatasetConfig(path=str(dataset)),
        goal=GoalConfig(description=goal),
        model_selection=ModelSelectionConfig(),
    )
