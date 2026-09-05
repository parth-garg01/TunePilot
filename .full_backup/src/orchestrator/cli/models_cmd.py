"""`orchestrator models` — discover, filter, rank models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

from ..config import load_config
from ..hardware.detect import detect_hardware
from ..models import (
    CandidateFilter,
    FilterCriteria,
    HuggingFaceSource,
    LocalRegistrySource,
    ModelDiscovery,
    ModelRanker,
    ModelScopeSource,
    NgcSource,
    OfficialProviderSource,
    SelectionPolicy,
    UnslothSource,
)
from ..models.filters import parse_size
from ..models.selection import Objective
from ..reporting.selection_report import render_selection_report

app = typer.Typer()


def _default_discovery() -> ModelDiscovery:
    return (
        ModelDiscovery()
        .add(OfficialProviderSource())
        .add(HuggingFaceSource())
        .add(ModelScopeSource())
        .add(UnslothSource())
        .add(NgcSource())
        .add(LocalRegistrySource())
    )


@app.command("discover")
def discover(query: str = typer.Argument("", help="Free-text query."), limit: int = 20) -> None:
    d = _default_discovery()
    candidates = d.discover(query, per_source_limit=limit)
    for c in candidates[:limit]:
        rprint(f"- [{c.source}] {c.identifier}  params={c.metadata.parameters}  license={c.metadata.license}")


@app.command("rank")
def rank(
    config: Path = typer.Option(Path("orchestrator.yaml")),
    task: str = typer.Option("sft"),
    query: str = typer.Argument("", help="Optional discovery query."),
) -> None:
    cfg = load_config(config)
    hw = detect_hardware()
    d = _default_discovery()
    candidates = d.discover(query)
    criteria = FilterCriteria(
        task=task,
        max_parameters=parse_size(cfg.model_selection.max_parameters),
        licenses=cfg.model_selection.licenses,
        allow_instruct=cfg.model_selection.allow_instruct_tuned,
    )
    filtered = CandidateFilter(criteria).apply(candidates)
    ranker = ModelRanker()
    ranked = ranker.rank(filtered, task=task, hardware=hw)
    policy = SelectionPolicy(Objective(cfg.selection.objective))
    selection = policy.select(ranked)
    rprint(render_selection_report(ranked, selection))
