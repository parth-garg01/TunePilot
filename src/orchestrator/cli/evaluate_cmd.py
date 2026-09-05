"""`orchestrator evaluate` — run evaluation layers on a model directory."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print as rprint

from ..evaluation import (
    AccuracyMetric,
    EvaluationRunner,
    ExactMatchMetric,
    F1Metric,
    ROUGEMetric,
)

app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def evaluate(
    predictions_file: Path = typer.Argument(..., help="JSONL of {prediction, reference}"),
) -> None:
    predictions: list[str] = []
    references: list[str] = []
    with predictions_file.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            predictions.append(str(row.get("prediction", "")))
            references.append(str(row.get("reference", "")))
    runner = EvaluationRunner([AccuracyMetric(), ExactMatchMetric(), F1Metric(), ROUGEMetric()])
    outcome = runner.layer2(predictions, references)
    for name, value in outcome.metrics.items():
        rprint(f"  {name}: {value:.4f}")
