"""`orchestrator dataset` — analyze datasets."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print as rprint

from ..dataset import DatasetAnalyzer, TaskDetector
from ..reporting.dataset_report import render_dataset_report

app = typer.Typer()


@app.command("analyze")
def analyze(path: Path, json_out: bool = typer.Option(False, "--json")) -> None:
    analyzer = DatasetAnalyzer()
    report = analyzer.analyze_path(path)
    detection = TaskDetector().detect(report)
    if json_out:
        payload = report.to_dict()
        payload["detected_task"] = {
            "task": detection.task,
            "confidence": detection.confidence,
            "alternatives": detection.alternatives,
        }
        print(json.dumps(payload, indent=2))
        return
    rprint(render_dataset_report(report, detection))
