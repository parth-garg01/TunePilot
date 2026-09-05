"""Formatter for dataset analysis output."""

from __future__ import annotations

from ..dataset.analyzer import DatasetReport
from ..dataset.task_detector import TaskDetection


def render_dataset_report(report: DatasetReport, detection: TaskDetection | None = None) -> str:
    body = report.render()
    if detection:
        body += "\n\n" + detection.render()
    return body
