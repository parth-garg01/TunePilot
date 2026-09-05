"""CLI-friendly status output (PRD section 33)."""

from __future__ import annotations

from typing import Iterable

from ..jobs.records import JobRecord


def format_job_table(jobs: Iterable[JobRecord]) -> str:
    header = f"{'JOB':<5}{'MODEL':<20}{'STATUS':<10}{'GPU':<6}{'TIME':<8}PROGRESS"
    lines = [header]
    for j in jobs:
        gpu = (j.gpu_type or "-")[:5]
        started = j.started_at or ""
        completed = j.completed_at or ""
        time_str = "-"
        if started:
            time_str = f"{(_seconds_between(started, completed) or 0) // 60:.0f}m"
        progress = "-"
        if j.status == "COMPLETED":
            progress = "100%"
        elif j.status == "RUNNING":
            progress = "~"
        model = (j.error or "")[:18] if j.status == "FAILED" else str(j.experiment_id)
        lines.append(f"{j.id:<5}{model[:18]:<20}{j.status:<10}{gpu:<6}{time_str:<8}{progress}")
    return "\n".join(lines)


def _seconds_between(a: str, b: str) -> float | None:
    try:
        from datetime import datetime
        return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() if b else None
    except Exception:
        return None


def format_pipeline_status(stage: str, detail: str = "") -> str:
    tail = f" — {detail}" if detail else ""
    return f"[{stage}]{tail}"
