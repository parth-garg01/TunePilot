"""Pipeline-wide sanity checks (PRD section 32)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..errors import SanityCheckError


@dataclass
class SanityResult:
    stage: str
    passed: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"[{self.stage}] {'OK' if self.passed else 'FAIL'}"]
        for name, ok, msg in self.checks:
            mark = "" if ok else " (fail)"
            extra = f" — {msg}" if msg else ""
            lines.append(f"  {'+' if ok else '-'} {name}{mark}{extra}")
        return "\n".join(lines)


def run_checks(stage: str, checks: list[tuple[str, Callable[[], Any]]]) -> SanityResult:
    result = SanityResult(stage=stage, passed=True)
    for name, fn in checks:
        try:
            fn()
            result.checks.append((name, True, ""))
        except Exception as e:
            result.checks.append((name, False, str(e)))
            result.passed = False
    return result


def before_dataset_processing(path: Path | str, *, supported: set[str]) -> SanityResult:
    p = Path(path)

    def _exists():
        if not p.exists() and ":" not in str(path):
            raise SanityCheckError(f"file not found: {p}")

    def _readable():
        if p.exists() and not p.is_file() and not p.is_dir():
            raise SanityCheckError(f"path is not readable: {p}")

    def _format():
        if p.suffix.lstrip(".").lower() not in supported and ":" not in str(path) and p.is_file():
            raise SanityCheckError(f"unsupported format: {p.suffix}")

    return run_checks(
        "before-dataset-processing",
        [("file exists", _exists), ("readable", _readable), ("supported format", _format)],
    )


def after_dataset_analysis(report) -> SanityResult:
    def _has_examples():
        if report.n_examples <= 0:
            raise SanityCheckError("no examples detected")

    def _has_fields():
        if not report.schema:
            raise SanityCheckError("no fields detected in schema")

    def _not_corrupt():
        if report.quality.malformed > report.n_examples * 0.5:
            raise SanityCheckError("more than 50% of records are malformed")

    return run_checks(
        "after-dataset-analysis",
        [("examples > 0", _has_examples), ("fields detected", _has_fields), ("no severe corruption", _not_corrupt)],
    )


def before_model_selection(candidates: list, detection) -> SanityResult:
    def _task():
        if not detection or not detection.task:
            raise SanityCheckError("task not detected")

    def _candidates():
        if not candidates:
            raise SanityCheckError("candidate list is empty")

    return run_checks("before-model-selection", [("task detected", _task), ("candidates non-empty", _candidates)])


def before_training(*, candidate, feasibility, soup_config_valid: bool, credentials_valid: bool, quota_ok: bool) -> SanityResult:
    def _downloadable():
        if not candidate.downloadable:
            raise SanityCheckError("candidate weights are not downloadable")

    def _feasible():
        if not feasibility.feasible:
            raise SanityCheckError(feasibility.reason)

    def _config():
        if not soup_config_valid:
            raise SanityCheckError("Soup configuration is invalid")

    def _creds():
        if not credentials_valid:
            raise SanityCheckError("backend credentials are invalid")

    def _quota():
        if not quota_ok:
            raise SanityCheckError("insufficient compute quota")

    return run_checks(
        "before-training",
        [
            ("model downloadable", _downloadable),
            ("hardware feasible", _feasible),
            ("soup config valid", _config),
            ("credentials valid", _creds),
            ("quota sufficient", _quota),
        ],
    )


def during_training(*, gpu_detected: bool, gpu_utilization: float, loss: float, oom_events: int, checkpoint_created: bool) -> SanityResult:
    import math

    def _gpu():
        if not gpu_detected:
            raise SanityCheckError("no GPU detected")

    def _util():
        if gpu_utilization <= 0:
            raise SanityCheckError("GPU utilization is 0")

    def _loss():
        if not math.isfinite(loss):
            raise SanityCheckError(f"loss is not finite: {loss}")

    def _oom():
        if oom_events > 1:
            raise SanityCheckError(f"repeated OOM events: {oom_events}")

    def _ckpt():
        if not checkpoint_created:
            raise SanityCheckError("no checkpoint was created")

    return run_checks(
        "during-training",
        [("gpu detected", _gpu), ("gpu util > 0", _util), ("loss finite", _loss),
         ("no repeated oom", _oom), ("checkpoint created", _ckpt)],
    )


def after_training(model_dir: Path | str) -> SanityResult:
    p = Path(model_dir)

    def _exists():
        if not p.exists():
            raise SanityCheckError(f"no checkpoint/model dir at {p}")

    def _has_config():
        if not (p / "config.json").exists() and not any(p.glob("**/config.json")):
            raise SanityCheckError("no config.json found in model dir")

    return run_checks("after-training",
                       [("checkpoint exists", _exists), ("model config exists", _has_config)])


def before_final_selection(candidates_with_results: list) -> SanityResult:
    def _evaluated():
        if not candidates_with_results:
            raise SanityCheckError("no candidates were evaluated")

    def _normalized():
        # Ensure at least one shared metric across candidates so the comparison is fair.
        keys = None
        for r in candidates_with_results:
            k = set(r.metrics.keys()) if hasattr(r, "metrics") and r.metrics else set()
            keys = k if keys is None else (keys & k)
        if not keys:
            raise SanityCheckError("candidates have no metrics in common")

    return run_checks(
        "before-final-selection",
        [("evaluation completed", _evaluated), ("shared normalized metrics", _normalized)],
    )
