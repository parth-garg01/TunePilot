"""Kaggle Notebooks backend (PRD sections 15, 16).

Wraps the official Kaggle API through subprocess so the orchestrator does
not hard-depend on the kaggle Python client at import time.

Key rules enforced here:
- one authorized identity per user/project (no account rotation),
- quotas are hard constraints,
- credentials are loaded from env or credentials/kaggle.env only,
- generated notebooks are scanned for secrets before upload.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import field
from pathlib import Path
from typing import Iterator

from ..errors import BackendError, QuotaExceeded
from ..hardware.detect import HardwareProfile, kaggle_p100, kaggle_t4x2
from ..logging_utils import get_logger
from ..security import load_kaggle_credentials
from ..security.notebook_scan import scan_notebook
from .base import (
    BackendCapabilities,
    BackendJob,
    BackendJobStatus,
    BackendQuota,
    ComputeBackend,
    JobSubmission,
)

log = get_logger(__name__)


class KaggleBackend(ComputeBackend):
    name = "kaggle"

    # Kaggle currently documents a weekly GPU quota around 30h.
    DEFAULT_WEEKLY_HOURS = 30.0

    def __init__(
        self,
        *,
        credential_root: Path | str | None = None,
        kaggle_binary: str = "kaggle",
        weekly_quota_hours: float = DEFAULT_WEEKLY_HOURS,
    ) -> None:
        self.kaggle_binary = kaggle_binary
        self.credential_root = credential_root
        self.weekly_quota_hours = weekly_quota_hours
        self._used_hours = 0.0
        self._jobs: dict[str, BackendJob] = {}

    # ---------------- Auth ----------------

    def _load_creds(self) -> dict[str, str]:
        creds = load_kaggle_credentials(self.credential_root)
        return creds.as_env()

    def validate_credentials(self) -> bool:
        try:
            env = self._load_creds()
        except Exception as e:
            log.warning("kaggle credentials unavailable: %s", e)
            return False
        if not shutil.which(self.kaggle_binary):
            log.info("kaggle CLI not installed; credentials are present but CLI is missing")
            return False
        code, _ = self._run([self.kaggle_binary, "config", "view"], env=env)
        return code == 0

    # ---------------- Capabilities / Quota ----------------

    def get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            gpu_options=[kaggle_p100(), kaggle_t4x2()],
            max_concurrent_jobs=2,
            supports_streaming_logs=True,
            supports_checkpoint_resume=True,
            supports_artifact_download=True,
            supports_pricing=False,
        )

    def get_quota(self) -> BackendQuota:
        # Kaggle does not currently expose remaining GPU hours via a stable API.
        # Report the configured weekly ceiling minus locally-tracked usage.
        remaining = max(0.0, self.weekly_quota_hours - self._used_hours)
        return BackendQuota(
            weekly_hours=self.weekly_quota_hours,
            used_hours=self._used_hours,
            remaining_hours=remaining,
            active_jobs=sum(1 for j in self._jobs.values() if j.status in {
                BackendJobStatus.QUEUED, BackendJobStatus.RUNNING, BackendJobStatus.STARTING,
            }),
        )

    def register_usage(self, hours: float) -> None:
        self._used_hours += max(0.0, hours)

    # ---------------- Jobs ----------------

    def _write_kernel_metadata(self, dest: Path, *, submission: JobSubmission) -> Path:
        username = os.environ.get("KAGGLE_USERNAME") or "user"
        slug = re.sub(r"[^a-z0-9-]+", "-", submission.name.lower()).strip("-") or f"job-{uuid.uuid4().hex[:6]}"
        meta = {
            "id": f"{username}/{slug}",
            "title": submission.name,
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": submission.hardware.gpu_count > 0,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
        }
        (dest / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return dest / "kernel-metadata.json"

    def _write_notebook(self, dest: Path, *, submission: JobSubmission) -> Path:
        nb_path = dest / "notebook.ipynb"
        cell = (
            "import subprocess, sys\n"
            "subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'soup-cli'])\n"
            f"subprocess.check_call(['soup', 'train', '--config', {str(submission.soup_config_path)!r}])\n"
        )
        notebook = {
            "cells": [
                {"cell_type": "code", "metadata": {}, "source": [cell], "outputs": [], "execution_count": None}
            ],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        nb_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")

        leaks = scan_notebook(nb_path)
        if leaks:
            raise BackendError(f"refusing to upload notebook with secret indicators: {leaks}")
        return nb_path

    def submit_job(self, submission: JobSubmission) -> BackendJob:
        env = self._load_creds()
        quota = self.get_quota()
        estimated = float(submission.max_hours or 1.0)
        if not quota.has_headroom(estimated):
            raise QuotaExceeded(
                f"Kaggle quota exhausted: needs {estimated:.1f}h, remaining "
                f"{quota.remaining_hours:.1f}h"
            )

        workdir = Path(tempfile.mkdtemp(prefix="kaggle-job-"))
        self._write_kernel_metadata(workdir, submission=submission)
        self._write_notebook(workdir, submission=submission)

        code, out = self._run(
            [self.kaggle_binary, "kernels", "push", "-p", str(workdir)],
            env=env,
        )
        if code != 0:
            raise BackendError(f"kaggle kernels push failed: {out}")

        job_id = f"kaggle-{uuid.uuid4().hex[:8]}"
        job = BackendJob(
            backend=self.name,
            id=job_id,
            status=BackendJobStatus.QUEUED,
            gpu_type=submission.hardware.gpus[0].name if submission.hardware.gpus else "cpu",
            gpu_count=submission.hardware.gpu_count,
            started_at=str(time.time()),
            metadata={"workdir": str(workdir), "kernel_slug": self._slug_from_meta(workdir)},
        )
        self._jobs[job_id] = job
        return job

    def _slug_from_meta(self, workdir: Path) -> str:
        try:
            meta = json.loads((workdir / "kernel-metadata.json").read_text(encoding="utf-8"))
            return meta.get("id", "")
        except Exception:
            return ""

    def get_job_status(self, job_id: str) -> BackendJob:
        job = self._jobs.get(job_id)
        if not job:
            raise BackendError(f"Unknown Kaggle job: {job_id}")
        slug = job.metadata.get("kernel_slug")
        if not slug:
            return job
        env = self._load_creds()
        code, out = self._run([self.kaggle_binary, "kernels", "status", slug], env=env)
        if code != 0:
            job.error = out
            return job
        status = out.strip().lower()
        job.status = self._map_status(status)
        if job.status in {BackendJobStatus.SUCCEEDED, BackendJobStatus.FAILED, BackendJobStatus.CANCELLED}:
            job.completed_at = job.completed_at or str(time.time())
        return job

    def _map_status(self, s: str) -> BackendJobStatus:
        if "running" in s:
            return BackendJobStatus.RUNNING
        if "queued" in s or "pending" in s:
            return BackendJobStatus.QUEUED
        if "complete" in s or "success" in s:
            return BackendJobStatus.SUCCEEDED
        if "error" in s or "fail" in s:
            return BackendJobStatus.FAILED
        if "cancel" in s:
            return BackendJobStatus.CANCELLED
        return BackendJobStatus.UNKNOWN

    def cancel_job(self, job_id: str) -> None:
        # Kaggle does not currently expose a direct cancel API for kernel runs.
        # Mark the local record cancelled and require the user to stop it in the UI.
        job = self._jobs.get(job_id)
        if job:
            job.status = BackendJobStatus.CANCELLED
        log.warning(
            "Kaggle does not expose programmatic cancel for kernels; %s marked cancelled locally",
            job_id,
        )

    def fetch_artifacts(self, job_id: str, dest: Path) -> Path:
        job = self._jobs.get(job_id)
        if not job:
            raise BackendError(f"Unknown Kaggle job: {job_id}")
        slug = job.metadata.get("kernel_slug")
        if not slug:
            raise BackendError("kernel slug missing; cannot download output")
        env = self._load_creds()
        dest.mkdir(parents=True, exist_ok=True)
        code, out = self._run(
            [self.kaggle_binary, "kernels", "output", slug, "-p", str(dest)], env=env,
        )
        if code != 0:
            raise BackendError(f"kaggle kernels output failed: {out}")
        return dest

    def stream_logs(self, job_id: str) -> Iterator[str]:
        # Kaggle streams logs through the notebook output; the artifact fetch above
        # brings them locally when the run terminates.
        return iter([])

    # ---------------- helpers ----------------

    def _run(self, cmd: list[str], *, env: dict[str, str] | None = None) -> tuple[int, str]:
        e = os.environ.copy()
        if env:
            e.update(env)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=180)
            return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
        except FileNotFoundError:
            return 127, f"binary not found: {cmd[0]}"
