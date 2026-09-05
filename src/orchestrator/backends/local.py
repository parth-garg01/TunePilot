"""Local Soup 4GB backend (PRD sections 15.1, 15A.1).

Runs Soup training on the local machine using a hardware-constrained strategy.
Verifies actual hardware before training.
"""

from __future__ import annotations

import subprocess
import time
import uuid
from dataclasses import field
from pathlib import Path
from typing import Iterator

from ..errors import BackendError
from ..hardware.detect import HardwareProfile, detect_hardware
from ..logging_utils import get_logger
from .base import (
    BackendCapabilities,
    BackendJob,
    BackendJobStatus,
    BackendQuota,
    ComputeBackend,
    JobSubmission,
)

log = get_logger(__name__)


class LocalSoup4GBBackend(ComputeBackend):
    name = "local"

    def __init__(self, soup_binary: str = "soup") -> None:
        self.soup_binary = soup_binary
        self._jobs: dict[str, BackendJob] = {}
        self._procs: dict[str, subprocess.Popen] = {}

    def validate_credentials(self) -> bool:
        # No credentials required for a local backend.
        return True

    def get_capabilities(self) -> BackendCapabilities:
        hw = detect_hardware()
        return BackendCapabilities(
            name=self.name,
            gpu_options=[hw],
            max_concurrent_jobs=1,
            supports_streaming_logs=True,
            supports_checkpoint_resume=True,
            supports_artifact_download=True,
            supports_pricing=False,
        )

    def get_quota(self) -> BackendQuota:
        return BackendQuota(weekly_hours=None, used_hours=0.0, remaining_hours=None)

    def _verify_hardware(self, requested: HardwareProfile) -> HardwareProfile:
        actual = detect_hardware()
        if requested.gpu_count > 0 and actual.gpu_count == 0:
            raise BackendError("No GPU detected; local backend requires at least one GPU")
        if actual.total_vram_gb + 0.5 < requested.total_vram_gb:
            raise BackendError(
                f"Requested {requested.total_vram_gb:.1f}GB VRAM, actual {actual.total_vram_gb:.1f}GB"
            )
        return actual

    def submit_job(self, submission: JobSubmission) -> BackendJob:
        actual = self._verify_hardware(submission.hardware)
        job_id = f"local-{uuid.uuid4().hex[:8]}"
        cmd = [self.soup_binary, "train", "--config", str(submission.soup_config_path)]
        log.info("starting local job %s: %s", job_id, " ".join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        job = BackendJob(
            backend=self.name,
            id=job_id,
            status=BackendJobStatus.RUNNING,
            gpu_type=actual.gpus[0].name if actual.gpus else "cpu",
            gpu_count=actual.gpu_count,
            started_at=str(time.time()),
            metadata={"pid": proc.pid, "cmd": cmd},
        )
        self._jobs[job_id] = job
        self._procs[job_id] = proc
        return job

    def get_job_status(self, job_id: str) -> BackendJob:
        job = self._jobs.get(job_id)
        if not job:
            raise BackendError(f"Unknown local job: {job_id}")
        proc = self._procs.get(job_id)
        if proc is None:
            return job
        ret = proc.poll()
        if ret is None:
            job.status = BackendJobStatus.RUNNING
        elif ret == 0:
            job.status = BackendJobStatus.SUCCEEDED
            job.completed_at = str(time.time())
        else:
            job.status = BackendJobStatus.FAILED
            job.error = f"exit code {ret}"
            job.completed_at = str(time.time())
        return job

    def cancel_job(self, job_id: str) -> None:
        proc = self._procs.get(job_id)
        if proc and proc.poll() is None:
            proc.terminate()
        if job_id in self._jobs:
            self._jobs[job_id].status = BackendJobStatus.CANCELLED

    def fetch_artifacts(self, job_id: str, dest: Path) -> Path:
        # Local backend writes artifacts directly to the destination; nothing to copy.
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    def stream_logs(self, job_id: str) -> Iterator[str]:
        proc = self._procs.get(job_id)
        if not proc or not proc.stdout:
            return iter([])
        def gen() -> Iterator[str]:
            assert proc.stdout is not None
            for line in proc.stdout:
                yield line.rstrip()
        return gen()
