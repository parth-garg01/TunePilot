"""Generic compute backend abstraction (PRD section 15)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from ..hardware.detect import HardwareProfile


class BackendJobStatus(str, Enum):
    QUEUED = "QUEUED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


@dataclass
class BackendCapabilities:
    name: str
    gpu_options: list[HardwareProfile] = field(default_factory=list)
    max_concurrent_jobs: int = 1
    supports_streaming_logs: bool = False
    supports_checkpoint_resume: bool = False
    supports_artifact_download: bool = True
    supports_pricing: bool = False


@dataclass
class BackendQuota:
    weekly_hours: float | None = None
    used_hours: float = 0.0
    remaining_hours: float | None = None
    reset_time: str | None = None
    active_jobs: int = 0

    def has_headroom(self, hours: float) -> bool:
        if self.remaining_hours is None:
            return True
        return hours <= self.remaining_hours


@dataclass
class JobSubmission:
    """The payload the orchestrator asks a backend to run."""
    name: str
    soup_config_path: Path
    dataset_path: Path
    hardware: HardwareProfile
    env: dict[str, str] = field(default_factory=dict)
    extra_files: list[Path] = field(default_factory=list)
    output_dir: Path | None = None
    priority: int = 0
    max_hours: float | None = None


@dataclass
class BackendJob:
    backend: str
    id: str
    status: BackendJobStatus = BackendJobStatus.QUEUED
    gpu_type: str = ""
    gpu_count: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ComputeBackend(ABC):
    """Abstract interface (PRD section 15)."""

    name: str = "abstract"

    @abstractmethod
    def validate_credentials(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_capabilities(self) -> BackendCapabilities:
        raise NotImplementedError

    @abstractmethod
    def get_quota(self) -> BackendQuota:
        raise NotImplementedError

    @abstractmethod
    def submit_job(self, submission: JobSubmission) -> BackendJob:
        raise NotImplementedError

    @abstractmethod
    def get_job_status(self, job_id: str) -> BackendJob:
        raise NotImplementedError

    @abstractmethod
    def cancel_job(self, job_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def fetch_artifacts(self, job_id: str, dest: Path) -> Path:
        raise NotImplementedError

    def stream_logs(self, job_id: str) -> Iterator[str]:  # noqa: D401
        """Default streaming: no-op iterator. Backends may override."""
        return iter([])
