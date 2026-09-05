"""Dataclass records for the experiment registry (PRD section 34)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    PLANNING = "PLANNING"
    SUBMITTING = "SUBMITTING"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    CHECKPOINTING = "CHECKPOINTING"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    CANCELLED = "CANCELLED"
    MULTI_GPU_UNDERUTILIZED = "MULTI_GPU_UNDERUTILIZED"


TERMINAL_STATUSES = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}


@dataclass
class ProjectRecord:
    id: Optional[int]
    name: str
    description: str
    created_at: str


@dataclass
class DatasetRecord:
    id: Optional[int]
    project_id: int
    path: str
    fingerprint: str
    examples: int
    tokens_est: int
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelRecord:
    id: Optional[int]
    source: str
    identifier: str
    revision: str
    architecture: str
    parameters: int
    context_length: int
    license: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentRecord:
    id: Optional[int]
    project_id: int
    name: str
    model_identifier: str
    dataset_fingerprint: str
    soup_config: dict[str, Any]
    status: str = JobStatus.QUEUED.value
    created_at: str = ""
    updated_at: str = ""


@dataclass
class BackendRecord:
    id: Optional[int]
    name: str
    capabilities: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobRecord:
    id: Optional[int]
    experiment_id: int
    backend: str
    backend_job_id: Optional[str]
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    gpu_type: Optional[str] = None
    gpu_count: int = 0
    estimated_hours: float = 0.0
    actual_hours: float = 0.0
    error: Optional[str] = None
    retries: int = 0
    priority: int = 0


@dataclass
class CheckpointRecord:
    id: Optional[int]
    job_id: int
    step: int
    path: str
    created_at: str
    metrics: dict[str, Any] = field(default_factory=dict)
    valid: bool = True


@dataclass
class EvaluationRecord:
    id: Optional[int]
    experiment_id: int
    layer: str
    metric: str
    value: float
    normalized: float
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


@dataclass
class ArtifactRecord:
    id: Optional[int]
    experiment_id: int
    kind: str
    path: str
    digest: str
    size_bytes: int
    created_at: str


@dataclass
class EventRecord:
    id: Optional[int]
    experiment_id: Optional[int]
    job_id: Optional[int]
    kind: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
