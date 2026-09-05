"""Compute backend adapters (PRD section 15)."""

from .base import (
    ComputeBackend,
    BackendCapabilities,
    BackendQuota,
    BackendJob,
    BackendJobStatus,
    JobSubmission,
)
from .kaggle import KaggleBackend
from .local import LocalSoup4GBBackend
from .manager import BackendManager

__all__ = [
    "ComputeBackend",
    "BackendCapabilities",
    "BackendQuota",
    "BackendJob",
    "BackendJobStatus",
    "JobSubmission",
    "KaggleBackend",
    "LocalSoup4GBBackend",
    "BackendManager",
]
