"""Retry policy and failure classification (PRD section 24)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureClass(str, Enum):
    OOM = "cuda_oom"
    DEPENDENCY = "dependency_error"
    MODEL_DOWNLOAD = "model_download_failure"
    DATASET_CORRUPT = "dataset_corruption"
    QUOTA_EXHAUSTED = "quota_exhausted"
    BACKEND = "backend_failure"
    NETWORK = "network_failure"
    CHECKPOINT_CORRUPT = "checkpoint_corruption"
    INVALID_CONFIG = "invalid_config"
    DIVERGENCE = "training_divergence"
    UNKNOWN = "unknown"


_PATTERNS: list[tuple[FailureClass, re.Pattern]] = [
    (FailureClass.OOM, re.compile(r"cuda\s*out\s*of\s*memory|OOM|out-of-memory", re.I)),
    (FailureClass.DEPENDENCY, re.compile(r"module\s+not\s+found|no module named|dependency|version", re.I)),
    (FailureClass.MODEL_DOWNLOAD, re.compile(r"failed to (download|resolve) model|404 client error", re.I)),
    (FailureClass.DATASET_CORRUPT, re.compile(r"dataset.*(corrupt|invalid|malformed)", re.I)),
    (FailureClass.QUOTA_EXHAUSTED, re.compile(r"quota|rate limit|429", re.I)),
    (FailureClass.NETWORK, re.compile(r"timed? ?out|connection reset|network|dns", re.I)),
    (FailureClass.CHECKPOINT_CORRUPT, re.compile(r"checkpoint.*(corrupt|missing|invalid)", re.I)),
    (FailureClass.INVALID_CONFIG, re.compile(r"invalid|schema|validation", re.I)),
    (FailureClass.DIVERGENCE, re.compile(r"loss (is )?(nan|inf|diverg)", re.I)),
]


class FailureClassifier:
    def classify(self, error_text: str) -> FailureClass:
        for cls, pat in _PATTERNS:
            if pat.search(error_text or ""):
                return cls
        return FailureClass.UNKNOWN


@dataclass
class RetryPolicy:
    max_retries: int = 3
    backoff_seconds: float = 30.0
    exponential: bool = True

    def next_delay(self, attempt: int) -> float:
        if not self.exponential:
            return self.backoff_seconds
        return self.backoff_seconds * (2 ** max(0, attempt - 1))

    def should_retry(self, attempt: int, failure: FailureClass) -> bool:
        if attempt >= self.max_retries:
            return False
        if failure in {FailureClass.QUOTA_EXHAUSTED, FailureClass.INVALID_CONFIG, FailureClass.DATASET_CORRUPT}:
            return False
        return True

    def recovery_actions(self, failure: FailureClass) -> list[str]:
        if failure == FailureClass.OOM:
            return [
                "reduce_micro_batch",
                "enable_gradient_checkpointing",
                "enable_qlora",
                "reduce_sequence_length",
            ]
        if failure == FailureClass.DIVERGENCE:
            return ["reduce_learning_rate", "resume_from_last_checkpoint"]
        if failure == FailureClass.CHECKPOINT_CORRUPT:
            return ["fall_back_to_previous_checkpoint"]
        if failure == FailureClass.NETWORK:
            return ["retry_with_backoff"]
        if failure == FailureClass.MODEL_DOWNLOAD:
            return ["retry_with_backoff", "switch_mirror"]
        if failure == FailureClass.DEPENDENCY:
            return ["reinstall_extras"]
        return []
