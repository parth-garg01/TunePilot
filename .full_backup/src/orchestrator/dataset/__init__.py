"""Dataset discovery, loading, analysis, and task detection (PRD sections 8, 9)."""

from .loader import DatasetLoader, LoadedDataset, detect_format
from .analyzer import DatasetAnalyzer, DatasetReport
from .task_detector import TaskDetector, TaskDetection
from .fingerprint import dataset_fingerprint
from .quality import QualityChecker, QualityReport

__all__ = [
    "DatasetLoader",
    "LoadedDataset",
    "detect_format",
    "DatasetAnalyzer",
    "DatasetReport",
    "TaskDetector",
    "TaskDetection",
    "dataset_fingerprint",
    "QualityChecker",
    "QualityReport",
]
