"""Central exception hierarchy for the orchestrator."""

from __future__ import annotations


class OrchestratorError(Exception):
    """Base class for all orchestrator errors."""


class ConfigError(OrchestratorError):
    """Invalid, missing, or malformed configuration."""


class DatasetError(OrchestratorError):
    """Dataset could not be loaded, parsed, or validated."""


class ModelDiscoveryError(OrchestratorError):
    """Model discovery source could not be queried or normalized."""


class FeasibilityError(OrchestratorError):
    """Candidate is not feasible on the requested hardware."""


class SoupError(OrchestratorError):
    """Soup adapter failed to generate, validate, or execute a config."""


class BackendError(OrchestratorError):
    """A compute backend failed to authenticate, submit, or query."""


class QuotaExceeded(BackendError):
    """Provider quota would be exceeded by this action."""


class SchedulerError(OrchestratorError):
    """Job scheduler encountered a state that cannot be resolved."""


class EvaluationError(OrchestratorError):
    """Evaluation failed to run or returned invalid metrics."""


class SanityCheckError(OrchestratorError):
    """A required sanity check did not pass."""
