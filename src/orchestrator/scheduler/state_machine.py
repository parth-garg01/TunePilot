"""Job state machine (PRD section 14)."""

from __future__ import annotations

from ..errors import SchedulerError
from ..jobs.records import JobStatus


_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.QUEUED: {JobStatus.PLANNING, JobStatus.CANCELLED},
    JobStatus.PLANNING: {JobStatus.SUBMITTING, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.SUBMITTING: {JobStatus.STARTING, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.RETRYING},
    JobStatus.STARTING: {JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.RUNNING: {
        JobStatus.CHECKPOINTING, JobStatus.EVALUATING, JobStatus.COMPLETED,
        JobStatus.FAILED, JobStatus.RETRYING, JobStatus.CANCELLED,
        JobStatus.MULTI_GPU_UNDERUTILIZED,
    },
    JobStatus.MULTI_GPU_UNDERUTILIZED: {
        JobStatus.RUNNING, JobStatus.CHECKPOINTING, JobStatus.EVALUATING, JobStatus.COMPLETED,
        JobStatus.FAILED, JobStatus.CANCELLED,
    },
    JobStatus.CHECKPOINTING: {JobStatus.RUNNING, JobStatus.EVALUATING, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.EVALUATING: {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.RETRYING: {JobStatus.QUEUED, JobStatus.SUBMITTING, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: {JobStatus.RETRYING, JobStatus.CANCELLED},
    JobStatus.CANCELLED: set(),
}


class StateTransitionError(SchedulerError):
    pass


class JobStateMachine:
    """Validates a proposed state transition."""

    def can_transition(self, current: JobStatus, target: JobStatus) -> bool:
        return target in _TRANSITIONS.get(current, set())

    def assert_transition(self, current: JobStatus, target: JobStatus) -> None:
        if not self.can_transition(current, target):
            raise StateTransitionError(f"Illegal transition {current.value} -> {target.value}")
