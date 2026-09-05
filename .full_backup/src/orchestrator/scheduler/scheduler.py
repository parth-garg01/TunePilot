"""High-level job scheduler that drives backends via the state machine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..backends.base import BackendJobStatus, ComputeBackend, JobSubmission
from ..backends.manager import BackendManager
from ..errors import SchedulerError
from ..jobs.records import EventRecord, JobRecord, JobStatus
from ..jobs.registry import ExperimentRegistry
from ..logging_utils import get_logger
from .queue import JobQueue
from .retry import FailureClassifier, RetryPolicy
from .state_machine import JobStateMachine

log = get_logger(__name__)


_BACKEND_TO_JOB_STATUS = {
    BackendJobStatus.QUEUED: JobStatus.QUEUED,
    BackendJobStatus.STARTING: JobStatus.STARTING,
    BackendJobStatus.RUNNING: JobStatus.RUNNING,
    BackendJobStatus.SUCCEEDED: JobStatus.COMPLETED,
    BackendJobStatus.FAILED: JobStatus.FAILED,
    BackendJobStatus.CANCELLED: JobStatus.CANCELLED,
    BackendJobStatus.UNKNOWN: JobStatus.RUNNING,
}


@dataclass
class ScheduledJob:
    experiment_id: int
    backend_name: str
    submission: JobSubmission
    job_record: JobRecord | None = None
    backend_job_id: str | None = None
    attempt: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class JobScheduler:
    """Coordinates a job queue with backends, retries, and the registry."""

    def __init__(
        self,
        *,
        registry: ExperimentRegistry,
        backends: BackendManager,
        max_concurrent: int = 2,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.registry = registry
        self.backends = backends
        self.max_concurrent = max_concurrent
        self.retry = retry_policy or RetryPolicy()
        self.state = JobStateMachine()
        self.classifier = FailureClassifier()
        self.queue = JobQueue()
        self._active: dict[int, ScheduledJob] = {}

    def enqueue(self, scheduled: ScheduledJob, *, priority: int = 0) -> int:
        record = JobRecord(
            id=None,
            experiment_id=scheduled.experiment_id,
            backend=scheduled.backend_name,
            backend_job_id=None,
            status=JobStatus.QUEUED.value,
            created_at="",
            gpu_type=scheduled.submission.hardware.gpus[0].name if scheduled.submission.hardware.gpus else "",
            gpu_count=scheduled.submission.hardware.gpu_count,
            estimated_hours=scheduled.submission.max_hours or 0.0,
            priority=priority,
        )
        scheduled.job_record = self.registry.create_job(record)
        self.registry.log_event(EventRecord(
            id=None, experiment_id=scheduled.experiment_id, job_id=scheduled.job_record.id,
            kind="scheduler.enqueue", message="job queued",
            payload={"backend": scheduled.backend_name, "priority": priority},
        ))
        return self.queue.push(scheduled, priority=priority)

    def _submit(self, scheduled: ScheduledJob) -> None:
        backend: ComputeBackend = self.backends.get(scheduled.backend_name)
        assert scheduled.job_record and scheduled.job_record.id
        self.registry.set_job_status(scheduled.job_record.id, JobStatus.SUBMITTING)
        try:
            job = backend.submit_job(scheduled.submission)
        except Exception as e:
            self._handle_failure(scheduled, str(e))
            return
        scheduled.backend_job_id = job.id
        self.registry.update_job(scheduled.job_record.id, backend_job_id=job.id,
                                  status=JobStatus.STARTING.value)
        self._active[scheduled.job_record.id] = scheduled

    def dispatch_ready(self) -> int:
        """Move as many queued items as possible into active state."""
        moved = 0
        while len(self._active) < self.max_concurrent:
            item = self.queue.pop_ready()
            if item is None:
                break
            self._submit(item)
            moved += 1
        return moved

    def poll(self) -> None:
        """Poll backends for status updates and finalize completed jobs."""
        for job_id, sched in list(self._active.items()):
            if not sched.backend_job_id:
                continue
            backend = self.backends.get(sched.backend_name)
            try:
                bjob = backend.get_job_status(sched.backend_job_id)
            except Exception as e:
                log.warning("polling %s failed: %s", sched.backend_job_id, e)
                continue
            new = _BACKEND_TO_JOB_STATUS.get(bjob.status, JobStatus.RUNNING)
            assert sched.job_record and sched.job_record.id
            current = JobStatus(self.registry.get_job(sched.job_record.id).status)
            if new != current and self.state.can_transition(current, new):
                self.registry.set_job_status(sched.job_record.id, new, error=bjob.error)
            if new in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
                self._finalize(sched, bjob.status, bjob.error)

    def _finalize(self, sched: ScheduledJob, backend_status: BackendJobStatus, error: str | None) -> None:
        assert sched.job_record and sched.job_record.id
        self._active.pop(sched.job_record.id, None)
        if backend_status == BackendJobStatus.SUCCEEDED:
            self.queue.resolve(sched.job_record.id)
        elif backend_status == BackendJobStatus.FAILED and error:
            self._handle_failure(sched, error)
        elif backend_status == BackendJobStatus.CANCELLED:
            pass

    def _handle_failure(self, sched: ScheduledJob, error_text: str) -> None:
        assert sched.job_record and sched.job_record.id
        failure = self.classifier.classify(error_text)
        sched.attempt += 1
        self.registry.log_event(EventRecord(
            id=None, experiment_id=sched.experiment_id, job_id=sched.job_record.id,
            kind="scheduler.failure", message=error_text[:500],
            payload={"failure": failure.value, "attempt": sched.attempt},
        ))
        if self.retry.should_retry(sched.attempt, failure):
            actions = self.retry.recovery_actions(failure)
            self.registry.set_job_status(sched.job_record.id, JobStatus.RETRYING, error=error_text)
            log.info("retrying job %s in %.1fs (%s)", sched.job_record.id,
                     self.retry.next_delay(sched.attempt), actions)
            time.sleep(0)  # backoff is caller's responsibility
            self.queue.push(sched, priority=sched.job_record.priority)
        else:
            self.registry.set_job_status(sched.job_record.id, JobStatus.FAILED, error=error_text)

    def wait_until_done(self, *, poll_interval: float = 5.0, timeout_seconds: float | None = None) -> None:
        start = time.time()
        while len(self._active) or len(self.queue):
            self.dispatch_ready()
            self.poll()
            if timeout_seconds is not None and time.time() - start > timeout_seconds:
                raise SchedulerError("scheduler wait timed out")
            time.sleep(poll_interval)
