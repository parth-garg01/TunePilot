import pytest

from orchestrator.jobs.records import JobStatus
from orchestrator.scheduler import (
    FailureClass,
    FailureClassifier,
    JobQueue,
    JobStateMachine,
    RetryPolicy,
)
from orchestrator.scheduler.state_machine import StateTransitionError


def test_state_machine_allows_valid_transitions():
    sm = JobStateMachine()
    assert sm.can_transition(JobStatus.QUEUED, JobStatus.PLANNING)
    assert sm.can_transition(JobStatus.RUNNING, JobStatus.COMPLETED)


def test_state_machine_rejects_invalid_transitions():
    sm = JobStateMachine()
    with pytest.raises(StateTransitionError):
        sm.assert_transition(JobStatus.COMPLETED, JobStatus.RUNNING)


def test_priority_queue_respects_priority_and_deps():
    q = JobQueue()
    q.push("low", priority=10)
    high = q.push("high", priority=1)
    dep = q.push("dep", priority=0, depends_on={high})
    first = q.pop_ready()
    # dep(prio=0) is highest priority but blocked; high(prio=1) fires first.
    assert first == "high"
    assert q.pop_ready() == "low"  # dep is still blocked
    q.resolve(high)
    assert q.pop_ready() == "dep"


def test_failure_classifier_recognizes_oom():
    fc = FailureClassifier()
    assert fc.classify("torch.cuda.OutOfMemoryError: CUDA out of memory.") == FailureClass.OOM


def test_retry_policy_backoff_and_actions():
    p = RetryPolicy(max_retries=3, backoff_seconds=5, exponential=True)
    assert p.next_delay(1) == 5
    assert p.next_delay(2) == 10
    actions = p.recovery_actions(FailureClass.OOM)
    assert "reduce_micro_batch" in actions
