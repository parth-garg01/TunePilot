import pytest

from orchestrator.backends import BackendManager, KaggleBackend, LocalSoup4GBBackend
from orchestrator.backends.base import BackendJob, BackendJobStatus, JobSubmission
from orchestrator.errors import BackendError
from orchestrator.hardware.detect import HardwareProfile, GpuInfo, kaggle_t4x2


def test_backend_manager_registers_and_lookup():
    mgr = BackendManager().register(LocalSoup4GBBackend()).register(KaggleBackend())
    assert "local" in mgr.names()
    assert "kaggle" in mgr.names()
    assert mgr.get("local") is not None
    with pytest.raises(BackendError):
        mgr.get("does-not-exist")


def test_kaggle_capabilities_are_advertised():
    b = KaggleBackend()
    caps = b.get_capabilities()
    assert caps.max_concurrent_jobs >= 1
    assert any(gpu.gpu_count > 0 for gpu in caps.gpu_options)


def test_kaggle_quota_reports_ceiling():
    b = KaggleBackend()
    q = b.get_quota()
    assert q.weekly_hours == KaggleBackend.DEFAULT_WEEKLY_HOURS
    assert q.remaining_hours == q.weekly_hours


def test_kaggle_records_usage():
    b = KaggleBackend()
    b.register_usage(5.0)
    q = b.get_quota()
    assert q.used_hours == 5.0
    assert q.remaining_hours == q.weekly_hours - 5.0
