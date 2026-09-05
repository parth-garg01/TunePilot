from pathlib import Path

from orchestrator.jobs.records import (
    CheckpointRecord,
    EvaluationRecord,
    ExperimentRecord,
    JobRecord,
    JobStatus,
)
from orchestrator.jobs.registry import ExperimentRegistry


def test_registry_end_to_end(tmp_path: Path):
    reg = ExperimentRegistry(tmp_path / "orch.sqlite")
    project = reg.ensure_project("demo", "desc")
    exp = reg.create_experiment(ExperimentRecord(
        id=None, project_id=project.id, name="e1",
        model_identifier="Qwen/Qwen2.5-7B", dataset_fingerprint="abc",
        soup_config={"training": {"method": "lora"}},
    ))
    job = reg.create_job(JobRecord(
        id=None, experiment_id=exp.id, backend="kaggle", backend_job_id=None,
        status=JobStatus.QUEUED.value, created_at="",
    ))
    reg.set_job_status(job.id, JobStatus.RUNNING)
    reg.record_checkpoint(CheckpointRecord(id=None, job_id=job.id, step=100, path="x", created_at="", metrics={"loss": 1.2}))
    reg.record_evaluation(EvaluationRecord(id=None, experiment_id=exp.id, layer="task", metric="accuracy", value=0.9, normalized=0.9))
    reg.set_job_status(job.id, JobStatus.COMPLETED)

    jobs = reg.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].status == "COMPLETED"
    evals = reg.evaluations_for(exp.id)
    assert evals and evals[0].metric == "accuracy"
    ck = reg.latest_valid_checkpoint(job.id)
    assert ck and ck.step == 100
    reg.close()
