from orchestrator.dataset import DatasetAnalyzer, DatasetLoader, TaskDetector
from orchestrator.dataset.fingerprint import dataset_fingerprint


def test_loader_reads_jsonl(sample_jsonl):
    ds = DatasetLoader().load(sample_jsonl)
    assert ds.n_examples == 30


def test_analyzer_produces_stats(sample_jsonl):
    report = DatasetAnalyzer().analyze_path(sample_jsonl)
    assert report.n_examples == 30
    assert report.tokens_est > 0
    assert report.lengths["max"] > 0
    assert "messages" in report.schema


def test_task_detection_chat(sample_jsonl):
    report = DatasetAnalyzer().analyze_path(sample_jsonl)
    detection = TaskDetector().detect(report)
    assert detection.task in {"sft", "chat", "instruction"}
    assert detection.confidence > 0


def test_fingerprint_is_order_independent(sample_jsonl):
    ds = DatasetLoader().load(sample_jsonl)
    a = dataset_fingerprint(ds.all_examples())
    b = dataset_fingerprint(list(reversed(ds.all_examples())))
    assert a == b
