from orchestrator.evaluation import (
    AccuracyMetric,
    BLEUMetric,
    CandidateComparison,
    CandidateResult,
    F1Metric,
    normalize_metric,
)


def test_accuracy_metric():
    m = AccuracyMetric()
    assert m.compute(["a", "b"], ["a", "b"]) == 1.0
    assert m.compute(["a", "c"], ["a", "b"]) == 0.5


def test_f1_metric():
    m = F1Metric()
    v = m.compute(["the quick brown fox"], ["the fast brown fox"])
    assert 0 < v < 1


def test_bleu_positive():
    m = BLEUMetric()
    v = m.compute(["hello world"], ["hello world"])
    assert v > 0


def test_normalize_metric_direction():
    lo = normalize_metric(1.0, higher_is_better=False, min_v=1.0, max_v=3.0)
    hi = normalize_metric(3.0, higher_is_better=False, min_v=1.0, max_v=3.0)
    assert lo > hi


def test_comparison_picks_winner():
    rows = [
        CandidateResult("A", val_loss=1.9, task_score=80),
        CandidateResult("B", val_loss=1.7, task_score=85),
        CandidateResult("C", val_loss=1.8, task_score=82),
    ]
    report = CandidateComparison("balanced").compare(rows)
    assert report.winner == "B"
