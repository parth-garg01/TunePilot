"""Tests for Metric Post-Processing and Dual Submission Packages."""

import pytest
from pathlib import Path
from orchestrator.evaluation.post_processing import MetricPostProcessor
from orchestrator.evaluation.submission_package import SubmissionPackageGenerator
from orchestrator.terminal.core import TunePilotCore


def test_qwk_threshold_optimization():
    processor = MetricPostProcessor(metric="qwk")
    preds = [1.2, 2.8, 3.1, 4.2, 0.9, 2.1, 3.9, 4.8, 1.1, 2.9]
    targets = [1, 3, 3, 4, 1, 2, 4, 5, 1, 3]
    res = processor.optimize_qwk_thresholds(preds, targets)
    assert res.metric_name == "Quadratic Weighted Kappa (QWK)"
    assert res.optimized_score >= res.baseline_score
    assert len(res.optimal_thresholds) > 0


def test_mcrmse_clipping():
    processor = MetricPostProcessor(metric="mcrmse")
    preds = [0.8, 2.5, 3.2, 5.8, 1.0]
    targets = [1.0, 2.5, 3.0, 5.0, 1.0]
    res = processor.optimize_mcrmse_clipping(preds, targets, min_val=1.0, max_val=5.0)
    assert res.metric_name == "MCRMSE (Multi-Column RMSE)"
    assert res.optimized_score <= res.baseline_score


def test_dual_submission_generator(tmp_path):
    gen = SubmissionPackageGenerator(project_name="test-comp", root=tmp_path)
    pkg = gen.generate_packages(
        single_model="Qwen/Qwen2.5-7B",
        ensemble_models=["Qwen/Qwen2.5-7B", "meta-llama/Llama-3.1-8B"],
        ensemble_weights=[0.6, 0.4],
        optimal_thresholds=[0.5, 1.5, 2.5, 3.5, 4.5],
    )
    assert Path(pkg.single_submission_path).exists()
    assert Path(pkg.ensemble_submission_path).exists()
    assert "submission_1_single_champion.py" in pkg.single_submission_path
    assert "submission_2_ensemble_gold.py" in pkg.ensemble_submission_path


def test_core_post_processing_facade(tmp_path):
    core = TunePilotCore(project_name="test-comp-facade", root=tmp_path)
    pp = core.optimize_thresholds(metric="qwk")
    assert "optimized_score" in pp
    assert pp["optimized_score"] >= pp["baseline_score"]
    
    sub = core.generate_competition_submissions()
    assert Path(sub["single_submission_path"]).exists()
    assert Path(sub["ensemble_submission_path"]).exists()
    core.close()
