"""Tests for Ensemble, Probability Blending, and Cross Validation."""

import pytest
from orchestrator.evaluation.ensemble import EnsembleBlender, EnsembleCandidate
from orchestrator.planning.cross_validation import CrossValidationPlanner
from orchestrator.terminal.core import TunePilotCore


def test_ensemble_blender_equal_weights():
    blender = EnsembleBlender()
    preds = [
        [0.8, 0.2, 0.9],
        [0.6, 0.4, 0.7],
    ]
    blended = blender.blend_probabilities(preds)
    assert len(blended) == 3
    assert pytest.approx(blended[0], 0.01) == 0.7
    assert pytest.approx(blended[1], 0.01) == 0.3
    assert pytest.approx(blended[2], 0.01) == 0.8


def test_ensemble_blender_weighted():
    blender = EnsembleBlender()
    preds = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    # 3:1 weighting
    blended = blender.blend_probabilities(preds, weights=[3.0, 1.0])
    assert pytest.approx(blended[0], 0.01) == 0.75
    assert pytest.approx(blended[1], 0.01) == 0.25


def test_ensemble_creation_and_boost():
    blender = EnsembleBlender()
    candidates = [
        EnsembleCandidate(model_identifier="Qwen/Qwen2.5-7B", task_score=88.5),
        EnsembleCandidate(model_identifier="meta-llama/Llama-3.1-8B", task_score=86.2),
        EnsembleCandidate(model_identifier="microsoft/deberta-v3-large", task_score=84.0),
    ]
    res = blender.create_ensemble(candidates)
    assert len(res.models) == 3
    assert len(res.weights) == 3
    assert res.ensemble_score > res.single_best_score
    assert res.improvement_pct > 0.0
    assert "generate_ensemble_predictions" in res.submission_script


def test_cross_validation_planner():
    planner = CrossValidationPlanner(n_splits=5, strategy="stratified")
    plan = planner.plan_splits(total_samples=1000)
    assert plan.n_splits == 5
    assert len(plan.splits) == 5
    for s in plan.splits:
        assert s.val_count == 200
        assert s.train_count == 800


def test_core_ensemble_facade(tmp_path):
    core = TunePilotCore(project_name="test-ens", root=tmp_path)
    ens = core.create_ensemble()
    assert "models" in ens
    assert "weights" in ens
    assert ens["ensemble_score"] > ens["single_best"]
    
    cv = core.plan_cross_validation(total_samples=500, n_splits=5)
    assert cv["n_splits"] == 5
    assert len(cv["splits"]) == 5
    core.close()
