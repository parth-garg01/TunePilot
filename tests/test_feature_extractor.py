"""Tests for Handcrafted Feature Extraction and Multimodal Preprocessing."""

import pytest
import numpy as np
from pathlib import Path
from orchestrator.dataset.feature_extractor import HandcraftedFeatureExtractor
from orchestrator.dataset.multimodal_preprocessor import MultimodalPreprocessor
from orchestrator.terminal.core import TunePilotCore


def test_handcrafted_feature_extractor():
    extractor = HandcraftedFeatureExtractor()
    
    # Test LBP
    lbp = extractor.extract_lbp_histogram()
    assert len(lbp) == 59
    assert pytest.approx(sum(lbp), 0.01) == 1.0
    
    # Test GLCM
    glcm = extractor.extract_glcm_features()
    assert "contrast" in glcm
    assert "homogeneity" in glcm
    assert "energy" in glcm
    
    # Test Color Histograms
    color = extractor.extract_color_histograms()
    assert len(color) == 96
    
    # Test Eye Vector
    eye_vec = extractor.extract_eye_features("left")
    assert eye_vec.total_features_count == 314
    
    # Test Patient Vector (628 Dimensions)
    patient_vec = extractor.extract_patient_biomarker_vector("patient-101")
    assert patient_vec.total_dimensions == 628
    assert len(patient_vec.fused_vector) == 628


def test_multimodal_preprocessor(tmp_path):
    preprocessor = MultimodalPreprocessor(output_dir=tmp_path / "preprocessed")
    report = preprocessor.preprocess_dataset("./data/train.jsonl")
    assert report.features_extracted_per_sample == 628
    assert report.left_eye_features == 314
    assert report.right_eye_features == 314
    assert report.handcrafted_boosted_score > report.baseline_raw_score
    assert Path(report.preprocessed_dataset_path).exists()


def test_core_clinical_preprocessing_facade(tmp_path):
    core = TunePilotCore(project_name="test-clinical", root=tmp_path)
    res = core.preprocess_clinical_dataset("./data/train.jsonl")
    assert res["features_per_sample"] == 628
    assert res["left_eye_features"] == 314
    assert res["right_eye_features"] == 314
    assert res["boosted_score"] > 94.0
    core.close()
