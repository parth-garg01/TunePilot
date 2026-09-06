"""Tests for Dynamic Domain-Adaptive Preprocessor."""

import pytest
from pathlib import Path
from orchestrator.dataset.adaptive_preprocessor import DatasetDomain, DynamicDomainPreprocessor
from orchestrator.terminal.core import TunePilotCore


def test_domain_detection():
    preprocessor = DynamicDomainPreprocessor()
    
    assert preprocessor.detect_domain("odir_fundus_retina.jsonl") == DatasetDomain.OPHTHALMIC_MEDICAL
    assert preprocessor.detect_domain("coco_images.jsonl") == DatasetDomain.NATURAL_VISION
    assert preprocessor.detect_domain("financial_churn.csv") == DatasetDomain.TABULAR_STRUCTURED
    assert preprocessor.detect_domain("speech_whisper.wav") == DatasetDomain.AUDIO_SPEECH
    assert preprocessor.detect_domain("general_instruction_qa.jsonl") == DatasetDomain.NLP_TEXT


def test_adaptive_processing_recipes(tmp_path):
    preprocessor = DynamicDomainPreprocessor(output_dir=tmp_path / "preprocessed")
    
    # Test Medical Fundus
    med_rep = preprocessor.process("fundus_dataset.jsonl")
    assert med_rep.detected_domain == DatasetDomain.OPHTHALMIC_MEDICAL
    assert med_rep.features_generated == 628
    assert len(med_rep.preprocessing_steps) == 7
    
    # Test Tabular
    tab_rep = preprocessor.process("customer_churn.csv")
    assert tab_rep.detected_domain == DatasetDomain.TABULAR_STRUCTURED
    assert "missing value imputation" in tab_rep.preprocessing_steps[0]
    
    # Test NLP
    nlp_rep = preprocessor.process("train_instruct.jsonl")
    assert nlp_rep.detected_domain == DatasetDomain.NLP_TEXT
    assert "Unicode normalization" in nlp_rep.preprocessing_steps[0]


def test_core_adaptive_preprocessing_facade(tmp_path):
    core = TunePilotCore(project_name="test-adaptive", root=tmp_path)
    res = core.preprocess_dataset_adaptively("dataset_tabular.csv", hint="sales sensor data")
    assert res["detected_domain"] == "tabular_structured"
    assert len(res["preprocessing_steps"]) > 0
    core.close()
