"""Test suite for TunePilot September 5 Feature Update: Hybrid CLI + Interactive Chat UI."""

from __future__ import annotations

from pathlib import Path
import pytest

from orchestrator.terminal.context import ConversationContext
from orchestrator.terminal.attachments import AttachmentManager, Attachment
from orchestrator.terminal.documents import DocumentEngine
from orchestrator.terminal.nlp_router import NLPRouter
from orchestrator.terminal.confirmation import ConfirmationManager
from orchestrator.terminal.session import TerminalChatSession
from orchestrator.terminal.help_system import get_help_panel


def test_conversation_context_pronoun_resolution():
    ctx = ConversationContext(project_name="test-proj")
    ctx.update_discovered_models([
        {"identifier": "Qwen/Qwen2.5-7B", "family": "qwen", "score": 0.95},
        {"identifier": "meta-llama/Llama-3.1-8B", "family": "llama", "score": 0.91},
        {"identifier": "mistralai/Mistral-7B-v0.3", "family": "mistral", "score": 0.88},
    ])

    # Test "the second candidate"
    m2 = ctx.resolve_model_reference("What about the second candidate?")
    assert m2 is not None
    assert m2["identifier"] == "meta-llama/Llama-3.1-8B"

    # Test "the Qwen one"
    mq = ctx.resolve_model_reference("Tell me about the qwen model")
    assert mq is not None
    assert mq["identifier"] == "Qwen/Qwen2.5-7B"

    # Test "train it" (pronoun resolution)
    mit = ctx.resolve_model_reference("Train it right now.")
    assert mit is not None
    assert mit["identifier"] == "Qwen/Qwen2.5-7B"


def test_experiment_reference_resolution():
    ctx = ConversationContext(project_name="test-proj")
    ctx.update_experiments([
        {"id": "exp-001", "name": "exp-qwen", "status": "RUNNING"},
        {"id": "exp-002", "name": "exp-llama", "status": "FAILED"},
    ])

    # Resolve failed experiment
    failed_exp = ctx.resolve_experiment_reference("Resume the failed experiment")
    assert failed_exp is not None
    assert failed_exp["id"] == "exp-002"

    # Resolve specific ID
    exp1 = ctx.resolve_experiment_reference("What is status of exp-001?")
    assert exp1 is not None
    assert exp1["id"] == "exp-001"


def test_attachment_manager(tmp_path: Path):
    mgr = AttachmentManager(vision_capable=False)

    ds_file = tmp_path / "train.jsonl"
    ds_file.write_text('{"text": "sample"}\n', encoding="utf-8")

    doc_file = tmp_path / "notes.md"
    doc_file.write_text("# Project Notes\nUse LoRA rank 16.", encoding="utf-8")

    text_input = f"Analyze {ds_file} and check @{doc_file}"
    paths = mgr.extract_attachment_paths(text_input)
    assert len(paths) >= 2

    att_ds = mgr.inspect(ds_file)
    assert att_ds.category == "dataset"
    assert att_ds.size_bytes > 0
    assert not att_ds.is_image

    card = mgr.format_attachments_card([att_ds])
    assert "train.jsonl" in card
    assert "Attachments" in card


def test_document_engine_chunking_and_retrieval(tmp_path: Path):
    engine = DocumentEngine(chunk_size=100, chunk_overlap=20)
    sample_text = (
        "TunePilot Fine-Tuning Guide.\n\n"
        "Section 1: Hyperparameters.\nUse learning rate 2e-4 with cosine decay.\n\n"
        "Section 2: CUDA Diagnostics.\nOut of memory errors are avoided using 4-bit QLoRA and gradient checkpointing.\n\n"
        "Section 3: Evaluation Metrics.\nPerplexity and exact match are primary metrics."
    )
    chunks = engine.chunk_text("guide.md", sample_text)
    assert len(chunks) >= 2

    engine.indexed_chunks.extend(chunks)
    results = engine.retrieve("How to avoid out of memory CUDA errors?", top_k=1)
    assert len(results) == 1
    assert "QLoRA" in results[0].text or "memory" in results[0].text


def test_nlp_router_intents():
    router = NLPRouter()

    # CLI Command
    p1 = router.parse("dataset analyze ./data/train.jsonl")
    assert p1.intent == "dataset_analyze"
    assert p1.action_type == "cli_command"

    # Natural Language
    p2 = router.parse("Find the best base models for this dataset. I don't care about training time.")
    assert p2.intent == "models_discover"
    assert p2.args.get("quality_first") is True

    # Natural Language Rejection
    p3 = router.parse("Why was this model rejected?")
    assert p3.intent == "explain_rejection"

    # Training with Confirmation
    p4 = router.parse("Train the top 3 candidates on cloud compute.")
    assert p4.intent == "train"
    assert p4.requires_confirmation is True
    assert p4.args.get("candidates_count") == 3

    # Help
    p5 = router.parse("/help")
    assert p5.intent == "help"


def test_confirmation_manager():
    mgr = ConfirmationManager()
    plan = mgr.create_training_plan(["Model A", "Model B", "Model C"], hours=6.0)
    assert plan.estimated_compute_hours == 6.0
    assert len(plan.models) == 3

    # Modify
    handled, mod = mgr.evaluate_response("only train top 1")
    assert handled is True
    assert mod == "modified"
    assert len(plan.models) == 1

    # Confirm
    handled, status = mgr.evaluate_response("yes")
    assert handled is True
    assert status == "confirmed"
    assert mgr.pending_plan is None


def test_terminal_chat_session_e2e(tmp_path: Path):
    session = TerminalChatSession(project_name="test_proj", root=tmp_path, in_memory_history=True)

    # 1. Ask for help
    r_help = session.execute_turn("help")
    assert "help" in r_help.lower()

    # 2. Discover models
    r_models = session.execute_turn("Find the best 7B models")
    assert "discovered" in r_models.lower() or "candidate" in r_models.lower()

    # 3. Ask about rejection
    r_rej = session.execute_turn("Why was it rejected?")
    assert "Analysis" in r_rej or "compatible" in r_rej

    # 4. Initiate training (triggers confirmation)
    r_train = session.execute_turn("Train it")
    assert "confirm" in r_train.lower()
    assert session.confirmation_mgr.pending_plan is not None

    # 5. Confirm training
    r_confirm = session.execute_turn("yes")
    assert "confirmed" in r_confirm.lower()
    assert "submitted" in r_confirm.lower()

    # 6. Status check
    r_status = session.execute_turn("What is the status?")
    assert "experiments" in r_status.lower()

    # 7. Exit
    r_exit = session.execute_turn("exit")
    assert session.is_running is False

    session.core.close()


def test_help_panel_render():
    panel = get_help_panel()
    assert panel is not None
