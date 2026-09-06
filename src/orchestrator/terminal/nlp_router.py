"""Natural Language Intent Router and Command Parser (PRD Section 11 & 15, Feature Update)."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .context import ConversationContext


@dataclass
class ParsedIntent:
    intent: str
    action_type: str  # "cli_command" | "core_action" | "system" | "chat"
    command_name: Optional[str] = None
    args: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    requires_confirmation: bool = False
    confidence: float = 1.0


class NLPRouter:
    """Parses natural language prompts and traditional CLI commands into structured TunePilot actions."""

    CLI_COMMANDS = {
        "init", "config", "dataset", "models", "hardware", "compute",
        "plan", "train", "jobs", "experiments", "evaluate", "compare",
        "export", "serve", "clean", "doctor", "run", "help", "exit", "quit", "clear",
    }

    def __init__(self, context: Optional[ConversationContext] = None) -> None:
        self.context = context or ConversationContext()

    def parse(self, text: str) -> ParsedIntent:
        trimmed = text.strip()
        if not trimmed:
            return ParsedIntent(intent="empty", action_type="system", raw_text=trimmed)

        # 1. System / Shell commands
        if trimmed in {"exit", "quit", ":q", "/exit"}:
            return ParsedIntent(intent="exit", action_type="system", raw_text=trimmed)

        if trimmed in {"clear", "cls", "/clear", "reset"}:
            return ParsedIntent(intent="clear", action_type="system", raw_text=trimmed)

        if trimmed in {"help", "/help", "?", "commands", "--help", "-h"}:
            return ParsedIntent(intent="help", action_type="system", raw_text=trimmed)

        # 2. Check traditional CLI commands (ensure not a natural language sentence)
        first_token = trimmed.split()[0].lower()
        second_token = trimmed.split()[1].lower() if len(trimmed.split()) > 1 else ""
        is_nl_sentence = second_token in {"the", "it", "top", "candidates", "my", "this", "that", "these", "all", "best", "for", "is", "are", "with"}

        if first_token in self.CLI_COMMANDS and not is_nl_sentence:
            try:
                tokens = shlex.split(trimmed)
                return self._parse_cli_command(tokens, trimmed)
            except Exception:
                pass


        # 3. Natural Language Intent Classification
        return self._parse_natural_language(trimmed)

    def _parse_cli_command(self, tokens: list[str], raw: str) -> ParsedIntent:
        cmd = tokens[0].lower()
        sub = tokens[1].lower() if len(tokens) > 1 else ""

        if cmd == "dataset":
            subcmd = sub or "analyze"
            path = tokens[2] if len(tokens) > 2 else "./data/train.jsonl"
            return ParsedIntent(
                intent="dataset_analyze",
                action_type="cli_command",
                command_name="dataset",
                args={"subcommand": subcmd, "path": path},
                raw_text=raw,
            )

        if cmd == "models":
            subcmd = sub or "discover"
            return ParsedIntent(
                intent="models_discover" if subcmd in {"discover", "search"} else "models_list",
                action_type="cli_command",
                command_name="models",
                args={"subcommand": subcmd},
                raw_text=raw,
            )

        if cmd == "hardware" or cmd == "compute":
            return ParsedIntent(
                intent="hardware_detect",
                action_type="cli_command",
                command_name="hardware",
                args={"subcommand": sub or "detect"},
                raw_text=raw,
            )

        if cmd == "plan":
            return ParsedIntent(
                intent="plan",
                action_type="cli_command",
                command_name="plan",
                args={},
                raw_text=raw,
            )

        if cmd == "train":
            return ParsedIntent(
                intent="train",
                action_type="cli_command",
                command_name="train",
                args={},
                raw_text=raw,
                requires_confirmation=True,
            )

        if cmd in {"jobs", "experiments"}:
            return ParsedIntent(
                intent="jobs_status",
                action_type="cli_command",
                command_name=cmd,
                args={"subcommand": sub or "status"},
                raw_text=raw,
            )

        if cmd == "evaluate":
            return ParsedIntent(
                intent="evaluate",
                action_type="cli_command",
                command_name="evaluate",
                args={},
                raw_text=raw,
            )

        if cmd == "compare":
            return ParsedIntent(
                intent="compare",
                action_type="cli_command",
                command_name="compare",
                args={},
                raw_text=raw,
            )

        if cmd == "export":
            return ParsedIntent(
                intent="export",
                action_type="cli_command",
                command_name="export",
                args={},
                raw_text=raw,
            )

        if cmd == "doctor":
            return ParsedIntent(
                intent="doctor",
                action_type="cli_command",
                command_name="doctor",
                args={},
                raw_text=raw,
            )

        if cmd == "init":
            name = tokens[1] if len(tokens) > 1 else "my-project"
            return ParsedIntent(
                intent="init",
                action_type="cli_command",
                command_name="init",
                args={"name": name},
                raw_text=raw,
            )

        return ParsedIntent(
            intent=f"cli_{cmd}",
            action_type="cli_command",
            command_name=cmd,
            args={"tokens": tokens[1:]},
            raw_text=raw,
        )

    def _parse_natural_language(self, text: str) -> ParsedIntent:
        low = text.lower()

        # 0. Remote Dataset URLs (Kaggle & Hugging Face)
        kaggle_match = re.search(r"https?://(?:www\.)?kaggle\.com/(?:code|datasets|competitions)/([a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+|[a-zA-Z0-9_\-]+)", text, re.IGNORECASE)
        hf_match = re.search(r"https?://(?:www\.)?huggingface\.co/(?:datasets/)?([a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+)", text, re.IGNORECASE)

        if kaggle_match or hf_match:
            slug = kaggle_match.group(1) if kaggle_match else hf_match.group(1)
            source = "kaggle" if kaggle_match else "huggingface"
            has_train_intent = any(k in low for k in ["train", "strt", "start", "find", "model", "fine-tune", "finetune", "run"])
            return ParsedIntent(
                intent="dataset_discover_and_train" if has_train_intent else "dataset_analyze",
                action_type="core_action",
                args={"dataset_slug": slug, "source": source, "url": kaggle_match.group(0) if kaggle_match else hf_match.group(0)},
                raw_text=text,
                requires_confirmation=has_train_intent,
            )

        # Domain-Specific Clinical & Medical Feature Preprocessing (628-D LBP, GLCM, CLAHE, Vessels, Lesions)
        if any(k in low for k in [
            "lbp", "glcm", "vessel", "vessels", "optic disc", "lesion", "handcrafted", "preprocessing",
            "preprocess", "biomarker", "628", "feature extraction", "feature engineering", "xgboost"
        ]):
            path_match = re.search(r"([\w\-\./\\]+\.(?:jsonl|parquet|csv|tsv))", text)
            path = path_match.group(1) if path_match else "./data/train.jsonl"
            return ParsedIntent(
                intent="clinical_preprocessing",
                action_type="core_action",
                args={"path": path},
                raw_text=text,
            )

        # Dataset inspection
        if any(k in low for k in ["analyze my dataset", "analyze the dataset", "analyze this", "inspect dataset", "validate dataset", "check dataset", "this is the dataset", "here is the dataset"]):
            path_match = re.search(r"([\w\-\./\\]+\.(?:jsonl|parquet|csv|tsv))", text)
            path = path_match.group(1) if path_match else "./data/train.jsonl"
            return ParsedIntent(
                intent="dataset_analyze",
                action_type="core_action",
                args={"path": path},
                raw_text=text,
            )


        # Model discovery & ranking
        if any(k in low for k in ["find the best base models", "find the best models", "discover models", "find models", "find model", "rank models", "which model is ranked", "show models", "best 7b", "find 7b", "search models"]):
            size_match = re.search(r"(\d+[bB])", text)
            size_filter = size_match.group(1) if size_match else None
            quality_first = "don't care about training time" in low or "best model possible" in low or "quality" in low
            return ParsedIntent(
                intent="models_discover",
                action_type="core_action",
                args={"size_filter": size_filter, "quality_first": quality_first},
                raw_text=text,
            )

        # Model rejection reasoning
        if any(k in low for k in ["why was this model rejected", "why was it rejected", "why was that model rejected", "rejection reason"]):
            ref_model = self.context.resolve_model_reference(text)
            return ParsedIntent(
                intent="explain_rejection",
                action_type="core_action",
                args={"model": ref_model},
                raw_text=text,
            )

        # Training (with flexible typo handling e.g. "strt trainig")
        if any(k in low for k in ["train the top", "train it", "start training", "strt trainig", "strt training", "train the candidates", "launch training", "fine-tune", "finetune", "start train"]):
            count_match = re.search(r"top\s+(\d+|three|two|1|2|3)", low)
            count = 3
            if count_match:
                c_str = count_match.group(1)
                if c_str in {"3", "three"}: count = 3
                elif c_str in {"2", "two"}: count = 2
                elif c_str in {"1", "one"}: count = 1
                elif c_str.isdigit(): count = int(c_str)


            # Check if reference to single model ("train it")
            ref_model = self.context.resolve_model_reference(text)
            models = [ref_model.get("identifier", "candidate")] if (ref_model and "train it" in low) else []

            backend = "kaggle_t4x2"
            if "local" in low:
                backend = "local_gpu"

            return ParsedIntent(
                intent="train",
                action_type="core_action",
                args={"candidates_count": count, "models": models, "backend": backend},
                raw_text=text,
                requires_confirmation=True,
            )

        # Resume failed experiment
        if any(k in low for k in ["resume the failed", "retry the failed", "resume experiment", "retry experiment", "restart failed"]):
            ref_exp = self.context.resolve_experiment_reference(text)
            return ParsedIntent(
                intent="resume_experiment",
                action_type="core_action",
                args={"experiment": ref_exp},
                raw_text=text,
            )

        # Status & Experiment inspection
        if any(k in low for k in ["show me the current experiments", "show experiments", "what's the status", "what is the status", "check status", "job status", "active jobs"]):
            return ParsedIntent(
                intent="jobs_status",
                action_type="core_action",
                args={},
                raw_text=text,
            )

        # Evaluation, Results & Comparison
        if any(k in low for k in [
            "result", "results", "show me the results", "show results", "get the result",
            "evaluate", "evaluation", "how did the model perform", "how did it perform",
            "metrics", "score", "scores", "performance", "compare", "winner", "selection report"
        ]):
            return ParsedIntent(
                intent="compare",
                action_type="core_action",
                args={},
                raw_text=text,
            )


        # Error / Screenshot Explanation
        if any(k in low for k in ["explain this training error", "explain this error", "what is this error", "diagnose error"]):
            return ParsedIntent(
                intent="explain_error",
                action_type="core_action",
                args={"text": text},
                raw_text=text,
            )

        # Ensemble & Model Soups (Kaggle Winning Strategy)
        if any(k in low for k in ["ensemble", "blend", "model soup", "model blend", "create ensemble", "winning strategy", "stacking"]):
            return ParsedIntent(
                intent="ensemble",
                action_type="core_action",
                args={},
                raw_text=text,
            )

        # Metric Post-Processing & Threshold Optimization
        if any(k in low for k in ["post-processing", "post processing", "threshold", "optimize threshold", "qwk", "mcrmse", "metric optimization"]):
            return ParsedIntent(
                intent="post_processing",
                action_type="core_action",
                args={"metric": "qwk" if "qwk" in low else ("mcrmse" if "mcrmse" in low else "qwk")},
                raw_text=text,
            )

        # Dual Competition Submissions Strategy
        if any(k in low for k in ["submit both", "dual submission", "competition submission", "generate submission", "submission package", "prepare submission"]):
            return ParsedIntent(
                intent="submission_package",
                action_type="core_action",
                args={},
                raw_text=text,
            )

        # Export
        if any(k in low for k in ["export winning model", "export model", "quantize and export", "save model"]):
            return ParsedIntent(
                intent="export",
                action_type="core_action",
                args={},
                raw_text=text,
            )



        # General question or chat
        return ParsedIntent(
            intent="chat_general",
            action_type="chat",
            args={"message": text},
            raw_text=text,
            confidence=0.7,
        )
