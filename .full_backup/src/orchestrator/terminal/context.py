"""Conversation Context and Entity Resolution Engine (PRD Section 7, Feature Update)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ChatTurn:
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    intent: Optional[str] = None
    entities: dict[str, Any] = field(default_factory=dict)
    attachments: list[str] = field(default_factory=list)


class ConversationContext:
    """Maintains active session memory and resolves conversational pronouns/references."""

    def __init__(self, project_name: str = "default", project_root: Path | None = None) -> None:
        self.project_name = project_name
        self.project_root = project_root or Path("./projects") / project_name
        self.history: list[ChatTurn] = []
        self.discovered_models: list[dict[str, Any]] = []
        self.active_experiments: list[dict[str, Any]] = []
        self.latest_dataset_summary: dict[str, Any] = {}
        self.rejected_models: dict[str, str] = {}  # model_id -> rejection reason
        self.last_selected_model: Optional[str] = None
        self.last_selected_experiment: Optional[str] = None

    def add_user_message(self, content: str, attachments: list[str] | None = None) -> ChatTurn:
        turn = ChatTurn(
            role="user",
            content=content,
            attachments=attachments or [],
        )
        self.history.append(turn)
        return turn

    def add_assistant_message(
        self,
        content: str,
        intent: Optional[str] = None,
        entities: dict[str, Any] | None = None,
    ) -> ChatTurn:
        turn = ChatTurn(
            role="assistant",
            content=content,
            intent=intent,
            entities=entities or {},
        )
        self.history.append(turn)
        return turn

    def update_discovered_models(self, models: list[dict[str, Any]]) -> None:
        self.discovered_models = models
        if models:
            self.last_selected_model = models[0].get("identifier") or models[0].get("name")

    def update_experiments(self, experiments: list[dict[str, Any]]) -> None:
        self.active_experiments = experiments
        if experiments:
            self.last_selected_experiment = str(experiments[0].get("id") or experiments[0].get("name"))

    def resolve_model_reference(self, text: str) -> Optional[dict[str, Any]]:
        """Resolve references like 'it', 'the second candidate', 'the Qwen one', 'top model'."""
        text_lower = text.lower().strip()

        if not self.discovered_models:
            return None

        # Ordinal matches
        ordinals = {
            "first": 0, "1st": 0, "top": 0, "#1": 0,
            "second": 1, "2nd": 1, "#2": 1,
            "third": 2, "3rd": 2, "#3": 2,
            "fourth": 3, "4th": 3, "#4": 3,
            "fifth": 4, "5th": 4, "#5": 4,
        }
        for ord_word, idx in ordinals.items():
            if re.search(rf"\b{ord_word}\b", text_lower) and idx < len(self.discovered_models):
                m = self.discovered_models[idx]
                self.last_selected_model = m.get("identifier") or m.get("name")
                return m

        # Keyword match in model identifier or family
        for m in self.discovered_models:
            ident = (m.get("identifier") or m.get("name") or "").lower()
            family = (m.get("family") or m.get("architecture") or "").lower()
            # If user mentions specific name (e.g. "qwen", "llama", "mistral", "phi")
            if any(k in text_lower for k in [ident.split("/")[-1], family]) and len(ident) > 0:
                self.last_selected_model = m.get("identifier") or m.get("name")
                return m

        # Pronoun matches: "it", "that model", "this model", "the candidate"
        if re.search(r"\b(it|that model|this model|the model|the candidate)\b", text_lower):
            if self.last_selected_model:
                for m in self.discovered_models:
                    if (m.get("identifier") or m.get("name")) == self.last_selected_model:
                        return m
            return self.discovered_models[0]

        return None

    def resolve_experiment_reference(self, text: str) -> Optional[dict[str, Any]]:
        """Resolve references like 'the failed experiment', 'exp-001', 'it', 'previous experiment'."""
        text_lower = text.lower().strip()

        if not self.active_experiments:
            return None

        # Check explicit exp ID
        for exp in self.active_experiments:
            exp_id = str(exp.get("id") or exp.get("name") or "").lower()
            if exp_id and exp_id in text_lower:
                self.last_selected_experiment = exp_id
                return exp

        # Failed experiment
        if "failed" in text_lower:
            for exp in self.active_experiments:
                if str(exp.get("status", "")).upper() == "FAILED":
                    self.last_selected_experiment = str(exp.get("id"))
                    return exp

        # "it" or "the previous experiment"
        if re.search(r"\b(it|the experiment|previous experiment|that run)\b", text_lower):
            if self.last_selected_experiment:
                for exp in self.active_experiments:
                    if str(exp.get("id") or exp.get("name")) == self.last_selected_experiment:
                        return exp
            return self.active_experiments[0]

        return None

    def get_context_summary(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "turns_count": len(self.history),
            "discovered_models_count": len(self.discovered_models),
            "active_experiments_count": len(self.active_experiments),
            "last_model": self.last_selected_model,
            "last_experiment": self.last_selected_experiment,
        }
