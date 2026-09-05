"""Post-training sanity checks: model/tokenizer load, inference smoke test."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..errors import SanityCheckError
from ..logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class ModelLoadCheck:
    """Verify model + tokenizer files are present and shape-consistent."""

    def check(self, model_dir: Path | str) -> dict:
        p = Path(model_dir)
        if not p.exists():
            raise SanityCheckError(f"model directory does not exist: {p}")
        required_hints = ["config.json"]
        weight_hints = ["pytorch_model.bin", "model.safetensors", "adapter_model.bin", "adapter_model.safetensors"]
        tokenizer_hints = ["tokenizer.json", "tokenizer.model", "vocab.json", "sentencepiece.bpe.model"]

        found = {f.name for f in p.iterdir() if f.is_file()}
        missing_required = [h for h in required_hints if h not in found]
        has_weights = any(h in found for h in weight_hints)
        has_tokenizer = any(h in found for h in tokenizer_hints)

        if missing_required:
            raise SanityCheckError(f"missing required files: {missing_required}")
        if not has_weights:
            raise SanityCheckError("no weight/adapter files found")
        if not has_tokenizer:
            raise SanityCheckError("no tokenizer files found")
        return {"weights": True, "tokenizer": True, "files": sorted(found)}


@dataclass
class InferenceSmokeTest:
    """A shallow inference test that only relies on the Soup adapter."""

    def check(self, adapter, model_dir: Path | str, prompt: str = "Hello, world.") -> str:
        try:
            return adapter.infer(model_dir, prompt)
        except Exception as e:
            raise SanityCheckError(f"inference smoke test failed: {e}") from e
