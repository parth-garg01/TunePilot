"""Validate Soup configurations before submission (PRD section 12)."""

from __future__ import annotations

from typing import Any

from ..errors import SoupError


class SoupValidationError(SoupError):
    pass


_VALID_METHODS = {"full", "lora", "qlora"}
_VALID_QUANTS = {"none", "8bit", "4bit"}


def validate_soup_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise SoupValidationError("Soup config must be a mapping")

    for key in ("base", "data", "training", "output"):
        if key not in config:
            raise SoupValidationError(f"Missing required key: {key}")

    training = config["training"]
    if not isinstance(training, dict):
        raise SoupValidationError("`training` must be a mapping")

    method = training.get("method")
    if method not in _VALID_METHODS:
        raise SoupValidationError(f"Invalid training.method: {method!r}")

    quant = training.get("quantization", "none")
    if quant not in _VALID_QUANTS:
        raise SoupValidationError(f"Invalid training.quantization: {quant!r}")

    if training.get("batch_size", 1) < 1:
        raise SoupValidationError("training.batch_size must be >= 1")
    if training.get("gradient_accumulation_steps", 1) < 1:
        raise SoupValidationError("training.gradient_accumulation_steps must be >= 1")
    if training.get("epochs", 1) < 1:
        raise SoupValidationError("training.epochs must be >= 1")
    if training.get("learning_rate", 1e-5) <= 0:
        raise SoupValidationError("training.learning_rate must be > 0")

    data = config["data"]
    if not isinstance(data, dict) or "train" not in data:
        raise SoupValidationError("data.train is required")

    output = config["output"]
    if not isinstance(output, dict) or "dir" not in output:
        raise SoupValidationError("output.dir is required")
