"""Soup CLI adapter (PRD section 12)."""

from .adapter import SoupAdapter, SoupExecutionResult
from .config_generator import SoupConfigGenerator, SoupConfig, SoupTrainingSection
from .validation import validate_soup_config, SoupValidationError

__all__ = [
    "SoupAdapter",
    "SoupExecutionResult",
    "SoupConfigGenerator",
    "SoupConfig",
    "SoupTrainingSection",
    "validate_soup_config",
    "SoupValidationError",
]
