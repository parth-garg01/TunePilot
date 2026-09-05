"""Logging setup with secret redaction.

Uses stdlib logging plus a filter that hides credentials before any log record
leaves the process. All modules should call `get_logger(__name__)` rather than
constructing their own logger.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Iterable

from .security.redaction import redact

_CONFIGURED = False


class RedactingFilter(logging.Filter):
    """Redacts secret-like values from log records."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        try:
            record.msg = redact(str(record.msg))
            if record.args:
                record.args = tuple(redact(str(a)) for a in record.args)
        except Exception:
            pass
        return True


def configure(level: str | int | None = None, *, extra_secrets: Iterable[str] | None = None) -> None:
    """Configure root logging. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = level or os.environ.get("ORCHESTRATOR_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    root = logging.getLogger()
    for h in root.handlers:
        h.addFilter(RedactingFilter())
    if extra_secrets:
        from .security.redaction import register_secrets
        register_secrets(extra_secrets)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure()
    return logging.getLogger(name)
