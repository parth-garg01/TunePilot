"""Secret redaction for logs, generated notebooks, and artifacts.

Registered secrets and a set of common credential patterns are replaced with
`********` before any string is written out.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, Set

_SECRETS: Set[str] = set()

_PATTERNS = [
    re.compile(r"(?i)(kaggle[_-]?key\s*[:=]\s*)([A-Za-z0-9]+)"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)([A-Za-z0-9_\-]+)"),
    re.compile(r"(?i)(hf[_-]?token\s*[:=]\s*)([A-Za-z0-9_\-]+)"),
    re.compile(r"(?i)(bearer\s+)([A-Za-z0-9_\-\.]+)"),
    re.compile(r"(?i)(password\s*[:=]\s*)(\S+)"),
    re.compile(r"(?i)(secret\s*[:=]\s*)(\S+)"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
]

_MASK = "********"


def register_secrets(values: Iterable[str]) -> None:
    """Register additional literal secret values that must be redacted."""
    for v in values:
        if v and isinstance(v, str) and len(v) >= 4:
            _SECRETS.add(v)


def _register_env_secrets() -> None:
    for key in ("KAGGLE_KEY", "HF_TOKEN"):
        v = os.environ.get(key)
        if v:
            _SECRETS.add(v)


_register_env_secrets()


def redact(text: str) -> str:
    """Return `text` with known secrets and credential-shaped tokens masked."""
    if not text:
        return text
    out = text
    for secret in _SECRETS:
        if secret in out:
            out = out.replace(secret, _MASK)
    for pat in _PATTERNS:
        out = pat.sub(lambda m: (m.group(1) + _MASK) if m.lastindex else _MASK, out)
    return out


def scan_for_secrets(text: str) -> list[str]:
    """Return matches suggesting a secret leak. Used before uploading notebooks."""
    hits: list[str] = []
    for secret in _SECRETS:
        if secret in text:
            hits.append("literal-secret")
    for pat in _PATTERNS:
        if pat.search(text):
            hits.append(pat.pattern[:40])
    return hits
