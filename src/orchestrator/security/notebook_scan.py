"""Scan generated notebooks for secret leaks before uploading."""

from __future__ import annotations

import json
from pathlib import Path

from .redaction import scan_for_secrets


def scan_notebook(path: Path | str) -> list[str]:
    """Return a list of leak indicators found in the notebook at `path`."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    hits = scan_for_secrets(text)
    try:
        nb = json.loads(text)
    except Exception:
        return hits
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        hits.extend(scan_for_secrets(src))
    return hits
