"""Content-hashed local artifact storage."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from ..logging_utils import get_logger

log = get_logger(__name__)


def content_hash(path: Path | str, *, algorithm: str = "sha256", chunk: int = 1 << 20) -> str:
    """Return the hex digest of `path` computed with `algorithm`."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def string_hash(text: str, *, algorithm: str = "sha256") -> str:
    return hashlib.new(algorithm, text.encode("utf-8")).hexdigest()


class ArtifactStore:
    """A very small content-addressable store rooted at a directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_file(self, source: Path | str) -> tuple[str, Path]:
        """Copy `source` into the store; return (digest, stored_path)."""
        digest = content_hash(source)
        target = self.root / digest[:2] / digest
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        return digest, target

    def put_json(self, name: str, data: Any) -> Path:
        payload = json.dumps(data, indent=2, sort_keys=True, default=str)
        digest = string_hash(payload)
        target = self.root / "objects" / f"{name}-{digest[:12]}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
        log.debug("stored artifact %s at %s", name, target)
        return target

    def exists(self, digest: str) -> bool:
        return (self.root / digest[:2] / digest).exists()
