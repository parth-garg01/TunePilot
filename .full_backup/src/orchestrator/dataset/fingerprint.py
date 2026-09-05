"""Content fingerprints for datasets and dataset splits."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def dataset_fingerprint(examples: Iterable[dict[str, Any]], *, algorithm: str = "sha256") -> str:
    """Order-independent fingerprint over example contents.

    Each example is serialized deterministically. Individual hex digests are
    sorted before being hashed together so that shuffled datasets fingerprint
    identically.
    """
    h = hashlib.new(algorithm)
    per: list[str] = []
    for ex in examples:
        payload = json.dumps(ex, sort_keys=True, default=str, ensure_ascii=False)
        per.append(hashlib.new(algorithm, payload.encode("utf-8")).hexdigest())
    per.sort()
    for d in per:
        h.update(d.encode("ascii"))
    h.update(str(len(per)).encode("ascii"))
    return h.hexdigest()
