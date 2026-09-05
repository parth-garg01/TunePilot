"""Load datasets from local files or Hugging Face references."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ..errors import DatasetError
from ..logging_utils import get_logger

log = get_logger(__name__)

SUPPORTED_FORMATS = ("json", "jsonl", "csv", "parquet", "hf")


def detect_format(path: Path | str) -> str:
    s = str(path)
    if s.startswith("hf://") or (":" in s and "/" in s and not Path(s).exists()):
        return "hf"
    p = Path(s)
    if p.is_dir():
        return "dir"
    ext = p.suffix.lower().lstrip(".")
    if ext == "jsonl":
        return "jsonl"
    if ext == "json":
        return "json"
    if ext == "csv":
        return "csv"
    if ext in {"parquet", "pq"}:
        return "parquet"
    raise DatasetError(f"Cannot detect dataset format for {path}")


@dataclass
class LoadedDataset:
    """A loaded, streaming-friendly dataset."""

    path: str
    format: str
    n_examples: int = 0
    size_bytes: int = 0
    splits: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def iter_examples(self, split: str = "train") -> Iterator[dict[str, Any]]:
        yield from self.splits.get(split, [])

    def all_examples(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for v in self.splits.values():
            out.extend(v)
        return out


class DatasetLoader:
    """Load JSON/JSONL/CSV/Parquet/HF datasets into a normalized structure."""

    def load(self, path: Path | str, *, format: str = "auto", split: str = "train") -> LoadedDataset:
        fmt = detect_format(path) if format == "auto" else format
        if fmt not in SUPPORTED_FORMATS and fmt != "dir":
            raise DatasetError(f"Unsupported dataset format: {fmt}")

        loader = {
            "jsonl": self._load_jsonl,
            "json": self._load_json,
            "csv": self._load_csv,
            "parquet": self._load_parquet,
            "hf": self._load_hf,
            "dir": self._load_dir,
        }[fmt]
        examples = loader(path)
        size = 0
        try:
            size = Path(path).stat().st_size if Path(path).exists() else 0
        except Exception:
            pass
        ds = LoadedDataset(path=str(path), format=fmt, n_examples=len(examples), size_bytes=size)
        ds.splits[split] = examples
        return ds

    def _load_jsonl(self, path: Path | str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log.warning("Line %d in %s is invalid JSON: %s", i, path, e)
                    out.append({"__malformed__": True, "__line__": i})
        return out

    def _load_json(self, path: Path | str) -> list[dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [d if isinstance(d, dict) else {"value": d} for d in data]
        if isinstance(data, dict):
            for key in ("data", "examples", "records", "train"):
                if key in data and isinstance(data[key], list):
                    return [d if isinstance(d, dict) else {"value": d} for d in data[key]]
            return [data]
        raise DatasetError(f"JSON file at {path} is not a list or dict of examples")

    def _load_csv(self, path: Path | str) -> list[dict[str, Any]]:
        with open(path, "r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]

    def _load_parquet(self, path: Path | str) -> list[dict[str, Any]]:
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as e:
            raise DatasetError("Reading Parquet requires pyarrow. Install with `pip install pyarrow`.") from e
        table = pq.read_table(path)
        return table.to_pylist()

    def _load_hf(self, ref: Path | str) -> list[dict[str, Any]]:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise DatasetError(
                "Loading Hugging Face datasets requires the `datasets` package."
            ) from e
        name = str(ref).replace("hf://", "")
        parts = name.split(":")
        ds_name = parts[0]
        cfg = parts[1] if len(parts) > 1 else None
        split = parts[2] if len(parts) > 2 else "train"
        ds = load_dataset(ds_name, cfg, split=split)
        return [dict(x) for x in ds]

    def _load_dir(self, path: Path | str) -> list[dict[str, Any]]:
        p = Path(path)
        examples: list[dict[str, Any]] = []
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in {".jsonl"}:
                examples.extend(self._load_jsonl(f))
            elif f.suffix.lower() == ".json":
                examples.extend(self._load_json(f))
            elif f.suffix.lower() == ".csv":
                examples.extend(self._load_csv(f))
        return examples
