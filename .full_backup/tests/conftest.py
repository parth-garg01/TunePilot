import json
import sys
from pathlib import Path

# Make `src/` importable in tests without an install step.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "sample.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for i in range(30):
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": f"Question {i}: how do I reset?"},
                    {"role": "assistant", "content": f"Reset by following link {i}."},
                ]
            }) + "\n")
    return p
