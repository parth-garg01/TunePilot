# Soup Orchestrator

Autonomous LLM Fine-Tuning Orchestrator. A local control plane that inspects
datasets, discovers and ranks candidate base models, generates Soup training
configurations, submits jobs to authorized cloud GPU backends (Kaggle by
default), monitors progress, evaluates candidates, and selects the best model.

The user's laptop stays lightweight. Heavy ML computation runs remotely.

## Status

Base version. Implements the MVP-1 through MVP-3 scope described in `prd.md`.

## Install

```bash
python -m pip install -e .[dev,huggingface,kaggle]
```

Requires Python 3.11+.

## Quick start

```bash
soup-orchestrator init my-project
soup-orchestrator dataset analyze ./data/train.jsonl
soup-orchestrator models discover
soup-orchestrator hardware detect
soup-orchestrator plan
soup-orchestrator train
soup-orchestrator jobs status
soup-orchestrator evaluate
soup-orchestrator compare
soup-orchestrator export
```

Or run the whole pipeline:

```bash
soup-orchestrator run --dataset ./data/my_dataset.jsonl \
  --goal "Create a helpful technical support assistant"
```

## Configuration

See `.env.example` for credentials and `examples/config.yaml` for a full
project configuration.

## Architecture

Local control plane (this package) drives remote training through a
`ComputeBackend` adapter. The initial backend implementations are:

- `LocalSoup4GBBackend` for constrained local training
- `KaggleBackend` for free-tier GPU jobs

See `prd.md` for the full design, safety guardrails, and roadmap.

## Safety

The system will never bypass provider quotas, rotate personal accounts, or
evade Terms of Service. Compute limits are treated as hard constraints.

## Development

```bash
pytest
ruff check src tests
```

## License

Apache-2.0.
