# ⚡ TunePilot (v1.0) — Complete Reference Manual & API Documentation

This document provides a comprehensive reference of all **predefined CLI commands**, **interactive chatbot prompts**, **Python SDK functions**, and **configuration schemas** in TunePilot.

---

## 📑 Table of Contents
1. [Quickstart & Global Execution](#1-quickstart--global-execution)
2. [Interactive Chatbot UI & Natural Language Prompts](#2-interactive-chatbot-ui--natural-language-prompts)
3. [Complete CLI Subcommands Reference](#3-complete-cli-subcommands-reference)
4. [Python SDK & Core Functions API](#4-python-sdk--core-functions-api)
   - [TunePilotCore Facade](#41-tunepilotcore-facade)
   - [Dataset Engine APIs](#42-dataset-engine-apis)
   - [Model Discovery & Ranking APIs](#43-model-discovery--ranking-apis)
   - [Hardware & VRAM Estimation APIs](#44-hardware--vram-estimation-apis)
   - [Scheduler & Backend APIs](#45-scheduler--backend-apis)
   - [SQLite Registry & Metadata APIs](#46-sqlite-registry--metadata-apis)
   - [Evaluation & Candidate Comparison APIs](#47-evaluation--candidate-comparison-apis)
5. [Configuration Schema (`orchestrator.yaml`)](#5-configuration-schema-orchestratoryaml)
6. [Security & Credentials Management](#6-security--credentials-management)

---

## 1. Quickstart & Global Execution

TunePilot can be invoked globally from any directory or terminal:

```powershell
# Open Interactive Chatbot UI
tunepilot

# Run System Diagnostics
tunepilot doctor

# Analyze Dataset
tunepilot dataset analyze ./data/train.jsonl
```

---

## 2. Interactive Chatbot UI & Natural Language Prompts

The interactive terminal chatbot supports free-form natural language, conversational context memory, pronoun resolution, and multi-modal attachments.

### 2.1 Natural Language Intent Guide

| Goal | Example Prompts | Action Performed |
|:---|:---|:---|
| **Analyze Data** | `Analyze ./examples/sample_dataset.jsonl`<br>`Inspect dataset @./data/train.csv` | Scans format, tokens, schema, quality anomalies, and task classification. |
| **Discover Models** | `Find the best 7B models for this task`<br>`Find models under 3B parameters for fast training`<br>`Quality first, discover best models` | Queries Hugging Face / ModelScope / seeds, filters hardware constraints, and displays ranked candidate table. |
| **Pronoun & Context Query** | `Why was that model rejected?`<br>`What about the second candidate?`<br>`Tell me about the Qwen model` | Resolves `"that model"`, `"it"`, or `"second candidate"` from conversation history and explains architecture/VRAM compatibility. |
| **Multi-Modal Attachment** | `@docs/guide.md Use these hyperparameters`<br>`attach ./screenshot.png explain error` | Chunks documents with overlap for context grounding, or inspects image metadata. |
| **Launch Training** | `Train the top 2 candidates on Kaggle T4x2`<br>`Train it right now`<br>`Start fine-tuning` | Formulates a training plan, displays the **Safety Card**, and asks for `[Y/n]` confirmation before submitting. |
| **Monitor Status** | `What is the status of my jobs?`<br>`Show active experiments`<br>`What is the progress?` | Renders the live table with Job IDs, models, statuses, progress percentages, and ETAs. |
| **Retry / Recovery** | `Resume the failed experiment`<br>`Retry job-001` | Resumes execution from the latest verified checkpoint without re-running completed epochs. |
| **Comparison & Winner** | `Compare the models`<br>`Which model is best?`<br>`Show selection report` | Evaluates loss, perplexity, task benchmark, and computes normalized composite score to select winner. |
| **Export Model** | `Export winning model to GGUF`<br>`Merge LoRA adapters` | Merges adapter weights into base model and exports standalone FP16 or GGUF artifacts. |

### 2.2 Built-in Slash Commands in Chat
- `/help` — Show categorized command reference.
- `/clear` — Clear terminal screen and re-render header banner.
- `exit` or `Ctrl+D` — Exit interactive session (background/cloud jobs continue running).

---

## 3. Complete CLI Subcommands Reference

### `tunepilot doctor`
Runs comprehensive system diagnostics on Python libraries, CUDA GPU, VRAM, and Kaggle API credentials.
```powershell
tunepilot doctor
```

### `tunepilot init <name>`
Initializes a new project directory layout, database, and starter configuration.
```powershell
tunepilot init my-assistant --goal "Helpful coding bot" --root ./projects
```
- `--root <path>`: Root artifacts folder (default: `./projects`).
- `--dataset <path>`: Initial dataset path (default: `./data/train.jsonl`).
- `--goal <text>`: Natural language project goal.

### `tunepilot dataset analyze <path>`
Analyzes a local dataset file or URL.
```powershell
tunepilot dataset analyze ./examples/sample_dataset.jsonl --json
```
- `--json`: Output raw JSON report instead of formatted table.

### `tunepilot models discover`
Discovers and ranks base models compatible with your dataset and GPU.
```powershell
tunepilot models discover --task sft --quality-first --limit 10
```
- `--task <name>`: Target task (`sft`, `chat`, `dpo`, `classification`).
- `--quality-first`: Prioritize parameter capacity and expected benchmark score over training speed.
- `--limit <int>`: Max number of candidates to display (default: `10`).

### `tunepilot hardware detect`
Detects local NVIDIA GPU(s), CPU cores, RAM, and lists cloud GPU profiles.
```powershell
tunepilot hardware detect
```

### `tunepilot plan`
Builds pilot and full experiment plans with compute and time estimates.
```powershell
tunepilot plan --backend kaggle_t4x2 --candidates 3
```
- `--backend <name>`: Compute backend (`kaggle_t4x2`, `local_gpu`).
- `--candidates <int>`: Number of candidates to evaluate in pilot phase (default: `3`).

### `tunepilot train`
Generates configuration and submits fine-tuning jobs.
```powershell
tunepilot train --model Qwen/Qwen2.5-7B --backend kaggle_t4x2
```
- `--model <id>`: Model identifier (e.g. `Qwen/Qwen2.5-7B`, `meta-llama/Llama-3.1-8B`).
- `--backend <name>`: Target compute backend.

### `tunepilot jobs status`
Displays the real-time status, progress %, and ETA of all submitted jobs.
```powershell
tunepilot jobs status
```

### `tunepilot experiments list`
Lists all registered experiments and their metadata from the SQLite database.
```powershell
tunepilot experiments list
```

### `tunepilot compare`
Computes normalized multi-factor composite scores across trained checkpoints and selects the winning candidate.
```powershell
tunepilot compare --objective quality_first
```
- `--objective <name>`: Selection objective (`quality_first`, `speed`, `balanced`, `quality_per_dollar`).

### `tunepilot export`
Merges LoRA adapter weights and exports model artifacts for deployment.
```powershell
tunepilot export --experiment-id 1 --format gguf
```
- `--experiment-id <int>`: Experiment ID to export.
- `--format <name>`: Target format (`hf` for Hugging Face FP16, `gguf` for llama.cpp/Ollama, `safetensors`).

### `tunepilot run`
Executes the full automated end-to-end pipeline in one shot.
```powershell
tunepilot run --dataset ./data/train.jsonl --goal "Customer support bot"
```

---

## 4. Python SDK & Core Functions API

You can import and use TunePilot directly inside your own Python scripts and notebooks:

```python
from orchestrator.terminal.core import TunePilotCore

core = TunePilotCore(project_name="my-project")
```

### 4.1 `TunePilotCore` Facade
The central service layer located in `orchestrator.terminal.core`:

```python
class TunePilotCore:
    def __init__(self, project_name: str = "my-llm-project", root: Path = Path("./projects")):
        """Initializes database layout, artifact store, and configuration."""

    def analyze_dataset(self, path: Path | str) -> dict[str, Any]:
        """Loads and analyzes dataset, returning schema, tokens, task, and quality report."""

    def discover_models(
        self,
        task: str = "instruction_tuning",
        quality_first: bool = False,
        size_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Searches multi-source catalogs, filters VRAM compatibility, and returns ranked models."""

    def get_hardware_status(self) -> dict[str, Any]:
        """Detects local GPUs, CPU cores, system RAM, and CUDA availability."""

    def plan_experiments(self, candidates: list[str] | None = None, strategy: str = "pilot_first") -> dict[str, Any]:
        """Generates pilot runs, full training runs, and estimated compute hours."""

    def launch_training(self, models: list[str], backend: str = "kaggle_t4x2") -> list[int]:
        """Creates experiment records and submits training jobs, returning list of job IDs."""

    def get_jobs_status(self) -> list[dict[str, Any]]:
        """Returns live job list with status, progress %, model name, and ETA."""

    def retry_failed_job(self, job_id: int | None = None) -> bool:
        """Resets failed job to QUEUED to resume from last checkpoint."""

    def compare_models(self) -> dict[str, Any]:
        """Computes normalized candidate comparisons and picks the winning checkpoint."""

    def close(self) -> None:
        """Closes SQLite database connections safely."""
```

---

### 4.2 Dataset Engine APIs
Located in `orchestrator.dataset`:

- **`DatasetLoader().load(path: Path | str) -> LoadedDataset`**
  Loads JSONL, CSV, TSV, or Parquet datasets into standardized line records.
- **`DatasetAnalyzer().analyze(path: Path | str) -> dict[str, Any]`**
  Profiles length percentiles ($P_{50}, P_{95}, P_{99}$), estimated tokens, and detected tasks.
- **`detect_format(sample_records: list[dict]) -> str`**
  Classifies schema (`messages[]`, `instruction/response`, `prompt/completion`, `text`).
- **`detect_leakage(train_path, val_path) -> LeakageReport`**
  Checks for exact and near-duplicate leakage between training and validation splits.
- **`fingerprint_dataset(path: Path) -> str`**
  Computes SHA256 content fingerprint for dataset versioning.

---

### 4.3 Model Discovery & Ranking APIs
Located in `orchestrator.models`:

- **`ModelDiscovery().add(source: ModelSource).discover(query: str = "") -> list[ModelCandidate]`**
  Searches `HuggingFaceSource`, `ModelScopeSource`, `UnslothSource`, and `CuratedSeedSource`.
- **`ModelRanker(weights: RankingWeights).rank(candidates, task, hardware) -> list[RankedCandidate]`**
  Computes normalized composite scores $[0, 100]$ based on task fit, context, efficiency, and feasibility.
- **`deduplicate(candidates: list[ModelCandidate]) -> list[ModelCandidate]`**
  Deduplicates identical models across multiple repositories prioritizing authoritative sources.

---

### 4.4 Hardware & VRAM Estimation APIs
Located in `orchestrator.hardware`:

- **`detect_hardware() -> HardwareProfile`**
  Inspects local NVIDIA GPU via NVML, CUDA drivers, CPU cores, and system memory.
- **`estimate_vram_gb(candidate: ModelCandidate, dtype_bytes: float = 2.0) -> float`**
  Estimates VRAM required for full fine-tuning with AdamW optimizer and activations.
- **`feasibility(candidate: ModelCandidate, hw: HardwareProfile) -> FeasibilityDecision`**
  Decides feasible strategy (`full`, `lora`, `qlora`, `sharded`, `infeasible`).

---

### 4.5 Scheduler & Backend APIs
Located in `orchestrator.scheduler` and `orchestrator.backends`:

- **`JobScheduler(registry, backends, max_concurrent=2)`**
  Coordinates priority queue, concurrency limits, state machine, and retry policies.
- **`KaggleBackend(credentials, poll_interval=15.0)`**
  Packages training code into Kaggle kernel, submits to Kaggle API with Dual T4 GPUs, and streams logs.
- **`LocalSubprocessBackend()`**
  Executes training subprocess locally using detected GPU or CPU.

---

### 4.6 SQLite Registry & Metadata APIs
Located in `orchestrator.jobs`:

- **`ExperimentRegistry(db_path: Path)`**
  Thread-safe SQLite database manager for tracking projects, experiments, jobs, checkpoints, and evaluations.
  - `ensure_project(name, description) -> ProjectRecord`
  - `create_experiment(record: ExperimentRecord) -> ExperimentRecord`
  - `create_job(record: JobRecord) -> JobRecord`
  - `set_job_status(job_id, status, error=None)`
  - `list_jobs(experiment_id=None) -> list[JobRecord]`
  - `list_experiments(project_id=None) -> list[ExperimentRecord]`
  - `record_checkpoint(record: CheckpointRecord) -> CheckpointRecord`
  - `record_evaluation(record: EvaluationRecord) -> EvaluationRecord`

---

### 4.7 Evaluation & Candidate Comparison APIs
Located in `orchestrator.evaluation`:

- **`CandidateComparison(objective="balanced").compare(rows: list[CandidateResult]) -> ComparisonReport`**
  Normalizes validation loss, task accuracy, tokens/sec, and VRAM across candidates and picks the winner.

---

## 5. Configuration Schema (`orchestrator.yaml`)

Each project contains an `orchestrator.yaml` configuration file:

```yaml
version: "1.0"
project:
  name: "my-llm-project"
  description: "Autonomous fine-tuning project"

dataset:
  path: "./data/train.jsonl"
  format: "jsonl"
  validation_split: 0.1
  max_seq_length: 2048

training:
  method: "qlora"           # "full" | "lora" | "qlora"
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  learning_rate: 0.0002
  batch_size: 4
  gradient_accumulation_steps: 4
  epochs: 3
  warmup_ratio: 0.03
  optimizer: "paged_adamw_8bit"
  gradient_checkpointing: true

compute:
  backend: "kaggle_t4x2"    # "kaggle_t4x2" | "local_gpu"
  max_gpu_hours: 6.0
  auto_fallback: true

selection:
  objective: "quality_first" # "quality_first" | "speed" | "balanced" | "quality_per_dollar"
```

---

## 6. Security & Credentials Management

TunePilot enforces strict credential isolation and automatic log redaction:
- **Secrets are NEVER stored in `orchestrator.yaml` or Git.**
- **Kaggle API Credentials:** Loaded from `.env`, `~/.kaggle/kaggle.json`, or `credentials/kaggle.env`.
- **Hugging Face Token:** Loaded from `HF_TOKEN` environment variable.
- **Log Masking:** Secret keys and tokens are automatically redacted (`[REDACTED_SECRET]`) in all logs and outputs.

---

*TunePilot v1.0 — Autonomous LLM Fine-Tuning Control Plane & Conversational Assistant*
