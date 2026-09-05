# ⚡ TunePilot (v1.0)
**Autonomous LLM Fine-Tuning Control Plane & Conversational Assistant**

TunePilot is an autonomous LLM fine-tuning orchestrator with a hybrid interactive terminal chatbot UI and scriptable CLI. It inspects datasets, discovers and ranks candidate base models, calculates hardware feasibility, plans experiments, generates training configurations, submits jobs to remote GPU backends (e.g. Kaggle Dual T4s with 30h/week free quota), and selects the winning model checkpoint.

---

## 🚀 First-Time Setup Guide

### 1. Prerequisites
- **Python 3.11+** installed on your system.
- An NVIDIA GPU (optional for local inference/QLoRA; cloud GPUs are supported out-of-the-box).

### 2. Installation
Open your terminal in the project directory:

```bash
# Windows (PowerShell):
py -3.11 -m pip install -e .

# Linux / macOS:
python3.11 -m pip install -e .
```

### 3. Setting Up Kaggle GPU Credentials (Optional, for Cloud Training)
TunePilot supports remote training on **Kaggle's free 30 hours/week Dual T4 GPUs**:
1. Log in to [Kaggle](https://www.kaggle.com/) and go to your **Account Settings** $\to$ **Create New API Token** to download `kaggle.json`.
2. TunePilot automatically reads credentials from either:
   - A `.env` file in the root project directory:
     ```ini
     KAGGLE_USERNAME=your_kaggle_username
     KAGGLE_KEY=your_kaggle_api_key
     ```
   - Standard `~/.kaggle/kaggle.json` (or `./kaggle.json`)
   - `credentials/kaggle.env`

### 4. Verify Setup
Run the diagnostic health check:

```bash
py -3.11 -m orchestrator doctor
```
All core imports, hardware detection, and Kaggle API readiness should report **`OK`**.

---

## 💬 Operating TunePilot: Interactive Chatbot Mode

Launch the interactive terminal pairing assistant:

```bash
# Using the installed CLI shortcut:
tunepilot

# Or directly via Python:
py -3.11 -m orchestrator chat
```

### Conversational Features:
- **Natural Language Instructions:** You don't need to memorize flags. Simply type:
  - `Analyze my dataset at ./examples/sample_dataset.jsonl`
  - `Find the best 7B models for instruction tuning. Quality first.`
  - `What is the status of my hardware?`
- **Pronoun & Context Memory:** Follow up naturally:
  - `Why was that second model rejected?`
  - `What about the Qwen candidate?`
  - `Train it on Kaggle T4x2.`
- **Multi-Modal & Document Attachments:**
  - `@./docs/hyperparameters.md Use these training notes`
  - Drag and drop `.jsonl`, `.parquet`, `.csv`, `.md`, or `.pdf` files directly into your prompt.
- **Interactive Safety Confirmation Cards:** TunePilot calculates compute hours, VRAM usage, and cost estimates, and prompts for your explicit `[Y/n]` approval before launching GPU training.

---

## 💻 Scriptable CLI Command Reference

TunePilot provides traditional deterministic CLI commands for scripts and CI/CD pipelines:

| Task | Command | Description |
|:---|:---|:---|
| **System Diagnostics** | `tunepilot doctor` | Validate Python environment, CUDA, GPU VRAM, and API keys. |
| **Project Initialization** | `tunepilot init <name> --goal "<text>"` | Initialize project layout, database, and starter YAML config. |
| **Dataset Analysis** | `tunepilot dataset analyze <path>` | Profile schema, sequence lengths, tokens, and data anomalies. |
| **Model Discovery** | `tunepilot models discover --task sft` | Search Hugging Face/ModelScope, filter licenses, and rank candidates. |
| **Hardware Detection** | `tunepilot hardware detect` | Inspect local GPUs and view available cloud GPU profiles. |
| **Experiment Planning** | `tunepilot plan --backend kaggle_t4x2` | Generate pilot test runs and full fine-tuning schedules. |
| **Training Execution** | `tunepilot train --model <id>` | Generate configuration and submit training job. |
| **Job Monitoring** | `tunepilot jobs status` | Monitor status of active and queued GPU jobs. |
| **Candidate Comparison** | `tunepilot compare` | Compare validation loss, task scores, and select winner. |
| **Model Export** | `tunepilot export --experiment-id 1` | Merge LoRA adapters and export to FP16 or GGUF format. |
| **End-to-End Pipeline** | `tunepilot run --dataset <path>` | Run one-shot automated pipeline from dataset to evaluation. |

---

## 📱 Live Mobile Monitoring (Optional)

Monitor your training runs and logs on your smartphone over local Wi-Fi or Laptop Hotspot:

```bash
py -3.11 tools/mobile_monitor.py
```
Open **`http://<your-ip>:8765`** in your mobile browser to view live SSE log streams, GPU status cards, and job progress.

---

## 🧪 Running Automated Tests

To run the complete automated test suite (44/44 tests):

```bash
# Set PYTHONPATH for tests
$env:PYTHONPATH = "src;."

# Run pytest
py -3.11 -m pytest tests/
```

---

## 📁 Repository & Architecture Overview

```text
├── src/orchestrator/
│   ├── artifacts/       # Content-hashed storage & project layout
│   ├── backends/        # Compute backends (Local Subprocess, Kaggle Dual T4)
│   ├── cli/             # Scriptable CLI subcommands & entrypoints
│   ├── dataset/         # Multi-format dataset parser, token estimator, leakage detector
│   ├── evaluation/      # Multi-layer evaluation & candidate comparison engine
│   ├── hardware/        # Hardware capability detector & VRAM feasibility estimator
│   ├── jobs/            # SQLite registry, records, and state tracking
│   ├── models/          # Multi-source discovery (HF, ModelScope, Unsloth) & ranking
│   ├── planning/        # Pilot/full experiment planner & compute cost selector
│   ├── reporting/       # Telemetry events and formatted markdown/JSON reports
│   ├── scheduler/       # State machine, priority queue, retry policy, backoff
│   ├── security/        # Credential resolver & secret log redaction
│   ├── soup/            # Soup training configuration generator & adapter
│   └── terminal/        # Hybrid Chatbot REPL, PromptToolkit, context memory, attachments
├── tests/               # 44 Comprehensive unit & integration tests
├── examples/            # Sample dataset and configuration templates
└── tools/               # Mobile live monitoring dashboard
```

---

## 🛡️ License & Contributing

Built for autonomous, reproducible, and cost-effective LLM fine-tuning. Distributed under the MIT License.
