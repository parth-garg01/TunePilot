# PRD: Autonomous LLM Fine-Tuning Orchestrator

**Status:** Draft\
**Version:** 1.0\
**Target:** Local Windows control plane + cloud GPU execution\
**Primary training framework:** Soup CLI\
**Primary free compute backend:** Kaggle Notebooks/API\
**Language:** Python 3.11\
**Control machine:** Windows laptop\
**Training machine:** Remote cloud GPU runtime

------------------------------------------------------------------------

## 1. Product Summary

Build an autonomous LLM fine-tuning orchestration system that allows a
user to provide:

-   a dataset
-   an optional task description/goal
-   optional constraints such as model size, license, context length, or
    maximum training time

The system should then:

1.  inspect and validate the dataset
2.  infer the likely ML task and data format
3.  identify suitable candidate base models
4.  filter candidates according to hardware, task, license, context
    length, and compatibility
5.  estimate training feasibility and cost/time
6.  generate a Soup training configuration
7.  submit training jobs to an authorized cloud compute backend
8.  monitor jobs
9.  resume/retry failed jobs safely
10. evaluate trained candidates
11. compare candidates
12. select the best model according to user-defined evaluation criteria
13. optionally merge adapters, quantize, export, and publish/deploy the
    winning model

The local laptop is the orchestration/control plane. Heavy ML
computation should happen remotely.

------------------------------------------------------------------------

# 2. Goals

## Primary Goals

### G1. Dataset-to-model automation

The user should be able to provide a dataset without manually
determining every training parameter.

### G2. Intelligent model selection

The system should discover candidate models and rank them rather than
requiring the user to know which model to use.

Model selection must consider:

-   task compatibility
-   model architecture
-   parameter count
-   context length
-   tokenizer compatibility
-   language/domain
-   license
-   Hugging Face availability
-   hardware feasibility
-   quantization availability
-   expected training time
-   existing evaluation evidence where available

The system must distinguish between: - a model that is theoretically
compatible - a model that can actually be trained on the selected
hardware

### G3. Soup integration

Use Soup as the primary training abstraction wherever its capabilities
are appropriate.

The system should generate and validate Soup configuration files rather
than hard-coding training logic into the orchestrator.

### G4. Cloud-first training

Do not require PyTorch/CUDA/model weights to be installed on the user's
laptop.

The laptop should primarily run:

-   orchestration
-   configuration generation
-   dataset validation/preparation
-   job monitoring
-   experiment tracking
-   results visualization
-   model-selection logic

### G5. Reproducibility

Every experiment must record:

-   base model
-   model revision/commit where available
-   dataset fingerprint
-   dataset version
-   Soup version
-   Python version
-   dependency versions
-   training configuration
-   hardware
-   random seeds
-   training duration
-   checkpoints
-   evaluation results
-   generated artifacts
-   timestamps
-   job/backend identifiers

------------------------------------------------------------------------

# 3. Non-Goals

The system will NOT:

-   bypass cloud-provider quotas
-   bypass authentication, rate limits, or account restrictions
-   automate creation of multiple accounts
-   rotate multiple personal accounts to evade a provider's quota
-   use stolen/shared credentials
-   falsely represent identity
-   evade provider Terms of Service
-   guarantee that a free GPU will always be available
-   assume that two GPUs automatically make one training job twice as
    fast

If the user has multiple legitimately authorized compute accounts or
organizational projects, the scheduler may support them as independent
backend credentials, subject to each provider's terms.

------------------------------------------------------------------------

# 4. Important Kaggle Constraint

Kaggle currently documents a weekly GPU quota of around 30 hours,
sometimes higher depending on demand/resources. Kaggle also recommends
using its API to avoid unnecessary interactive sessions.

The system MUST treat provider quotas as hard resource constraints.

The scheduler MUST NOT implement "account hopping" intended to
circumvent Kaggle limits.

Instead, support:

-   one authorized Kaggle identity
-   multiple authorized cloud backends
-   organization/team compute where explicitly permitted
-   optional paid compute backends
-   legitimate additional quota mechanisms offered by the provider

The architecture should still use a generic `ComputeBackend` abstraction
so other providers can be added later.

------------------------------------------------------------------------

# 5. Parallelism Strategy

Parallel execution is a core feature, but it must be implemented
correctly.

## 5.1 Parallel candidate experiments

If three candidate models need to be tested:

``` text
Candidate A ──> GPU job 1
Candidate B ──> GPU job 2
Candidate C ──> GPU job 3
```

they can run concurrently when sufficient authorized compute is
available.

This reduces wall-clock time for a model-selection experiment.

It does NOT reduce total GPU compute consumed.

Example:

``` text
Sequential:
A = 2h
B = 2h
C = 2h

Wall time ≈ 6h
GPU consumption ≈ 6h

Parallel:
A ── 2h
B ── 2h
C ── 2h

Wall time ≈ 2h
GPU consumption ≈ 6 GPU-hours
```

## 5.2 Multi-GPU training

A single training job may use multiple GPUs when the training framework
and model support distributed training.

Example:

``` text
              Training Job
                   |
          +--------+--------+
          |                 |
       GPU 0             GPU 1
          |                 |
          +--------+--------+
                   |
              Checkpoint
```

The orchestrator must NEVER assume linear scaling.

Expected speedup depends on:

-   model
-   batch size
-   communication overhead
-   gradient accumulation
-   sequence length
-   data loader performance
-   interconnect
-   distributed training implementation

## 5.3 Kaggle 2-GPU environments

If a supported Kaggle runtime exposes multiple GPUs, the system may
select it for distributed training when beneficial.

The training adapter must verify:

``` text
nvidia-smi
torch.cuda.device_count()
```

and confirm that the training process actually uses the available
devices.

If only one GPU is utilized, mark the job as:

`MULTI_GPU_UNDERUTILIZED`

and record a warning.

## 5.4 Parallel hyperparameter experiments

The orchestrator should support:

``` text
Model A
 ├── LR 1e-5
 ├── LR 2e-5
 └── LR 5e-5

Model B
 ├── LR 1e-5
 ├── LR 2e-5
 └── LR 5e-5
```

but should use a configurable experiment budget.

Default:

-   2-3 model candidates
-   1-2 training configurations per model
-   short pilot runs first
-   full training only for promising candidates

This prevents wasting limited free GPU hours.

------------------------------------------------------------------------

# 6. Proposed Architecture

``` text
                         LOCAL CONTROL PLANE
+-------------------------------------------------------------+
|                                                             |
|  CLI / Web UI                                               |
|       |                                                     |
|       v                                                     |
|  Project Manager                                            |
|       |                                                     |
|       +---- Dataset Analyzer                                |
|       |                                                     |
|       +---- Model Discovery & Ranking                       |
|       |                                                     |
|       +---- Hardware/Feasibility Estimator                  |
|       |                                                     |
|       +---- Soup Config Generator                           |
|       |                                                     |
|       +---- Experiment Planner                              |
|       |                                                     |
|       +---- Job Scheduler                                   |
|       |                                                     |
|       +---- Backend Manager                                 |
|       |                                                     |
|       +---- Evaluation Manager                              |
|       |                                                     |
|       +---- Experiment Registry                             |
|                                                             |
+----------------------------+--------------------------------+
                             |
                             | API
                             v
                  +-----------------------+
                  | Compute Backend       |
                  | Adapter               |
                  +-----------+-----------+
                              |
             +----------------+----------------+
             |                                 |
             v                                 v
      Kaggle Notebook                    Other Cloud GPU
          API                              Backends
             |                                 |
             v                                 v
       GPU Training                     GPU Training
             |                                 |
             +----------------+----------------+
                              |
                              v
                       Artifact Storage
                              |
                              v
                        Evaluation
                              |
                              v
                       Model Ranking
                              |
                              v
                         Best Model
```

------------------------------------------------------------------------

# 7. Major Components

## 7.1 CLI

Example:

``` bash
soup-orchestrator init
soup-orchestrator dataset analyze ./data/train.jsonl
soup-orchestrator models discover
soup-orchestrator plan
soup-orchestrator train
soup-orchestrator jobs status
soup-orchestrator evaluate
soup-orchestrator select-best
soup-orchestrator export
```

Recommended commands:

``` text
init
config
dataset
models
hardware
plan
train
jobs
experiments
evaluate
compare
export
serve
clean
doctor
```

------------------------------------------------------------------------

# 8. Dataset Analyzer

The analyzer must inspect the supplied dataset before any expensive GPU
job starts.

## Inputs

Support:

-   JSON
-   JSONL
-   CSV
-   Parquet
-   Hugging Face datasets
-   local directories
-   optional remote dataset references

## Analyze

Calculate:

-   number of examples
-   file size
-   token estimate
-   average input length
-   maximum input length
-   output length
-   distribution of lengths
-   missing fields
-   duplicate examples
-   malformed examples
-   language distribution where practical
-   class distribution for classification
-   conversation structure
-   instruction/response structure
-   preference pairs
-   multimodal fields
-   train/validation/test availability

## Dataset quality checks

Detect:

-   empty examples
-   duplicate records
-   conflicting labels
-   extreme outliers
-   invalid JSON
-   missing required fields
-   train/evaluation leakage
-   excessive duplication
-   suspiciously repetitive samples

The system should produce:

``` text
DATASET REPORT

Examples: 42,381
Estimated tokens: 17.4M
Average sequence length: 411
P95 sequence length: 982
Maximum: 8,421

Detected task:
Instruction tuning

Detected format:
messages[]

Warnings:
- 1.2% duplicate examples
- 0.3% malformed records
- 0.7% unusually long samples
```

------------------------------------------------------------------------

# 9. Automatic Task Detection

The system should classify the dataset into one or more likely tasks:

-   SFT/chat
-   instruction tuning
-   classification
-   preference optimization
-   DPO
-   KTO
-   ORPO
-   reasoning
-   code generation
-   summarization
-   extraction
-   embedding
-   reward modeling
-   pretraining/continued pretraining
-   multimodal training

The system must show confidence:

``` text
Detected task: SFT
Confidence: 94%

Alternative:
DPO: 4%
Continued pretraining: 2%
```

The user must be able to override the detection.

------------------------------------------------------------------------

# 10. Model Discovery Engine

## 10.1 Model Sources

ModelForge must use a multi-source model discovery strategy rather than relying on a single model repository. The discovery engine should distinguish between sources that provide downloadable model weights/checkpoints and services that only expose hosted inference. Only models whose weights are legally downloadable and compatible with the selected training pipeline should be considered for fine-tuning.

### Tier 1: Primary model repositories

-   Hugging Face Hub — primary source for pretrained/base checkpoints, model metadata, revisions, licenses, architectures, tokenizers, quantization variants, and community models.
-   ModelScope — additional model repository and discovery source, particularly useful for models and releases that may not first appear on Hugging Face.
-   GitHub/GitLab release repositories — scan approved official repositories for model releases, checkpoint links, conversion scripts, and official base-model releases.

### Tier 2: Model-lab and optimized model ecosystems

-   Unsloth model ecosystem — discover pretrained/open-weight models and Unsloth-compatible variants/configurations where the underlying weights are available for download.
-   NVIDIA NGC Model Catalog — discover GPU-optimized pretrained models and model artifacts, subject to format, license, and fine-tuning compatibility checks.
-   Official model-provider repositories — e.g. Meta, Google, Mistral AI, Qwen/Alibaba, DeepSeek, Microsoft, NVIDIA, and other organizations that publish downloadable open-weight checkpoints.

### Tier 3: Curated and organization sources

-   Local model registry
-   Organization/private model registry
-   Approved internal model catalog
-   User-specified model repositories or URLs
-   Research/project repositories containing officially released checkpoints

### Hosted model catalogs

Platforms such as Together AI may be used as supplementary discovery/evaluation sources when they expose open-weight models, metadata, provenance, or model identifiers. A hosted inference endpoint alone must NOT be treated as a trainable base model. If the platform does not provide downloadable weights or an authorized path to fine-tune/export the resulting model, it should remain an inference/evaluation source rather than a candidate training source.

### Source priority

The default discovery order should be:

1. official model release
2. Hugging Face Hub
3. ModelScope
4. official/verified model ecosystem such as Unsloth or NVIDIA NGC
5. verified GitHub/GitLab release repository
6. organization/local registry
7. other approved catalogs

The same underlying model may appear in multiple sources. ModelForge must deduplicate candidates using model identity, architecture, revision/hash, repository metadata, and weight provenance rather than treating every repository copy as a separate model.

### Source validation

Every discovered candidate must be checked for:

-   downloadable weights or an explicitly supported fine-tuning mechanism
-   model architecture supported by the training stack
-   tokenizer availability
-   license and redistribution/fine-tuning permissions
-   model revision or commit where available
-   base/pretrained vs instruction-tuned status
-   parameter count
-   context length
-   quantization availability
-   framework compatibility
-   file integrity/checksums where available
-   source reputation/provenance
-   compatibility with Soup or another supported training adapter

Models that are already instruction-tuned, RL-tuned, merged, or otherwise post-trained should be explicitly labeled. The system should prefer a true pretrained/base checkpoint when the user's selected strategy requires starting from an untuned model, while still allowing instruction-tuned checkpoints when the user explicitly permits them.

## 10.2 Candidate filtering

Filter models by:

``` text
Task
Architecture
Parameter count
License
Language
Context length
Quantization
Tokenizer
Hardware
Memory
Model maturity
Availability
```

## 10.3 Hardware-aware ranking

Example:

``` text
User hardware:
2x T4 16GB

Candidate:

Qwen-X 7B
Estimated VRAM: 12GB
Feasible: YES
Score: 91

Model Y 14B
Estimated VRAM: 25GB
Feasible: YES with QLoRA + sharding
Score: 77

Model Z 32B
Estimated VRAM: 52GB
Feasible: NO
Score: 21
```

The ranking system must penalize models that require aggressive memory
tricks.

------------------------------------------------------------------------

## 10.4 Multi-source model discovery

Model discovery must not depend exclusively on one model registry.

Primary and supplementary sources may include:

- Hugging Face Hub
- ModelScope
- Unsloth model ecosystem
- NVIDIA NGC model catalog
- official model-provider repositories
- GitHub/GitLab model releases
- local model registry
- organization/private model registry
- approved model catalogs
- user-specified repositories or model URLs
- supplementary hosted model catalogs where model metadata is available

The source adapters must normalize model metadata into a common internal
schema.

The system must record:

- source
- repository/model identifier
- revision
- architecture
- parameter count
- context length
- tokenizer
- license
- supported task types
- quantization variants
- download size where available
- model-card metadata
- benchmark evidence
- base/instruct/post-trained classification
- availability of downloadable weights
- training/fine-tuning compatibility

A hosted inference endpoint alone must NOT be treated as a trainable base
model source unless downloadable weights or an authorized training/export
mechanism is available.

## 10.5 Base-model and untuned-model detection

When the user requests an untuned or base model, the discovery engine must
prefer pretrained/base checkpoints and avoid blindly selecting instruction-
tuned, RL-tuned, merged, or otherwise post-trained variants.

Classification should use, where available:

- repository/model naming
- model-card metadata
- architecture/configuration
- tags
- training-stage metadata
- official documentation
- source-provider metadata

The system should assign a confidence score and show why a model was
classified as base, instruct, post-trained, merged, or uncertain.

Models with uncertain training status may remain candidates but should be
penalized or flagged when the user explicitly requires an untuned base
model.

## 10.6 Source deduplication

The same underlying model may appear across multiple sources. The
orchestrator should deduplicate candidates using repository identity,
model metadata, revision, architecture, parameter count, and weight
fingerprints where available.

Source reliability and freshness should be recorded rather than allowing
duplicate listings to artificially improve a model's ranking.


# 11. Model Selection Policy

The system should NOT blindly choose the largest model.

Default score:

``` text
Model Score =
  25% task compatibility
+ 20% hardware feasibility
+ 15% expected quality
+ 10% dataset/model fit
+ 10% context compatibility
+ 10% training efficiency
+ 5% ecosystem maturity
+ 5% license suitability
```

Weights must be configurable.

The planner should return:

``` text
Recommended:
Qwen candidate

Alternatives:
Llama candidate
Gemma candidate

Reason:
Best expected quality/compute tradeoff for this dataset.
```

------------------------------------------------------------------------

# 12. Soup Integration

Soup must be isolated behind a `SoupAdapter`.

Responsibilities:

-   generate `soup.yaml`
-   validate configuration
-   install required extras
-   execute training
-   inspect logs
-   locate checkpoints
-   merge adapters
-   export models
-   run inference
-   invoke evaluation where supported

The adapter must record the exact Soup version used.

Example generated configuration:

``` yaml
base: <selected-model>

data:
  train: /workspace/data/train.jsonl
  val_split: 0.1

training:
  method: lora
  quantization: 4bit

  epochs: 2
  learning_rate: 2e-5
  batch_size: 1
  gradient_accumulation_steps: 16

output:
  dir: /workspace/output
```

The exact schema must be generated from the installed Soup version and
validated before submission.

------------------------------------------------------------------------

# 13. Pilot Training

Before committing to full training, the system should run a short pilot on the selected compute strategy. For automatic compute selection, the pilot may validate the selected provider/hardware and refine resource estimates before full training.

Example:

``` text
Pilot:
- 100-500 steps
- small sample of dataset
- representative sequence lengths
- target hardware

Measure:
- samples/sec
- tokens/sec
- VRAM
- loss
- OOM events
- GPU utilization
- estimated full-training time
```

Then estimate:

``` text
Estimated full training:
3h 47m

Expected GPU usage:
~3.8 GPU-hours
```

If the estimate exceeds available compute, the planner should suggest:

-   smaller model
-   lower sequence length
-   QLoRA
-   fewer epochs
-   smaller dataset
-   gradient accumulation changes
-   more efficient backend
-   paid/authorized compute

------------------------------------------------------------------------


# 13A. Adaptive Telemetry-Based Model Re-Ranking

Pilot results must feed back into the model and compute selection process.
The orchestrator must not rely solely on static model metadata or estimated
hardware requirements.

The adaptive loop is:

``` text
Initial candidate ranking
        ↓
Compute strategy selection
        ↓
Pilot training
        ↓
Actual telemetry
        ↓
Compare predicted vs actual performance
        ↓
Update candidate scores
        ↓
Re-rank candidates
        ↓
Promote promising candidates
        ↓
Full training
        ↓
Final evaluation
```

Telemetry should include, where available:

- actual VRAM usage
- tokens/sec
- samples/sec
- GPU utilization
- training loss
- validation loss
- OOM events
- startup/download time
- training duration
- checkpoint behavior
- estimated full-training time
- estimated compute consumption
- provider cost where applicable

The system should compare predicted and observed values and improve its
feasibility and performance estimates for subsequent experiments.

Candidates may be:

- promoted to full training
- retained for additional pilots
- deprioritized
- rejected as infeasible
- rejected as inefficient

The system must preserve the reasons for each decision.

This feedback loop is a core part of the autonomous optimization system.

# 14. Job Scheduler

Jobs have states:

``` text
QUEUED
PLANNING
SUBMITTING
STARTING
RUNNING
CHECKPOINTING
EVALUATING
COMPLETED
FAILED
RETRYING
CANCELLED
```

The scheduler must support:

-   priority
-   dependencies
-   retries
-   cancellation
-   checkpoint recovery
-   backend health
-   resource availability
-   quota awareness
-   concurrency limits

------------------------------------------------------------------------

# 15. Compute Strategy Selection and Backend Abstraction

Before training begins, the user must be able to choose how compute will be provided. The orchestrator must support three modes:

### 15.1 Soup Local 4 GB GPU Strategy

Use Soup's hardware-constrained training strategy for approximately 4 GB of GPU VRAM. The planner should optimize compatible settings such as model size, 4-bit quantization, LoRA/QLoRA, sequence length, batch size, gradient accumulation, and gradient checkpointing. The system must verify the actual local hardware before training.

### 15.2 Cloud Compute Provider

Use an API-integrated cloud GPU provider selected and authorized by the user. The provider layer should expose GPU model/VRAM, GPU count, availability, estimated cost where available, quota/credits, concurrency limits, job submission/status, logs, and artifact transfer. The orchestrator should generate the remote environment and execute Soup there.

### 15.3 Automatic Compute Selection

The user may allow the orchestrator to select the compute strategy automatically. It should compare available options using model VRAM requirements, expected training time, GPU availability, estimated cost, remaining quota/credits, provider limits, reliability, training efficiency, and user-defined budget/time constraints. The selected strategy and reasoning must be shown before expensive training unless unattended execution is explicitly enabled.

Create:

``` python
class ComputeBackend:
    name: str

    def validate_credentials():
        ...

    def get_capabilities():
        ...

    def get_quota():
        ...

    def submit_job():
        ...

    def get_job_status():
        ...

    def cancel_job():
        ...

    def fetch_artifacts():
        ...

    def stream_logs():
        ...
```

Initial implementations:

``` text
LocalSoup4GBBackend
KaggleBackend
```

Future:

``` text
ColabBackend
LightningBackend
RunPodBackend
LambdaBackend
VastBackend
ModalBackend
LocalBackend
```

Only providers with supported APIs and terms should be enabled.

------------------------------------------------------------------------

# 15.4 Compute Selection Flow

``` text
Dataset + Goal
      ↓
Model Candidates
      ↓
Training Requirements
      ↓
Compute Strategy
      │
      ├── Soup Local ~4 GB GPU
      ├── Cloud GPU Provider via API
      └── Automatic Selection
             ↓
       Feasibility + Cost + Time + Quota
             ↓
          Pilot Training
             ↓
          Full Training
```

The compute layer must remain independent of model-selection and training logic so additional providers can be added without changing the core orchestrator.

------------------------------------------------------------------------


# 15A. Compute Strategy Selection

Before training, the user must be able to select how compute will be
provided.

## Strategy 1: Soup Local 4 GB GPU Strategy

Use a hardware-constrained strategy optimized for approximately 4 GB of
GPU memory.

The planner may automatically adjust:

- model size
- quantization
- LoRA/QLoRA
- sequence length
- batch size
- gradient accumulation
- checkpointing
- training duration

The objective is to make the selected model feasible on the local machine
without requiring the full training stack to be installed unless needed.

## Strategy 2: API-Integrated Cloud GPU Provider

The user may select an authorized cloud GPU provider accessible through a
supported API.

The provider adapter must expose, where supported:

- available GPU types
- VRAM
- GPU count
- pricing
- availability
- quota/credits
- job submission
- job status
- logs
- cancellation
- artifacts
- checkpoint recovery

The architecture must remain provider-agnostic.

## Strategy 3: Automatic Compute Selection

The orchestrator may automatically choose between local and available
cloud compute based on:

- model requirements
- GPU memory
- GPU count
- expected training time
- expected model quality
- cost
- remaining quota/credits
- provider availability
- reliability
- concurrency
- checkpoint/recovery support

The decision and reasoning must be shown to the user.

Example:

``` text
Compute options

Local 4 GB:
  Feasible: YES
  Estimated time: 11h
  Estimated cost: $0
  Expected quality: 76

Cloud T4:
  Feasible: YES
  Estimated time: 5h
  Estimated cost: $X
  Expected quality: 82

Cloud A100:
  Feasible: YES
  Estimated time: 1.8h
  Estimated cost: $Y
  Expected quality: 86

Recommendation:
Cloud A100

Reason:
Higher-quality candidate is feasible and the user has selected
quality-first optimization.
```

A compute provider must never be selected merely because it is faster if
a materially better model is feasible and the user's objective is
quality-first.

## Compute preference policy

The default optimization objective is:

``` text
quality_first
```

Training time is NOT a hard constraint unless explicitly configured by the
user.

The orchestrator may therefore:

- select a larger model
- use more powerful GPUs
- run longer training
- run additional pilot experiments
- test additional candidate models
- perform more hyperparameter experiments
- perform more extensive evaluation

when the expected quality improvement justifies the additional compute.

Users may explicitly configure constraints such as:

``` yaml
optimization:
  objective: quality_first

constraints:
  max_training_hours: null
  max_cost: null
  max_model_size: null
```

If a constraint is provided, it becomes a hard or configurable planning
constraint according to the user's setting.


# 16. Kaggle Backend

## Authentication

Use environment variables or a secure secret store.

Example:

``` text
KAGGLE_USERNAME
KAGGLE_KEY
```

Never commit credentials to Git.

Do not store raw API keys in `config.yaml`.

Use:

``` text
credentials/
    kaggle.env
```

with strict local permissions, or OS credential storage.

## Quota tracking

Track:

``` text
weekly_quota
used_hours
remaining_hours
reset_time
active_jobs
```

The system should query the current provider state where possible rather
than assuming a fixed 30-hour quota.

## Account policy

The system must support one authorized Kaggle identity per user/project
by default.

If multiple legitimate identities are available under an organization,
they must be represented as separate authorized backend profiles and
must comply with Kaggle's terms.

Do not implement quota evasion through multiple personal accounts.

------------------------------------------------------------------------

# 17. Checkpointing

Every long-running job should checkpoint frequently enough to tolerate
interruption.

Store:

``` text
checkpoint/
    step-100
    step-200
    step-300
```

Also save:

``` text
training_state
optimizer_state
scheduler_state
random_state
config
dataset fingerprint
```

The orchestrator must detect the latest valid checkpoint and offer
resume.

------------------------------------------------------------------------

# 18. Artifact Storage

Local structure:

``` text
projects/
  project-name/
    data/
    configs/
    experiments/
    jobs/
    checkpoints/
    models/
    evaluations/
    logs/
```

Remote structure:

``` text
artifacts/
  project/
    experiment/
      config/
      checkpoints/
      logs/
      model/
      evaluation/
```

Use content hashes to prevent accidental duplication.

------------------------------------------------------------------------

# 19. Evaluation

Evaluation must happen automatically after successful training.

Evaluation layers:

### Layer 1: Training metrics

-   train loss
-   validation loss
-   perplexity where appropriate
-   learning curves

### Layer 2: Dataset-specific metrics

Examples:

-   accuracy
-   F1
-   BLEU
-   ROUGE
-   exact match
-   pass@k
-   task-specific metrics

### Layer 3: LLM judge

Optional.

Use a separately configured evaluator.

Avoid evaluating a model solely with itself.

### Layer 4: Human evaluation

Optional manual review interface.

------------------------------------------------------------------------

# 20. Candidate Comparison

Example report:

``` text
MODEL COMPARISON

                Model A     Model B     Model C
------------------------------------------------
Validation      1.82        1.76        1.91
Task score      82.4        86.1        80.3
Tokens/sec      41          29          52
VRAM            12GB        15GB        8GB
Train time      2.4h        4.1h        1.8h

Overall          81          87          78
```

The system should recommend a winner while showing the reasoning.

------------------------------------------------------------------------

# 21. Automatic Best-Model Selection

The user can configure:

``` yaml
selection:
  objective: quality
  max_training_hours: 8
  max_model_size: 14B
```

Possible objectives:

``` text
quality_first
quality
quality_per_dollar
quality_per_gpu_hour
speed
memory_efficiency
balanced
```

The system must never silently select a model. It must produce a
selection report.

------------------------------------------------------------------------

# 22. Parallel Experiment Planner

The planner should determine whether parallelism is useful.

Example:

``` text
Candidate A pilot: 30 min
Candidate B pilot: 35 min
Candidate C pilot: 25 min

Sequential: ~90 min
Parallel:   ~35 min
```

If resources are insufficient:

``` text
Run A
then B
then C
```

If resources are sufficient:

``` text
A ──────────────
B ──────────────
C ──────────────
```

The scheduler must respect:

-   provider concurrency limits
-   GPU availability
-   project budget
-   quota
-   memory
-   dependencies

------------------------------------------------------------------------

# 23. Training Speed Optimization

The system should consider:

-   mixed precision
-   bf16/fp16
-   gradient checkpointing
-   Flash Attention where compatible
-   4-bit quantization
-   LoRA/QLoRA
-   sequence packing
-   dataloader workers
-   gradient accumulation
-   distributed data parallelism
-   DeepSpeed where supported
-   efficient attention
-   model-specific optimizations

The optimizer must never enable an optimization without checking
compatibility.

------------------------------------------------------------------------

# 24. Failure Recovery

Common failures:

``` text
CUDA OOM
Dependency error
Model download failure
Dataset corruption
Kaggle quota exhausted
Kaggle job failure
Network failure
Checkpoint corruption
Invalid Soup configuration
Training divergence
```

Example automatic response:

``` text
CUDA OOM
   ↓
Reduce micro-batch
   ↓
Enable gradient checkpointing
   ↓
Enable QLoRA
   ↓
Retry pilot
```

Maximum retries must be configurable.

------------------------------------------------------------------------

# 25. Safety and Cost Guardrails

Before any expensive job:

``` text
Estimated compute:
4.2 GPU-hours

Available:
6.7 GPU-hours

Proceed? YES
```

If insufficient:

``` text
Estimated:
11.4 GPU-hours

Available:
6.7 GPU-hours

ACTION:
Do not submit full training.
Run reduced pilot instead.
```

The system should require explicit approval for:

-   paid compute
-   large-scale experiments
-   long jobs
-   multiple candidate training jobs
-   destructive artifact deletion

------------------------------------------------------------------------

# 26. Security

Credentials must:

-   never appear in logs
-   never appear in Git
-   never appear in experiment artifacts
-   never appear in generated notebooks
-   be loaded through environment variables/secure storage

Add secret redaction:

``` text
KAGGLE_KEY=********
```

All generated notebooks must be scanned before upload for accidental
secrets.

------------------------------------------------------------------------

# 27. Configuration

Example:

``` yaml
project:
  name: my-llm-project

dataset:
  path: ./data/train.jsonl

goal:
  type: chat
  description: "Create a domain-specific assistant"

model_selection:
  automatic: true
  candidates: 3
  max_parameters: 14B

training:
  framework: soup
  method: auto
  quantization: auto
  pilot: true

compute:
  preferred:
    - kaggle

  max_gpu_hours: 10

parallelism:
  enabled: true
  max_concurrent_jobs: 2

selection:
  objective: balanced

artifacts:
  backend: local
```

------------------------------------------------------------------------

# 28. Suggested Project Structure

``` text
llm-orchestrator/
│
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
│
├── src/
│   └── orchestrator/
│       ├── cli/
│       ├── config/
│       ├── dataset/
│       ├── models/
│       ├── hardware/
│       ├── planning/
│       ├── soup/
│       ├── backends/
│       │   ├── base.py
│       │   └── kaggle.py
│       ├── scheduler/
│       ├── jobs/
│       ├── evaluation/
│       ├── artifacts/
│       ├── security/
│       └── reporting/
│
├── templates/
│   ├── kaggle/
│   └── soup/
│
├── tests/
│
├── examples/
│
└── projects/
```

------------------------------------------------------------------------

# 29. Technology Stack

## Core

-   Python 3.11
-   Typer or Click
-   Pydantic
-   PyYAML
-   SQLite
-   Rich
-   HTTPX
-   asyncio

## ML/Data

-   Soup CLI
-   Hugging Face Hub
-   datasets
-   transformers
-   tokenizers

Training dependencies should remain remote where possible.

## Cloud

-   Kaggle API/CLI
-   provider-specific adapters

## Storage

-   SQLite for metadata
-   filesystem for local artifacts
-   optional object storage later

## Evaluation

-   configurable benchmark framework
-   task-specific evaluators
-   optional LLM judge

------------------------------------------------------------------------

# 30. MVP

The first release should implement only:

### MVP-1

``` text
Dataset
  ↓
Analyze
  ↓
Select candidate models
  ↓
Generate Soup config
  ↓
Submit Kaggle job
  ↓
Monitor
  ↓
Download model
```

### MVP-2

Add:

-   pilot training
-   automatic feasibility estimation
-   checkpointing
-   evaluation
-   candidate comparison

### MVP-3

Add:

-   parallel candidate jobs
-   automatic best-model selection
-   automatic retry
-   model export
-   richer reports

### MVP-4

Add:

-   additional cloud providers
-   web dashboard
-   experiment tracking
-   advanced optimization

------------------------------------------------------------------------

# 31. Example End-to-End User Experience

The ideal command should eventually be:

``` bash
soup-orchestrator run \
  --dataset ./data/my_dataset.jsonl \
  --goal "Create a helpful technical support assistant"
```

The system responds:

``` text
Analyzing dataset...

Examples: 84,210
Estimated tokens: 31.2M

Detected task:
Instruction tuning
Confidence: 96%

Searching compatible models...

1. Qwen candidate       Score: 91
2. Llama candidate      Score: 87
3. Gemma candidate      Score: 84

Checking Kaggle hardware...

Candidate 1: FEASIBLE
Candidate 2: FEASIBLE
Candidate 3: FEASIBLE

Planning pilot experiments...

Submitting 3 pilot jobs.

Job A: RUNNING
Job B: RUNNING
Job C: RUNNING

Estimated wall time: 42 minutes
Estimated GPU usage: 2.1 hours

Waiting...
```

After pilots:

``` text
Pilot results:

Qwen:
Quality: 84
Speed: 42 tok/s
VRAM: 13.1 GB

Llama:
Quality: 86
Speed: 31 tok/s
VRAM: 15.4 GB

Gemma:
Quality: 81
Speed: 49 tok/s
VRAM: 10.2 GB

Recommended:
Llama

Reason:
Highest expected task quality within the available compute budget.

Start full training? [Y/n]
```

Then:

``` text
Full training started.

Checkpoint: 25%
Checkpoint: 50%
Checkpoint: 75%
Checkpoint: 100%

Evaluation complete.

FINAL MODEL:
Llama candidate + LoRA

Task score: 88.7
Training time: 3h 41m
```

------------------------------------------------------------------------

# 32. Sanity Checks

Every stage must have a sanity check.

## Before dataset processing

-   file exists
-   readable
-   supported format

## After dataset analysis

-   examples \> 0
-   required fields detected
-   no severe corruption

## Before model selection

-   task detected
-   candidate list non-empty

## Before training

-   model downloadable
-   hardware feasible
-   Soup config valid
-   credentials valid
-   compute quota sufficient

## During training

-   GPU detected
-   GPU utilization \> 0 where expected
-   loss is finite
-   no repeated OOM
-   checkpoint created

## After training

-   checkpoint exists
-   model loads
-   tokenizer loads
-   inference works

## Before final selection

-   evaluation completed
-   metrics valid
-   no candidate is being compared using incompatible metrics without
    normalization

------------------------------------------------------------------------

# 33. Observability

The CLI should show:

``` text
JOB  MODEL       STATUS     GPU   TIME    PROGRESS
001  Model A     RUNNING    T4    18m     42%
002  Model B     RUNNING    T4    16m     38%
003  Model C     COMPLETE   T4    21m     100%
```

Logs should be streamed where the backend supports it.

Persist all logs locally after completion.

------------------------------------------------------------------------

# 34. Database Schema

Minimum tables:

``` text
projects
datasets
models
experiments
jobs
backends
checkpoints
evaluations
artifacts
events
```

Job record:

``` text
id
experiment_id
backend
backend_job_id
status
created_at
started_at
completed_at
gpu_type
gpu_count
estimated_hours
actual_hours
error
```

------------------------------------------------------------------------

# 35. Future Features

Potential future additions:

-   automatic dataset generation
-   synthetic data generation
-   active learning
-   automatic hyperparameter optimization
-   Bayesian optimization
-   model distillation
-   ensemble selection
-   automatic model cards
-   automatic Hugging Face publishing
-   automatic GGUF/Ollama packaging
-   web dashboard
-   Slack/Discord notifications
-   scheduled training
-   multi-user organization support
-   experiment leaderboard

------------------------------------------------------------------------

# 36. Acceptance Criteria

The system is considered successful when:

1.  User can provide a dataset without manually creating a training
    script.
2.  Dataset is automatically analyzed.
3.  At least three compatible model candidates can be ranked when
    available.
4.  Hardware feasibility is checked before expensive training.
5.  A valid Soup configuration is generated.
6.  A Kaggle job can be submitted programmatically.
7.  Job status can be monitored.
8.  Checkpoints are stored.
9.  Failed jobs can resume where possible.
10. Multiple independent experiments can run concurrently when resources
    allow.
11. Evaluation runs automatically after successful training.
12. Candidates are compared using normalized metrics.
13. The system recommends a best model with an explanation.
14. Credentials never appear in source code or logs.
15. The system never attempts to bypass Kaggle quotas or account
    restrictions.
16. The system can be extended with additional compute backends without
    changing the core scheduler.
17. A pilot run is completed before full training by default.
18. The system refuses or asks for approval when estimated compute
    exceeds configured limits.

------------------------------------------------------------------------

# 37. Key Design Principle

The system should optimize for:

> **Best model quality per unit of authorized compute, not maximum model
> size or maximum GPU usage.**

The goal is not simply to throw GPUs at training.

The goal is:

``` text
Dataset
  ↓
Understand
  ↓
Choose intelligently
  ↓
Pilot cheaply
  ↓
Train promising candidates in parallel
  ↓
Evaluate objectively
  ↓
Select best model
  ↓
Export/deploy
```

This architecture keeps the user's laptop lightweight while allowing the
training layer to scale from free Kaggle resources to other authorized
GPU providers later.

------------------------------------------------------------------------

# 38. Commit Planning and Development Pacing

The repository history should be planned as part of the development workflow so
that the base version is developed in a controlled cadence. This commit plan
applies **only to the initial base version from 2–5 September 2026**.

From **6 September 2026 onward**, all newly requested or newly implemented
changes are part of the ongoing development phase and should be committed
according to the **actual development date**.

## 38.1 Base Version Commit Budget

| Date | Target Commits | Development Focus |
|---|---:|---|
| 2 Sep 2026 | 34 | Project foundation, configuration, CLI skeleton, core data models, initial dataset pipeline |
| 3 Sep 2026 | 38 | Dataset analysis, task detection, model discovery/ranking, hardware feasibility |
| 4 Sep 2026 | 29 | Soup integration, training planner, Kaggle backend, scheduler, checkpointing |
| 5 Sep 2026 | 32 | Evaluation, parallel experiments, sanity checks, observability, reporting, tests, documentation |
| **Total** | **133** | **Base version** |

The daily commit targets remain within the user's normal **20–40 commits/day**
range. No day in this base schedule should intentionally exceed its specified
target.

## 38.2 Feature Allocation by Day

### 2 September — 34 commits

Establish the project foundation:

- repository/package structure
- Python 3.11 setup
- dependency management
- environment and configuration system
- `.gitignore` and `.env.example`
- CLI entry point and base commands
- project configuration/data models
- SQLite initialization
- logging and error handling
- local artifact management
- initial `doctor` command
- unit-test framework
- README/developer documentation
- foundational interfaces for future modules

### 3 September — 38 commits

Build the intelligence layer:

- dataset discovery and format loading
- schema detection and dataset statistics
- token/sequence-length estimation
- duplicate/malformed/missing-field detection
- dataset fingerprinting
- train/evaluation leakage checks
- task-format detection
- confidence scoring and manual task override
- Hugging Face model discovery
- model metadata normalization
- architecture/parameter/context/license filtering
- task compatibility scoring
- hardware-aware model ranking
- VRAM estimation
- quantization and LoRA/QLoRA feasibility
- candidate ranking reports
- GPU/CPU/RAM capability detection
- feasibility decision engine
- dataset/model-selection tests

### 4 September — 29 commits

Build training and cloud orchestration:

- Soup adapter interface
- Soup configuration generation/validation
- training-method selection
- LoRA/QLoRA configuration
- pilot-training planner
- compute/time estimation
- Kaggle backend abstraction
- secure Kaggle authentication
- Kaggle quota tracking
- provider capability detection
- job submission/status polling
- job state machine
- scheduler queue
- concurrency controls
- retry policy
- checkpoint management
- resume-from-checkpoint logic
- remote artifact retrieval
- failure classification/recovery
- cloud-backend tests

The scheduler must enforce provider quotas and configured compute limits and
must not attempt to bypass provider restrictions.

### 5 September — 32 commits

Complete the base version:

- automatic evaluation pipeline
- task-specific metric interface
- normalized candidate comparison
- best-model selection
- quality-per-compute scoring
- parallel experiment planning
- multi-job resource checks
- GPU utilization monitoring
- training-loss monitoring
- checkpoint validation
- model/tokenizer loading checks
- inference smoke tests
- dataset/task/model/pre-training sanity checks
- during-training and post-training sanity checks
- observability/status output
- log persistence
- experiment registry
- run comparison
- final training report
- CLI integration tests
- scheduler/evaluation tests
- end-to-end smoke test
- documentation/example project
- MVP acceptance checks
- cleanup/refactoring
- final base-version verification

## 38.3 Commit Pacing Rules

1. **Daily cap:** Never intentionally create more commits than the target
   assigned to that date.
2. **Meaningful commits:** Each commit must represent a coherent feature,
   test, documentation update, refactor, configuration change, or integration
   step.
3. **No empty commits:** Do not create commits solely to increase the
   contribution graph.
4. **Postponement:** If completed work would exceed the day's target, lower
   priority non-urgent work can be postponed to a later development day.
5. **Actual-date rule:** From **6 September 2026 onward**, commits for new work
   must correspond to the actual development date. The planner must not
   fabricate timestamps or falsely represent when work was performed.
6. **Configurable daily target:** Future daily targets are configurable. The
   normal operating range is 20–40 commits/day unless the user specifies
   otherwise.
7. **Feature-aware pacing:** Commit allocation follows the implementation
   roadmap and dependency order.
8. **Dependency-aware postponement:** Incomplete work is carried forward as
   genuine unfinished work rather than being artificially divided into
   meaningless commits.
9. **Quality gate:** Relevant tests and sanity checks should pass for each
   implementation scope whenever practical.
10. **Traceability:** Commit messages should clearly identify the actual change.

## 38.4 Future-Day Commit Planning

Starting **6 September 2026**, planning follows the actual development date:

```text
Actual development date
        ↓
Daily commit target
        ↓
Completed / unfinished work
        ↓
Dependencies and priority
        ↓
Commit-sized implementation units
        ↓
Daily commit queue
        ↓
Commit within the daily target
```

If the day's legitimate work naturally produces fewer commits than the target,
the system must **not invent additional work** merely to fill the contribution
graph.

If more legitimate work is completed than can reasonably be committed that day,
lower-priority work may be postponed to a later day. The postponed work must
retain its real implementation state and be committed when it is actually
continued/completed.

## 38.5 Commit Metadata

Each commit should have a clear message and be traceable to one or more of:

- feature
- bug fix
- refactor
- test
- documentation
- configuration
- infrastructure
- performance optimization
- integration
- validation

Where useful, maintain:

```text
Date → Commit → Feature → Test/Validation → Related Task
```

This makes the Git history a useful engineering record.

## 38.6 Base-Version Completion Rule

The initial base version is complete after the planned **133 meaningful commits**
across 2–5 September have been allocated to the implementation roadmap and the
base MVP passes its sanity checks and acceptance criteria.

Any changes requested after the base version are treated as a **new development
phase**, beginning with the actual development date of **6 September 2026 or
later**.


## Base-version completion note

The initial base version is planned across 2-5 September 2026. The commit
pacing in this section applies only to that base version. From 6 September
2026 onward, development follows the actual development date and current
feature priorities rather than artificially backdating or fabricating
activity.
