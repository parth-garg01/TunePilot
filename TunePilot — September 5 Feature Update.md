# TunePilot — September 5 Feature Update
## Hybrid CLI + Interactive Terminal Chat UI

Continue development of the existing **TunePilot** project. Do not restart, rewrite, or replace the existing architecture.

This feature work is specifically for the **September 5, 2026 development phase**.

The goal is to make TunePilot a **hybrid CLI + interactive terminal chatbot**, similar in interaction style to modern AI coding agents, while keeping all existing TunePilot functionality accessible.

---

## 1. Product Experience

TunePilot should be launched directly from the terminal:

```bash
tunepilot
```

This should open an interactive terminal interface.

The user should be able to control the entire TunePilot system through:

1. Traditional CLI commands
2. Natural-language chat
3. File/document attachments
4. Image attachments
5. Keyboard navigation and editing

The CLI and chatbot must operate on the **same underlying TunePilot core and project state**.

Do NOT create two separate applications.

---

## 2. CLI Mode

All existing CLI functionality must remain available.

Examples:

```bash
tunepilot init
tunepilot dataset analyze ./data/train.jsonl
tunepilot models discover
tunepilot models list
tunepilot hardware
tunepilot compute
tunepilot plan
tunepilot train
tunepilot jobs status
tunepilot evaluate
tunepilot compare
tunepilot export
tunepilot doctor
```

The CLI must remain:

- scriptable
- automatable
- usable without the chatbot
- suitable for CI/headless execution
- compatible with existing TunePilot commands

Do not break existing commands while implementing the interactive interface.

---

# 3. Interactive Chat Mode

Running:

```bash
tunepilot
```

should launch an interactive terminal session.

Example:

```text
╭────────────────────────────────────────────────────╮
│                     TunePilot                      │
│       Autonomous LLM Fine-Tuning Assistant          │
╰────────────────────────────────────────────────────╯

Project: my-project
Status: Ready

TunePilot>
```

The user can type natural-language instructions.

Examples:

```text
TunePilot> Analyze my dataset.

TunePilot> Find the best base models for this dataset.

TunePilot> I don't care about training time. Give me the best
           model possible.

TunePilot> Use cloud compute if necessary.

TunePilot> Show me the current experiments.

TunePilot> Why was this model rejected?

TunePilot> Train the top 3 candidates.

TunePilot> Resume the failed experiment.

TunePilot> Compare the models.
```

The chatbot must translate these requests into calls to the existing TunePilot services.

---

# 4. Shared Core Architecture

The architecture must be:

```text
                       TunePilot
                           │
              ┌────────────┴────────────┐
              │                         │
        Traditional CLI          Interactive Chat
              │                         │
              └────────────┬────────────┘
                           ↓
                    TunePilot Core
                           │
            ┌──────────────┼──────────────┐
            ↓              ↓              ↓
       Dataset Engine  Model Engine  Compute Engine
            │              │              │
            └──────────────┼──────────────┘
                           ↓
                    Experiment Engine
                           ↓
                       SoupAdapter
```

The chatbot must NOT contain its own implementation of:

- model discovery
- training
- evaluation
- compute management
- experiment scheduling
- checkpoint handling

It should call the existing core services.

---

# 5. Terminal UX

Use a proper terminal interaction framework where appropriate.

Preferred technologies:

- `prompt_toolkit`
- `Textual`
- `Rich`

Choose the smallest and most appropriate combination for the existing codebase.

The interface should feel like a **terminal-native AI assistant**, not a web application rendered inside the terminal.

Support:

- streaming responses
- syntax highlighting where useful
- formatted tables
- progress indicators
- status panels
- errors/warnings
- command suggestions
- readable logs

Avoid excessive visual decoration.

---

# 6. Keyboard Controls

The interactive input must behave like a modern terminal editor.

At minimum:

```text
← / →       Move cursor
↑ / ↓       Navigate command history
Home        Beginning of line
End         End of line
Backspace   Delete character
Enter       Submit
Ctrl+C      Cancel current operation
Ctrl+D      Exit
Ctrl+L      Clear/redraw screen
```

Support multiline prompts.

For example:

```text
TunePilot> Analyze this dataset and determine whether SFT,
           DPO, or another training strategy would provide
           the best result.
```

The user must be able to freely edit the text before submission.

---

# 7. Chat History

Maintain an interactive session history.

The user should be able to:

```text
↑
```

to retrieve previous prompts.

The session should preserve relevant conversational context.

Example:

```text
User:
Find the best 7B models.

TunePilot:
I found 8 candidates...

User:
What about the Qwen one?

TunePilot:
The Qwen candidate ranks #1 because...

User:
Train it.

TunePilot:
I will start the pilot training...
```

The system should understand references such as:

- "it"
- "that model"
- "the previous experiment"
- "the second candidate"
- "that dataset"

using the current project/session state.

---

# 8. File Attachments

The chat interface must support attaching local files.

Examples:

```text
TunePilot> Analyze this:
./data/train.jsonl
```

and, where the terminal framework supports it, an explicit attachment mechanism.

Support:

- JSON
- JSONL
- CSV
- Parquet
- TXT
- YAML
- Markdown
- PDF
- DOCX
- logs
- configuration files
- images

Before processing attachments, display what was attached.

Example:

```text
Attachments

✓ train.jsonl       42.3 MB
✓ evaluation.jsonl   8.1 MB

Ready to analyze.
```

Do not silently upload files to external services.

---

# 9. Image Input

Allow users to provide images to the interactive assistant.

Example:

```text
TunePilot> Explain this training error.
```

with an attached screenshot.

Useful image inputs include:

- training-error screenshots
- GPU-monitor screenshots
- training curves
- model architecture diagrams
- dataset screenshots
- terminal output
- evaluation plots

The system should determine whether the configured LLM/backend supports vision before attempting image reasoning.

If vision is unavailable, explain this clearly rather than failing silently.

---

# 10. Document Context

Documents attached through chat should be usable as context.

For small documents, direct parsing/context may be sufficient.

For large documents, use:

```text
Document
   ↓
Parse
   ↓
Chunk
   ↓
Retrieve relevant content
   ↓
LLM context
```

Do not blindly inject entire large documents into prompts.

Dataset files should continue going through TunePilot's existing dataset-analysis pipeline rather than generic document processing.

---

# 11. Natural Language → TunePilot Actions

The chat layer must map natural language to structured TunePilot operations.

Example:

```text
User:

Find the best models for this dataset.
I don't care about training time.
Use cloud compute if it gives better results.
```

Internally interpret approximately as:

```yaml
intent: model_discovery

optimization:
  objective: quality_first

compute:
  strategy: auto

constraints:
  max_training_hours: null
```

Another:

```text
Use my local GPU only.
```

should map to:

```yaml
compute:
  strategy: local_4gb
```

Another:

```text
Train the top three models in parallel.
```

should map to:

```yaml
experiments:
  candidates: 3
  execution: parallel
```

The chatbot should call the existing planner/scheduler rather than implementing these operations itself.

---

# 12. Confirmation Before Expensive Operations

The chatbot should not blindly launch expensive training jobs from ambiguous requests.

Before a significant compute operation, display a concise plan.

Example:

```text
Training Plan

Models:
  1. Model A
  2. Model B
  3. Model C

Compute:
  Cloud GPU

Strategy:
  Pilot → Re-rank → Full Training

Objective:
  Quality-first

Estimated compute:
  ~8 GPU-hours

Proceed? [Y/n]
```

The user can respond:

```text
yes
```

or modify it:

```text
Only train the first two.
```

or:

```text
Use local compute instead.
```

---

# 13. Background Operations

Long-running jobs must not freeze the interactive chat.

The UI should remain responsive while:

- model discovery runs
- dataset analysis runs
- cloud jobs are submitted
- training runs
- evaluations run
- artifacts download

Example:

```text
TunePilot> Train the top candidates.

✓ Experiment plan created
✓ 3 jobs submitted

Active jobs:
  exp-001  Model A  RUNNING  42%
  exp-002  Model B  RUNNING  31%
  exp-003  Model C  RUNNING  18%

TunePilot>
```

The user should be able to ask:

```text
TunePilot> What's the status?
```

without interrupting the jobs.

---

# 14. Streaming Status

Long-running operations should provide live progress where possible.

Example:

```text
Experiment exp-001

Model: Qwen ...
Method: QLoRA
Compute: Cloud GPU

Step:          640 / 2000
Progress:      32%
Loss:          1.82
Tokens/sec:    31.4
VRAM:          14.2 / 16 GB
GPU utilization: 93%

Checkpoint:
  step-600

Elapsed:
  01:14:22
```

The UI should update rather than continuously dumping duplicate log lines.

Detailed raw logs should remain available through an explicit command such as:

```text
logs exp-001
```

---

# 15. Interactive Commands + Natural Language

Both styles should work in the same session.

For example:

```text
TunePilot> models list
```

and:

```text
TunePilot> Which model is currently ranked first?
```

must both work.

The user should also be able to mix them:

```text
TunePilot> jobs status
```

followed by:

```text
TunePilot> Why is exp-002 slower?
```

The system should maintain shared state between command execution and chat.

---

# 16. Help / Discoverability

The interactive interface should provide:

```text
/help
```

or:

```text
help
```

showing available commands and examples.

Example:

```text
TunePilot Commands

Project
  init
  status
  config

Dataset
  dataset analyze
  dataset inspect

Models
  models discover
  models list

Compute
  hardware
  compute
  compute providers

Training
  plan
  train
  jobs
  experiments

Evaluation
  evaluate
  compare
  select-best

Other
  doctor
  export
  help
  exit
```

Natural language should still be the primary conversational mechanism inside chat mode.

---

# 17. Exit / Resume

The user should be able to exit the terminal interface while jobs continue running remotely.

Example:

```text
TunePilot> exit
```

Later:

```bash
tunepilot
```

TunePilot should restore the project state and show:

```text
Welcome back.

Active experiments:
  exp-001  RUNNING  72%
  exp-002  COMPLETE
  exp-003  FAILED

Latest checkpoint:
  exp-001 / step-1400
```

---

# 18. Implementation Requirements

Before making changes:

1. Inspect the existing TunePilot codebase.
2. Identify the current CLI implementation.
3. Identify the core service layer.
4. Reuse existing project/state management.
5. Do not duplicate business logic.
6. Choose the smallest appropriate terminal UI dependency.
7. Add tests for the new interface.
8. Ensure existing CLI commands continue to work.

The interactive interface should be implemented as a thin presentation/input layer over the existing TunePilot core.

---

# 19. Testing / Sanity Checks

Add tests for:

- CLI startup
- interactive startup
- command history
- cursor movement
- multiline input
- Ctrl+C handling
- exit handling
- natural-language intent parsing
- command-to-core routing
- file attachment handling
- image attachment handling
- large-document handling
- background job interaction
- streaming status
- session restoration
- confirmation before expensive operations

Do not require an actual cloud GPU or LLM API for basic UI tests.

Use mocks/fakes for external services.

---

# 20. September 5 Commit Requirement

This feature belongs to the **September 5, 2026** development phase.

All commits created for this specific feature should use:

```text
Date: September 5, 2026
```

Keep the commits meaningful and feature-oriented.

Suggested logical progression:

```text
1. Add interactive terminal session foundation
2. Add terminal input editing/history
3. Add CLI/chat shared command routing
4. Add chat session state
5. Add streaming assistant output
6. Add formatted terminal status views
7. Add file attachment handling
8. Add image attachment handling
9. Add document context handling
10. Add natural-language action routing
11. Add confirmation flow
12. Add background job interaction
13. Add session restoration
14. Add help/discoverability
15. Add UI tests
16. Fix integration issues
17. Final CLI/chat validation
```

Do not create meaningless commits merely to increase commit count. Each commit must represent an actual implementation, test, fix, refactor, or documentation change.

---

# Final Acceptance Criteria

The feature is complete when:

```text
$ tunepilot
```

opens an interactive terminal assistant where the user can:

✓ Type natural-language requests  
✓ Execute traditional CLI commands  
✓ Edit input with arrow keys  
✓ Navigate command history  
✓ Use multiline prompts  
✓ Attach datasets/files  
✓ Attach images  
✓ Provide documents  
✓ Receive streaming responses  
✓ View running jobs  
✓ Continue interacting while jobs run  
✓ Confirm expensive operations  
✓ Reference previous models/experiments naturally  
✓ Exit and later resume the project  
✓ Access all existing TunePilot functionality  
✓ Use the same underlying core as the normal CLI  

The result should feel like:

**"A CLI that has the interaction model of an AI coding agent."**

Do not turn TunePilot into a web application at this stage. The interface being implemented here is **terminal-native**. A future GUI/Web UI may be built on top of the same core architecture.