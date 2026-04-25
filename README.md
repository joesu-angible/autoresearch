# autoresearch

Autonomous ML research platform where AI agents explore model configurations, train experiments, and discover better architectures without human intervention.

This project applies the [autoresearch](https://github.com/karpathy/autoresearch) pattern to visual re-identification (ReID): a lightweight student model (LCNet) is distilled from multiple large teacher models (DINOv2, DINOv3, C-RADIO) through autonomous experimentation.

## Project Structure

Each research topic lives in its own subfolder, designed to be run with that directory as the working directory.

```
autoresearch/
  student_finetune/     Student model distillation (LCNet) from multiple teachers
    prepare.py          Data loading, teacher definitions, TEACHER_REGISTRY
    train.py            Training loop, model architecture, loss functions
    build_caches.py     Pre-build teacher embedding caches
    tests/              Test suite for student distillation code
  dino_finetune/        DINOv3 ViT-H+ fine-tuning with LoRA for teacher model
    train_dino.py       DINOv3 contrastive fine-tuning with LoRA adapters
  RADIO/                Local clone of C-RADIO model repository (torch.hub source)
  workspace/            Runtime artifacts (teacher caches, outputs, results)
  program.md            Agent instructions for autonomous experimentation
  pyproject.toml        Project dependencies
```

## Usage

### Student Distillation

```bash
cd student_finetune

# Build teacher embedding caches (one-time, requires GPU)
python build_caches.py

# Run training
python train.py
```

### DINOv3 Fine-tuning

```bash
cd dino_finetune
python train_dino.py
```

### Running Tests

```bash
cd student_finetune
python -m pytest tests/ -x -q
```

---

## 🤖 Autoreason: training that improves itself overnight

This repo ships a **fully autonomous research loop** modeled on the [NousResearch autoreason paper](https://github.com/NousResearch/autoreason). Three fresh LLM agents — **Critic**, **Author B**, **Synthesizer** — generate code patches, run them under a time budget, and let objective ML metrics pick the winner. Loops until the incumbent wins `k=2` consecutive rounds. **Zero human intervention between passes.**

```
Each pass:
  Critic LLM  ─── reads train_v2.py + last 30 results_v2.tsv rows + last 10 outcomes
              └─→ structured Critique (problems only, no fixes)

  Author B LLM ── reads critique + trainer source
              └─→ unified-diff patch (the new candidate B)

  Synthesizer LLM ── reads A + B's patch (anonymized X / Y, no metrics)
              └─→ conservative AB synthesis patch

  For each kind ∈ {A, B, AB}:
      git apply (V1-safety + auto-revert)
        → adapter.train(max_seconds=N) → metrics
        → promote.decide() with productness-aware guardrails
        → Outcome record persisted to history.jsonl

  Decision recorded. If A won → consecutive_count++. Convergence at k=2.
```

### Quickstart — pre-flight smoke (~30s, ~3 LLM calls)

Verify the LLM round-trip works before launching a multi-hour training run:

```bash
# Default: hermes (matches the repo's existing autoresearch convention)
.venv/bin/python -m research_loop.tools.autoreason_smoke

# Or pick a specific CLI
.venv/bin/python -m research_loop.tools.autoreason_smoke --llm-cli claude --llm-model claude-sonnet-4-6
.venv/bin/python -m research_loop.tools.autoreason_smoke --llm-cli codex
```

### Recommended config — mixed-model setup

Different roles benefit from different model classes (paper §7.3). Suggested baseline that respects token budgets:

| Role | CLI | Why |
|---|---|---|
| **Critic** (analytical, low temp) | `hermes --provider openai-codex` | User-habit-tuned codex catches problems specific to your project |
| **Author B** (creative volume, temp 0.8) | `codex` direct | Unlimited; built for code patches; raw diversity > habit constraints |
| **Synthesizer** (conservative pick) | `hermes --provider openai-codex` | User-taste-tuned for "halve and keep the safer part" |

```bash
.venv/bin/python -m research_loop.tournament autoreason \
    --target student_v2 \
    --max-passes 15 --convergence 2 \
    --max-seconds-per-candidate 9000 \
    --critic-cli hermes --critic-model openai-codex/gpt-5-codex \
    --author-cli codex --author-model gpt-5-codex \
    --synthesizer-cli hermes --synthesizer-model openai-codex/gpt-5-codex
```

Or use Opus for everything if tokens aren't the bottleneck:

```bash
.venv/bin/python -m research_loop.tournament autoreason \
    --target student_v2 \
    --max-passes 15 --convergence 2 --max-seconds-per-candidate 9000 \
    --llm-cli claude --llm-model claude-opus-4-7
```

### Escalation — break a stalled run

If A keeps winning but `combined` plateaus, inject one Opus critic pass to surface a deeper analytical view:

```bash
.venv/bin/python -m research_loop.tournament autoreason \
    --target student_v2 --max-passes 1 --convergence 99 \
    --max-seconds-per-candidate 9000 \
    --critic-cli claude --critic-model claude-opus-4-7 \
    --author-cli codex --author-model gpt-5-codex \
    --synthesizer-cli hermes --synthesizer-model openai-codex/gpt-5-codex
```

`--max-passes 1 --convergence 99` = "one pass, don't try to converge, just inject Opus's view". Then resume the cheap config.

### Supported CLIs

`hermes` (default), `claude`, `codex`. Selection precedence: explicit `--llm-cli` flag → `AUTORESEARCH_LLM_CLI` env var → `hermes`. Hermes routes through 21 providers (`auto`, `openrouter`, `nous`, `anthropic`, `gemini`, `xai`, `ollama-cloud`, `huggingface`, `kimi-coding`, `stepfun`, `minimax`, `arcee`, `nvidia`, …) — pick via `--llm-provider <name>`.

### Audit trail

Every LLM call's raw text is persisted to `research_loop/history.jsonl` as `record_type ∈ {critique, patch_proposal, synthesis, candidate, outcome, decision}`. A 5-pass run writes 50+ replayable records.

### "How is training going?" — Slack-bot-friendly status

Every autoreason run writes a single canonical summary file an external agent can read:

```
research_loop/runs/<run_id>/
  ├── summary.json       ← structured run state, refreshed every pass
  ├── autoreason.log     ← narrative output (Critic findings, decisions, errors)
  └── autoreason.pid     ← process id (for liveness check)
```

A `<target>_CURRENT.txt` pointer in `research_loop/runs/` tracks the active run per target.

One command bots can run to answer "what's happening?":

```bash
$ .venv/bin/python -m research_loop.tournament status --target student_v2

autoreason status — student_v2
  Run:          run1777891234-a3f8c9 (alive (4h 23m))
  Started:      2026-04-26T12:34:56Z
  Status:       running
  Pass:         5 / 15  (consecutive A wins: 1 / 2)
  Last decision: A wins (round r1777891234-abc, deployable=True)
                 reason: candidate B beats A by Δcombined=+0.0034, recall@1=0.9012 vs A=0.8989
  Best so far:  combined=0.8341  recall@1=0.9012  neg_acc=0.84
  Latest critique: productness neg_acc plateau at 0.84; commodity ratio may be too high
  Logs:
    narrative: research_loop/runs/run1777891234-a3f8c9/autoreason.log
    history:   research_loop/history.jsonl (60 records)
    run dir:   research_loop/runs/run1777891234-a3f8c9
```

Resolution order: `--run RUN_ID` → `--target TARGET` (uses CURRENT pointer) → newest run dir.

Wire `tournament status` into a Slack slash-command, ntfy webhook, or Hermes tool — same shell command, paste output verbatim.

See [HANDOFF.md](HANDOFF.md) for the full operator runbook (modes 2a / 2b / 2c, cache invariants, smoke procedure, troubleshooting).

## Adding New Research Topics

Create a new subfolder following the existing pattern:

1. Create `new_topic/` with its own `train.py` and supporting modules
2. Use relative paths with `../` prefix to reference shared resources (e.g., `../workspace/`, `../RADIO/`)
3. Each subfolder should be self-contained and runnable from its own directory as CWD
