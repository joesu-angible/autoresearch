# Handoff — `feat/autoreason-cls-umbrella` branch

Last touched 2026-04-25 by joesu-angible. Branch implements GitHub umbrella issue [#6](https://github.com/joesu-angible/autoresearch/issues/6) (children #4 autoreason tournament, #5 productness CLS branch). All scope was redirected from V1 → V2 per project decision.

---

## TL;DR — what's done

| Status | Item |
|---|---|
| ✅ | Autoreason tournament scaffold (`research_loop/`) — candidate schema, rule-based judges, evaluators, V2-only target adapters, CLI |
| ✅ | Productness CLS branch wired into `train_v2.py` — `ProductnessLCNet`, BCE loss, train + val eval, 10% deterministic holdout |
| ✅ | Two-output ONNX export (`export_onnx.py --include-productness`) |
| ✅ | Tournament promotion guardrails + deployment gate (productness as separate AND-clause veto, **not** blended into `combined`) |
| ✅ | 73 unit + integration tests pass |
| ✅ | 1-epoch GPU smoke run completed end-to-end (`combined=0.792`, `productness pos_acc=0.99 / neg_acc=0.76` on 45k val holdout) |
| ⏸️ | Full 30-epoch V2 production run (~7 h) — not yet launched |
| ⏸️ | DINO V2 smoke (T18) — adapter exists but not exercised end-to-end |
| ⏸️ | Nothing committed yet — all work is in working tree |

---

## Read these first (in order)

1. [SPEC.md](SPEC.md) — what we're building, success criteria, metric model.
2. [PLAN.md](PLAN.md) — 4-wave dependency graph + verification checkpoints.
3. [TASKS.md](TASKS.md) — T1–T20 tasks with acceptance criteria. T1–T9, T10–T17, T19 done; T18 + T20 outstanding.
4. [research_loop/autoreason_program_v2.md](research_loop/autoreason_program_v2.md) — how the tournament works (do-nothing-A rule, V2-only writes, judges, promotion + deployment gates).
5. [student_finetune/program_final_student_v2.md](student_finetune/program_final_student_v2.md) — V2 training protocol (preexisting).

Project memories that auto-load each session:
- `~/.claude/projects/-home-whiskey-workspace-project-central-v2-training-autoresearch/memory/MEMORY.md`
  - **Productness mandatory** — `USE_PRODUCTNESS_CLS=True` is the default; flag-off is debug only.
  - **Metric strategy** — `combined` stays retrieval-only; productness enters as separate guardrails (no weighted blend).

---

## Files I changed

```
M  .gitignore                                  # ignore productness_val_paths.txt + tournament artifacts
M  student_finetune/train.py                   # additive: forward_embeddings_train_with_summary, EpochStats productness fields, run_train_epoch productness kwargs (all default off)
M  student_finetune/train_v2.py                # ProductnessLCNet, val holdout loader, eval_productness, optimizer + run_train_epoch wiring, OUTPUT_DIR/cache fallback
M  student_finetune/export_onnx.py             # LCNetProductnessExport + --include-productness flag
M  student_finetune/prepare.py                 # bug fix: load_teacher_embeddings now backfills None entries before np.stack
?? SPEC.md, PLAN.md, TASKS.md, HANDOFF.md      # planning artifacts (this file)
?? research_loop/                              # entire tournament scaffold (new)
?? student_finetune/tests/test_productness_*.py
?? student_finetune/tests/test_export_onnx_productness.py
?? student_finetune/tools/build_productness_val.py
```

V1 default behavior is preserved — verified by `test_v1_default_path_unchanged` and by the V1 unit-test pass count being unchanged before/after my edits (19 pre-existing failures are unrelated).

---

## How to pick up where I left off

### 0. Sanity check the environment

```bash
cd /home/whiskey/workspace/project/central/v2/training/autoresearch

# Tests should be green
.venv/bin/python -m pytest research_loop/tests \
  student_finetune/tests/test_productness_targets.py \
  student_finetune/tests/test_productness_head.py \
  student_finetune/tests/test_export_onnx_productness.py \
  student_finetune/tests/test_productness_integration_smoke.py -q
# expect: 73 passed
```

### 1. Regenerate the productness val holdout (one-time, ~30 s, deterministic)

The file `student_finetune/productness_val_paths.txt` is gitignored. Regenerate via:

```bash
.venv/bin/python student_finetune/tools/build_productness_val.py
# expect: ~45k holdout paths from 450k total (10%)
```

### 2. Train V2 via fully-autonomous `tournament autoreason` (recommended)

The driver is the **autoreason loop** — Critic + Author B + Synthesizer LLM agents propose patches each pass, the adapter trains them under a per-candidate time budget, `promote.decide()` picks a winner via objective metrics, and the loop terminates when the incumbent wins `k` consecutive rounds. Zero human intervention between passes.

```bash
# Pre-flight: validate end-to-end with mocked subprocess (no GPU)
.venv/bin/python -m research_loop.tournament autoreason \
    --target student_v2 --max-passes 1 --dry-run

# Real run on student_v2 (productness ON; ~6-7h × per-candidate budget)
nohup .venv/bin/python -u -m research_loop.tournament autoreason \
    --target student_v2 \
    --max-passes 15 \
    --convergence 2 \
    --max-seconds-per-candidate 9000 \
    --hypothesis-seed "close productness_neg_acc gap below 0.85" \
    > research_loop/autoreason_student_v2.log 2>&1 &

# Real run on dino_v2 teacher (productness ON; ~50h × per-candidate budget)
nohup .venv/bin/python -u -m research_loop.tournament autoreason \
    --target dino_v2 \
    --max-passes 10 \
    --convergence 2 \
    --max-seconds-per-candidate 18000 \
    > research_loop/autoreason_dino_v2.log 2>&1 &
```

What each pass does (no manual steps):

1. **Critic LLM** reads `train_v2.py` + last 30 rows of `results_v2.tsv` + last 10 outcome JSON records → produces a structured `Critique`. CritiqueRecord written to history.
2. **Author B LLM** reads critique + trainer source → produces a unified-diff patch. PatchProposalRecord written.
3. **Synthesizer LLM** reads A's empty patch + B's patch (anonymized X/Y labels, no metric history) → produces a conservative AB synthesis patch. SynthesisRecord written.
4. For each kind ∈ {A, B, AB}: `apply_patch` (V1-safety + revert-on-exit) → `adapter.train(max_seconds=N)` → parse metrics → `Outcome` record. Working tree restored after each candidate.
5. `promote.decide()` picks winner under guardrails (`combined > A + 0.003`, `recall@1` not regressed, `productness_neg_acc` not regressed, status=success). `Decision` record written.
6. If A won → consecutive_count++. If `>= --convergence`, exit 0. Else next pass.
7. On `--max-passes` exhausted: exit non-zero, decision history preserved.

**Pre-conditions:**
- `ANTHROPIC_API_KEY` set in env or `~/.hermes/.env`
- Working tree clean (autoreason refuses dirty trees — patches need to revert cleanly)
- `productness_val_paths.txt` regenerated (see §1)

**Cost ceiling (default 15 passes × 5 LLM calls):** ≤ ~$5 per autoreason run on `claude-sonnet-4-6`. If you remove the time budget, a single hallucinated patch could chew 50h GPU. Always set `--max-seconds-per-candidate`.

**Audit trail:** `research_loop/history.jsonl` records every LLM call's raw text plus parsed dataclass fields. A 5-pass run writes ≥ 50 records spanning candidate / outcome / decision / critique / patch_proposal / synthesis types. Replayable post-hoc.

### 2b. Manual baseline run (when LLM access is unavailable)

If you can't reach the Anthropic API, fall back to the original `run-round --baseline-only` flow:

```bash
nohup .venv/bin/python -u -m research_loop.tournament run-round \
    --target student_v2 --baseline-only \
    > student_finetune/run_v2_production.log 2>&1 &
```

This emits a single A candidate, runs it, and writes the Outcome+Decision records — no Critic/Author B/Synthesizer, just the productization path.

### 2c. Single-candidate experiments

To run a hand-crafted patch outside the autoreason loop:

```bash
.venv/bin/python -m research_loop.tournament propose --target student_v2 --hypothesis "..."
# Edit the B / AB patches in research_loop/history.jsonl, then:
.venv/bin/python -m research_loop.tournament run --candidate <B_id>
.venv/bin/python -m research_loop.tournament promote --round <round_id>
```

### Cache / adapter notes (apply to all 3 modes)

Stage-1 (DINO) writes the LoRA adapter to `dino_finetune/output/best_adapter/`. Stage-2 (student) reads it from the same path. The teacher embedding cache is keyed on the adapter's SHA-256 prefix (`workspace/output/teacher_cache/dinov3_ft/<adapter_sha>/`) — a teacher retrain → new sha → fresh cache subdir built automatically. No `mv`, no `rm -rf`, no swap. Student first-epoch is ~14 min (cache warmup); subsequent epochs ~5–8 min.

Watch progress:

```bash
# Last lines without the noisy progress bar
awk 'BEGIN{RS="\r"} {print}' student_finetune/run_v2_production.log | grep -vE '^\s*[0-9]+/[0-9]+ \[' | tail -50
```

Success looks like:

```
Epoch 30/30 | loss=... distill=... arc=... cosine=...
  Productness (train): loss=... acc=... pos_acc=... neg_acc=...
  Productness (val):   loss=... acc=... pos_acc>0.97 neg_acc>0.85
  Retrieval: recall@1>=0.90 recall@5=...
  Combined: >=0.8588   <-- target
V2 TRAINING COMPLETE
```

### 2d. Real-LLM smoke (T8 — pre-flight before launching autoreason)

Before kicking off a multi-hour autoreason run, verify the LLM round-trip works against your `ANTHROPIC_API_KEY`. This costs ~$0.05 (3 Sonnet 4.6 calls), takes ~30 seconds, and never touches the GPU:

```bash
ANTHROPIC_API_KEY=sk-... .venv/bin/python -m research_loop.tools.autoreason_smoke
```

What it verifies:

- API key is reachable (env or `~/.hermes/.env`)
- Critic returns a parseable Critique with a non-empty summary
- Author B's diff passes `git apply --check` (or returns NO_PATCH)
- Synthesizer's diff passes `git apply --check` (or returns NO_PATCH)
- Working tree is unchanged after the smoke (`git apply --check` only, never applied)

Run this once before any production `tournament autoreason` invocation. If the smoke fails, the autoreason loop will fail in the same place — but on a 10-hour run instead of 30 seconds.

> **T8 status**: smoke script delivered (`research_loop/tools/autoreason_smoke.py`). The dev environment that built this PR has Claude Code OAuth but no raw `ANTHROPIC_API_KEY` exposed to the shell, so the manual verification has been deferred to the operator who launches the production run. Re-record the smoke output here once executed.

### 3. After the run — export ONNX + log to results_v2.tsv

```bash
cd student_finetune
../.venv/bin/python export_onnx.py --include-productness \
  --checkpoint workspace/output/distill_final_lcnet050_v2/best.pt \
  --output workspace/output/distill_final_lcnet050_v2/lcnet_student_productness.onnx
```

Then append a row to `student_finetune/results_v2.tsv` (same 9-column schema as V1):

```
commit  combined_metric  recall_1  recall_5  mean_cosine  distill_loss  peak_vram_mb  status  description
```

### 4. (Optional) DINO V2 smoke (T18)

The tournament adapter is wired but not yet exercised end-to-end. To smoke it:

```bash
cd dino_finetune
nohup ../.venv/bin/python -u train_dino_v2.py > run_v2.log 2>&1 &
```

DINO V2 doesn't have productness — it's the teacher, not the student. Just verify it produces `metrics_final_v2.json` so the tournament `parse_metrics` works.

### 5. Commit when ready

The branch has a non-trivial diff. Suggested commit boundaries:

```bash
# 1) The autoreason tournament (new isolated subsystem)
git add research_loop/ SPEC.md PLAN.md TASKS.md
git commit -m "feat(research_loop): autoreason tournament with V2-only target adapters"

# 2) Productness branch (changes to train.py / train_v2.py / export_onnx.py)
git add student_finetune/train.py student_finetune/train_v2.py \
        student_finetune/export_onnx.py \
        student_finetune/tests/test_productness_*.py \
        student_finetune/tests/test_export_onnx_productness.py \
        student_finetune/tools/build_productness_val.py \
        .gitignore
git commit -m "feat(student): productness CLS branch (issue #5) + V2 training wiring"

# 3) Bug fix in prepare.py (separable, atomic)
git add student_finetune/prepare.py
git commit -m "fix(prepare): backfill None embeddings before np.stack on unreadable images"

# 4) Handoff note
git add HANDOFF.md
git commit -m "docs: handoff note for feat/autoreason-cls-umbrella"
```

> **Per project memory**: ask the user which merge strategy (squash vs merge commit) before merging the PR.

---

## Key code locations (jump points)

| Concern | File:lines |
|---|---|
| Productness model class | `student_finetune/train_v2.py` (`ProductnessLCNet`) |
| Productness data → BCE in training | `student_finetune/train.py` `run_train_epoch` (search for `_productness_active`) |
| Productness val holdout loader | `student_finetune/train_v2.py` (`load_productness_val_paths`, `eval_productness`) |
| ONNX 2-output export | `student_finetune/export_onnx.py` (`LCNetProductnessExport`) |
| Tournament candidate schema | `research_loop/candidate.py` |
| Promotion guardrails | `research_loop/promote.py` (`decide`, `is_deployable`) |
| Rule-based judges | `research_loop/judges.py` |
| V2-only safety check | `research_loop/targets/_base.py` (`V1_FORBIDDEN_PATHS`, `assert_no_v1_writes`) |
| Tournament CLI | `research_loop/tournament.py` |

---

## Things to know that aren't obvious

1. **`combined_metric` is retrieval-only by design** — do not blend productness into it. See [project memory: Metric strategy](.claude/projects/.../memory/project_metric_strategy.md). The promotion logic adds productness as a *separate* AND-clause veto.
2. **`USE_PRODUCTNESS_CLS = True` is the default** — productness is mandatory for V2+ deployment. Flag-off path exists only as a V1-parity proof.
3. **V1 files are tournament-time read-only, dev-time editable** — only additive edits that preserve V1 default behavior. The `V1_FORBIDDEN_PATHS` list in `research_loop/targets/_base.py` enforces this at tournament runtime.
4. **Teacher cache lives at `workspace/output/teacher_cache/dinov3_ft/<adapter_sha>/`** (local fallback when `/data/training/reid/workspace` isn't writable). Adapter-versioned: a new teacher adapter automatically writes to a new `<sha>/` subdir on first student epoch — no manual `rm -rf` needed when teacher is retrained. Old subdirs can be deleted to reclaim disk but training won't break if you leave them.
5. **`productness_val_paths.txt` is gitignored**. The build script in `student_finetune/tools/build_productness_val.py` uses SHA-1 bucketing on relative paths so the holdout is deterministic and portable across machines.
6. **Pre-existing V1 test failures (19 of them) predate this branch** — verified with `git stash`. Don't try to fix them as part of this work.

---

## Open questions / known gaps

- The `bg` smoke run process (PID 2519006) finished and exited; no live training is running.
- `DEPLOY_MIN_COMBINED = 0.86` in `research_loop/promote.py` is a placeholder — calibrate after the first 30-epoch run lands.
- LLM-judge plug-in (`Judge` Protocol in `research_loop/judges.py`) is stubbed but not implemented; rule-based only for now.
- `metrics_final_v2.json` includes productness keys when the flag is on, but the tournament adapter `log_row()` doesn't auto-append to `results_v2.tsv` yet — that's a small follow-up if you want fully automated tournament cycles.

If you have questions for me, the conversation transcript and project memories are in `~/.claude/projects/-home-whiskey-workspace-project-central-v2-training-autoresearch/`.
