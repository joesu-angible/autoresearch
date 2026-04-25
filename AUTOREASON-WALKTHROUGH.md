# Autoreason Walkthrough — how it actually works in our training

This doc walks through one concrete autoreason run end-to-end so you can see exactly what each component does, what data flows where, and what you'll see in logs. For the architectural overview see [README.md §Autoreason](README.md#-autoreason-training-that-improves-itself-overnight).

## Setup

```bash
.venv/bin/python -m research_loop.tournament autoreason \
    --target student_v2 --max-passes 5 --convergence 2 \
    --max-seconds-per-candidate 9000
```

Three roles are LLMs called via subprocess (`hermes` / `claude` / `codex` CLI). The "winner" of each pass is decided by **objective ML metrics** through `promote.decide()`, not by an LLM judge.

---

## Pass 1 (cold start)

### Step 0 — A = current state

`A` is whatever `student_finetune/train_v2.py` currently is on the working tree. Suppose its last training run produced:

```
combined=0.812, recall@1=0.889, productness_neg_acc=0.83
```

### Step 1 — Critic finds problems

**Inputs** to Critic:
- Full source of `train_v2.py`
- Last 30 rows of `results_v2.tsv` (historical metric trajectory)
- Last 10 outcomes from `history.jsonl`

**System prompt**: "find concrete problems only — do not propose fixes" (temperature 0.3 for consistency).

**Output** (persisted to `history.jsonl` as `record_type="critique"`):
```
Problems:
1. PRODUCTNESS_CLS_WEIGHT=0.02 looks too low — productness_neg_acc plateaus at 0.83
2. WARMUP_STEPS=100 is short for cosine schedule, early loss is noisy
3. asymmetric label smoothing eps_neg=0.02 may be insufficient for hard negatives
```

### Step 2 — Author B writes a unified diff

**Inputs**: critique + trainer source.
**System prompt** explicitly enumerates `V1_FORBIDDEN_PATHS` (defense in depth).
**Output** (temperature 0.8 for diversity), persisted as `record_type="patch_proposal"`:

```diff
--- a/student_finetune/train_v2.py
+++ b/student_finetune/train_v2.py
@@ -45,7 +45,7 @@
-PRODUCTNESS_CLS_WEIGHT = 0.02
+PRODUCTNESS_CLS_WEIGHT = 0.08
@@ -78,3 +78,3 @@
-WARMUP_STEPS = 100
+WARMUP_STEPS = 500
```

This is candidate **B**.

### Step 3 — Synthesizer creates a conservative blend

**Inputs**: A's source + B's patch — labels anonymized to `X` / `Y`, **no metric history shown**.
**Principles in the prompt**: smallest-subset, halve magnitudes, prefer additions, willing to NO_PATCH.
**Output** persisted as `record_type="synthesis"`:

```diff
-PRODUCTNESS_CLS_WEIGHT = 0.02
+PRODUCTNESS_CLS_WEIGHT = 0.04   # halve B's magnitude
-WARMUP_STEPS = 100
+WARMUP_STEPS = 500              # low-risk change, take it whole
```

This is candidate **AB**.

### Step 4 — Run all three under the time budget

```python
for kind in [A, B, AB]:
    git apply <patch>                      # A is empty patch
    adapter.train(max_seconds=9000)        # real training, real GPU
    git apply -R <patch>                   # working tree restored
```

Suppose the trainers produce:

| candidate | combined | recall@1 | productness_neg_acc | status |
|---|---|---|---|---|
| A | 0.812 | 0.889 | 0.83 | success |
| B | 0.834 | 0.901 | 0.86 | success |
| AB | 0.828 | 0.895 | 0.85 | success |

### Step 5 — `promote.decide()` picks the winner via objective metrics

For B vs A:
- combined: +0.022 > NOISE_BAND (0.003) ✅
- recall@1: no regression ✅
- productness_neg_acc: no regression ✅
- → **B wins**

### Step 6 — B becomes the new A

B's patch is committed to the working tree; the loop continues.

```
consecutive_A_wins = 0   # reset, B won this pass
```

---

## Pass 2 (A = previous B)

Critic re-reads the *new* `train_v2.py` plus updated `history.jsonl`. It might say:

```
Problems:
1. productness_neg_acc reached 0.86 but still below 0.90 deploy gate
2. recall@1 gain came from productness weighting; retrieval-only metric flat
```

Author B writes another patch (perhaps changing ArcFace margin), Synthesizer blends, three more trainings run, winner chosen.

---

## Convergence

```
Loop ends when EITHER:
  - A wins k=2 consecutive passes (LLM proposals stop helping → converged)
  - max_passes (5) reached (forced stop)
```

Example trajectory:
```
Pass 1: B wins  → consecutive_A=0
Pass 2: B wins  → consecutive_A=0
Pass 3: A wins  → consecutive_A=1
Pass 4: A wins  → consecutive_A=2  → converged
```

---

## What you observe while it runs

```
research_loop/runs/run1777891234-a3f8c9/
  ├── summary.json       refreshed every pass — bot-readable run state
  ├── autoreason.log     full narrative (Critic findings, decisions, errors)
  └── autoreason.pid     liveness check

research_loop/history.jsonl   every LLM call's raw text + every outcome + every decision
```

Anytime:
```bash
.venv/bin/python -m research_loop.tournament status --target student_v2
```

prints current pass, latest critique, best metrics so far, and liveness.

---

## Safety guardrails (why this can't go off the rails)

1. **V2-only**: if Author B tries to edit `student_finetune/train.py` (V1), `patch.py` rejects it before `git apply` runs. The system prompt forbids V1 edits *and* the applicator double-checks — defense in depth.
2. **Time budget**: any candidate exceeding 9000s gets SIGTERM → 30s grace → SIGKILL. `Outcome.status="timeout"` → `promote.decide()` rejects it regardless of partial metrics. Without this, one hallucinated patch could chew 50h.
3. **Reversible patches**: every candidate's run is wrapped in `try / git apply -R ... finally` — working tree is always clean after a pass, even on errors.
4. **Productness deploy gate**: even if `combined` improves, deployment is blocked unless `productness_neg_acc ≥ 0.85`, `pos_acc ≥ 0.97`, and `combined ≥ DEPLOY_MIN_COMBINED (0.86)`. `decide()` and `is_deployable()` are separate — winning a tournament round ≠ shippable.
5. **Synthesizer sees no metrics**: synthesizer is given anonymized X/Y patches only, never metric values, to prevent over-fitting to one data point.

---

## One-sentence summary

> **Each pass: LLM reads the trainer + history → LLM finds problems → LLM writes a patch → three versions (current, revised, blended) actually train on GPU → winner picked by objective metrics → winner becomes next pass's starting point, until LLM can no longer improve it.**

---

## Where the code lives

| Component | File |
|---|---|
| Critic / Author B / Synthesizer agents | `research_loop/agents/{critic,author,synthesizer}.py` |
| Multi-CLI subprocess wrapper | `research_loop/agents/client.py` |
| Patch applicator (V1 safety + revert) | `research_loop/patch.py` |
| Tournament loop + convergence | `research_loop/tournament.py` (`cmd_autoreason`) |
| Time budget primitive | `research_loop/targets/_base.py` (`adapter.train(max_seconds=N)`) |
| Promotion guardrails | `research_loop/promote.py` (`decide`, `is_deployable`) |
| Audit trail records | `research_loop/candidate.py` (Critique / PatchProposal / Synthesis / Outcome / Decision) |
