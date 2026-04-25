# Autoreason Tournament Protocol — V2

Outer-loop experiment controller for `train_v2.py` (student) and `train_dino_v2.py` (DINO). Inspired by `NousResearch/autoreason`'s blind-tournament pattern, reimplemented here (no code copied — license unclear).

## Cardinal rules

1. **Do-nothing is a candidate.** Every round must include `A = incumbent baseline`. If A wins, no GPU is spent on patching; this is by design.
2. **V2 only.** The tournament must never write into V1 files. Forbidden paths are listed in [`research_loop/targets/_base.py`](targets/_base.py) (`V1_FORBIDDEN_PATHS`). Adapters refuse log paths whose name is `results.tsv` or anything not `*_v2.tsv` / `results_v2`.
3. **Objective eval is the final authority.** Judge scores rank proposals; they do not promote. Promotion requires `combined ≥ A + NOISE_BAND` AND `recall@1` not regressed beyond `RECALL_REGRESSION_TOLERANCE`. See [`research_loop/promote.py`](promote.py).

## Candidate triple

Each round emits three candidates of `kind ∈ {A, B, AB}`:

- **A** — do nothing. `patch=""`, no `changed_files`, `rollback="N/A"`.
- **B** — the proposed change. Must include hypothesis, expected metric (numeric / quantifiable), risks list, and rollback condition.
- **AB** — conservative synthesis (e.g. apply B at half strength, or restrict to a subset).

The candidate schema is [`research_loop/candidate.py`](candidate.py); JSONL history lives at `research_loop/history.jsonl`.

## Judges (rule-based)

Three rubric scorers in [`research_loop/judges.py`](judges.py), each in `[0, 1]`:

| Judge | Signal |
|---|---|
| clarity | hypothesis ≥ 20 chars; expected_metric is quantifiable |
| risk | risks listed AND rollback present (A is auto-1.0) |
| prior_evidence | candidate cites historical `results_v2.tsv` rows |

Aggregate = mean of the three. Ties resolve to incumbent A.

LLM judges are an extension point (see `Judge` Protocol). Out of scope for the first cut.

## Promotion guardrails

Pure functions in [`research_loop/promote.py`](promote.py). All must hold for a non-A candidate to be promoted:

1. `Δcombined > NOISE_BAND` (currently `0.003`).
2. `recall@1` not regressed beyond `RECALL_REGRESSION_TOLERANCE` (currently `0.005`).
3. `productness_neg_acc` not regressed beyond `PRODUCTNESS_NEG_ACC_REGRESSION_TOLERANCE` (currently `0.02`). Skipped when either A or the challenger lacks the field — keeps V1-history compatibility.
4. Rollback condition present.

If none hold, the incumbent A wins.

**`combined` stays retrieval-only** (`0.5 * recall@1 + 0.5 * mean_cosine`) — productness is *not* blended in. Project decision 2026-04-25: a weighted blend hides tradeoffs (a +productness / -recall candidate could win the wrong way), so productness regression is a separate AND-clause veto.

## Deployment gate (separate from tournament)

`is_deployable(result) → DeployVerdict` answers a different question: *can this checkpoint ship?*

Independent thresholds (absolute, not deltas):
- `combined ≥ 0.86`
- `productness_pos_acc ≥ 0.97`  (don't reject products as personal items)
- `productness_neg_acc ≥ 0.85`  (don't accept personal items as products)

A candidate can win promotion (it's the new best) yet still fail the deployment gate. Use `is_deployable()` before shipping; `decide()` for routine experiment iteration.

## Targets

Adapters in [`research_loop/targets/`](targets/) provide a uniform `apply_patch / train / log_row` surface. Two targets are wired in:

- `student_v2` → `student_finetune/train_v2.py`, logs to `student_finetune/results_v2.tsv`, metrics at `/data/.../metrics_final_v2.json`.
- `dino_v2` → `dino_finetune/train_dino_v2.py`, logs to `dino_finetune/results_v2.tsv`.

Adapter constructors enforce that `RESULTS_TSV` is a V2 path; any patch that touches a `V1_FORBIDDEN_PATHS` entry is rejected before training.

## CLI

```
python -m research_loop.tournament propose --target {student_v2|dino_v2} [--hypothesis "..."]
python -m research_loop.tournament rank
python -m research_loop.tournament run --candidate <id> [--no-dry-run]
```

Default `run` is `--dry-run`. Live training requires `--no-dry-run`.

## When A keeps winning

If incumbent A wins repeatedly along the same search axis, stop perturbing that axis and try another (per Autoreason's "convergence restraint"). Capture the search-axis pivot as a project memory rather than an ever-larger candidate B.
