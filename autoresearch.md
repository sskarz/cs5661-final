# Autoresearch: pathZ M3A-format SFT smoke

## Objective

Validate the AndroidLab-style **SFT recipe** on Gemma 4 E2B before any
full training run. The pathZ plan (ANDROID_WORLD_PLAN.md) commits to a
multi-source SFT mixture in **M3A's exact prompt + action vocabulary**.
Before scaling to the full 8K-step plan we must prove on a tiny slice
that:

1. The data converter actually produces well-formed M3A-format examples.
2. The trainer converges (training loss drops).
3. The trained LoRA, evaluated offline on M3A-format prompts, **beats
   the zero-shot Gemma 4 E2B baseline** on a held-out 200-row eval.

If any of the above fails at smoke scale, the full SFT will fail too —
better to find that out in 5 minutes than 24 hours.

We do NOT run full AW evals here. The smoke metric is offline action-
match on M3A-format prompts derived from AndroidControl-val. Once smoke
shows positive results, we move to full SFT (separate plan).

## Metrics

- **Primary**: `full_match` — % of eval rows where the model emits a
  parseable `Reason: ... Action: {...}` AND both action_type and the
  primary grounding arg (index / direction / app_name) match gt.
  **Higher is better.**
- **Secondary**:
  - `type_match` — % where action_type alone matches gt (looser, signals
    whether the model picked the right *kind* of action).
  - `parse_pct` — % rows that produced a parseable `Action: {...}`.
  - `reason_pct` — % rows with a `Reason:` prefix.
  - `train_loss_final` — final training loss (training runs only).
  - `type_match_<at>` — per-action-type breakdowns when ≥5 rows.

## How to Run

`./autoresearch.sh` — outputs `METRIC name=number` lines.

Modes:
- `RECIPE=baseline ./autoresearch.sh` — zero-shot Gemma 4 E2B on the
  M3A-format eval set, no training. Sets the floor.
- `./autoresearch.sh` (default) — trains using the current recipe in
  `scripts/pathZ/train_smoke.py`, evals the resulting LoRA.

Smoke data (cached after first build):
- `data/pathZ/smoke/train.jsonl` (2000 rows)
- `data/pathZ/smoke/eval.jsonl`  (200 rows held out from AC-val)

## Files in Scope

- `scripts/pathZ/m3a_format.py` — M3A prompt rendering, Path-W → M3A
  action conversion, parser, action-match scorer. Stable; rarely edit.
- `scripts/pathZ/prepare_smoke_data.py` — builds the cached train/eval
  JSONL. Edit to vary data weighting, prompt format, history shape.
- `scripts/pathZ/train_smoke.py` — **the recipe**. Edit hyperparams,
  LoRA config, loss masking each iteration.
- `scripts/pathZ/eval_smoke.py` — eval harness. Edit only if changing
  scoring methodology.
- `autoresearch.sh` — one-iter wrapper.

## Off Limits

- `data/androidcontrol_a11y_native_v3/` — source data, do not modify.
- `outputs/gemma4-e2b-pathW-lora-runI/`..`runL/` — prior project
  artifacts, immutable.
- `android_world/` (sibling repo) — wrapper + dispatch already written;
  smoke does not touch them.

## Constraints

- One RTX 4090, 24 GB VRAM.
- No full AW benchmark runs from this loop.
- Single iteration must finish in ≤ ~10 min wall to keep the loop tight.
- The smoke must use M3A's exact action vocabulary so anything we ship
  here is directly compatible with the AW M3A wrapper at deploy time.

## What's Been Tried

(updated each iteration; see `experiments/worklog.md` for the running
narrative)

- (pending) Run 1: zero-shot baseline on M3A-format prompts.
