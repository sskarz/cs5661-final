# Autoresearch: pathZ M3A-format SFT smoke (AndroidLab-faithful)

## Objective

Reproduce the **AndroidLab XML-mode SFT recipe** (Xu et al. 2024,
arXiv:2410.24024) at smoke scale on Gemma 4 E2B, in M3A's exact prompt
and action vocabulary. AndroidLab reports 2.17 → 23.91% on the
AndroidLab benchmark using SFT alone; the bar for our pathZ plan is
beating the M3A AW baseline by ≥15pp absolute.

### Phase 1 — infra validation (DONE through run 17)

Validated on AC single-step alone:
- M3A action vocabulary alignment between training labels and eval
  parser ✓
- Balanced training prevents click-class collapse ✓
- 300-step QLoRA at lr=2e-4 + projector unlocked + lora_r=32
  → +2.6pp full_match vs zero-shot baseline ✓
- 99% parse rate / 99% reason rate after training ✓

### Phase 2 — AndroidLab data integration (CURRENT, run 18+)

The infra works. Now make the smoke faithful to AndroidLab's actual
training distribution:

1. **Pull `THUDM/Android-Lab` Instruction dataset** (~6K multi-step
   trajectories with history baked into prompt).
2. **Convert their trajectories to our unified M3A action vocabulary**.
3. **Mix AndroidLab + AndroidControl** at the day-2 weighting (45/35
   split).
4. **Use a longer, ReAct-style Reason** (Thought: rationale → Action:
   JSON) per their CoT format.
5. **Smoke-train and prove the AndroidLab data shifts the offline
   metrics differently than AC-only training**.

### What we still do NOT do

We do NOT run full AW evals here. Phase 2 success criterion is offline
action-match on a mixed (AndroidLab-val + AndroidControl-val) eval set,
showing the recipe is positive transfer at smoke scale.

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
