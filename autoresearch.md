# Autoresearch: Gemma 4 E2B → ≥15% on AndroidWorld-116

## Objective

**Ship a Gemma 4 E2B agent that scores ≥15% success rate on the full
AndroidWorld benchmark (116 tasks, M3A harness)** without cheating.

The 15% bar is the success criterion. Pure-prompt baseline floor on
AW-116 was 0% (see `FUTURE_WORK.md`). Best smoke result so far is
r22 at 50% on a curated AW-10 slice but variance studies (r22/r26/r28/r29
= 50/10/20/0%) showed that slice is too noisy to distinguish recipes.
The ship gate is full AW-116, not AW-10 or AW-20.

### Workflow gate

1. **Iterate on AW-20 smoke** (curated 20-task slice via M3AA11Y harness)
   to ratchet recipes cheaply. Target: ≥15% AW-20 SR with low variance.
2. **When AW-20 SR ≥ 15%** on a recipe, run the **full AW-116** to
   confirm the smoke generalises. This is the ship gate.
3. **No cheating**: do not curate AW-116 task selection, do not train on
   AW task templates, do not contaminate the train set with AW
   instructions, do not relax the harness's success criterion.

### Lineage of attempts

- Phase 1 (AC-only SFT): plateaued at +2.6pp AC offline, no AW lift.
- Phase 2 (AC+AL mix, M3A schema): r22 50% on AW-10 outlier, true mean
  ~25% with σ≈21pp. Pure-a11y r31 = 10% on AW-20.
- Phase 3 (current): teacher distillation (r34) — Gemma 4 31B 4-bit
  generates blind, kept rows have either matched verb+arg or 2nd-pass
  salvaged reason for matched verb. Goal: replace synthetic one-line
  reasons with model-grade rationales. Off-policy notes in
  `scripts/pathZ/distill_teacher.py`.

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

- **Primary**: `aw_success_rate` — fraction of AW tasks the agent
  completes per the harness `task.is_successful()` check.
  **Higher is better.** Measured on AW-20 smoke slice each iter; on full
  AW-116 only when AW-20 ≥ 15%.
- **Secondary** (cheap signals computed each iter):
  - `aw_n_ok` / `aw_total` — raw success counts.
  - `ac_full_match` — AC-val offline action-match (grounding proxy).
  - `al_full_match` — AndroidLab-val offline action-match (trajectory proxy).
  - `train_loss_final` — final training loss.
  - `parse_pct` / `reason_pct` — emission well-formedness.
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
- Full AW-116 runs are **gated**: only triggered when an AW-20 smoke
  scores ≥15% on the same recipe. AW-116 takes ~6h on this hardware so
  is not the iteration loop, only the ship test.
- AW-20 smoke iter target: ≤ ~15 min wall.
- Harness is no longer fixed to M3A. You may experiment with new
  harnesses (e.g. swapping in a different agent loop, prompt structure,
  history representation, action vocabulary) as long as success is still
  measured by the AndroidWorld task `is_successful()` check. New
  harnesses should live alongside `m3a_a11y.py` in
  `android_world/agents/` and be wired through `run.py`. The action
  vocabulary the model emits must still be the AW-compatible one (or
  losslessly mappable to it) so the harness can dispatch actions.
- **No cheating**: no AW-task-template mining into train data, no
  selecting which AW tasks to score on after the fact, no relaxing of
  harness success criteria.

## What's Been Tried

(updated each iteration; see `experiments/worklog.md` for the running
narrative)

### Phase 1 — AC-only smoke (runs 1-18, COMPLETE)

- **Best AC-only**: run 16 — `lora_r=32, alpha=64, 300 steps, balanced
  250×6 cls, projector on, lr=2e-4` → 23.40% full_match (+2.6 vs 20.80
  baseline at 500-row eval).
- **Validated levers**: balanced training (+6pp swing vs class-collapse),
  projector unlock (+4.5pp vs frozen), max_new_tokens=384 (avoids
  truncation), 300 steps is the sweet spot (under-trains at 250, over-
  balances at 400).
- **Discarded**: schema-anchored Reason (no full-match gain), longer
  training at higher rank (click drift), seed variance is ±2pp at
  200-row eval → went to 500.
- **Plateau**: AC-only smoke caps near +2.6pp. Per-class wait is stuck
  at 0% (data has 7.4% wait but model never learns it) and AC entirely
  lacks `status` (terminal) — this matches the M3A AW baseline's
  dominant failure mode (67.9% max_steps_no_terminate).

### Phase 2 — AndroidLab integration (CURRENT, run 19+)

- Pulled THUDM Android-Lab Instruct dataset (Google Drive zip, 569MB).
- Wrote `convert_androidlab_som.py`: maps SoM trajectories →
  M3A-format smoke rows. 6053 converted, action mix:
  click 4318 / status 716 / input_text 513 / scroll 471 / nav_back 35.
- `prepare_smoke_data.py --androidlab-jsonl ... --include-status`
  builds 50/50 AC/AL mixed train per class (status is AL-only).
- Trainer/eval honor per-row `_image_root` so AC and AL images coexist.
- Run 19 (in progress): first AC+AL mixed run @ run-16 recipe. Goal:
  beat 23.40% AC-only ceiling.
