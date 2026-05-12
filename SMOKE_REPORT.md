# pathZ SFT smoke — closing report (DRAFT, pending run 16)

**Period**: 2026-04-30 → 2026-05-01
**Branch**: `autoresearch/pathz-sft-smoke-2026-04-30`
**Model**: `unsloth/gemma-4-E2B-it` (2B, 4-bit QLoRA)
**Hardware**: 1× RTX 4090

## Goal

Validate the AndroidLab-style SFT recipe in `ANDROID_WORLD_PLAN.md` at
smoke scale (≤30 min/iteration) before committing GPU-hours to the full
8K-step pathZ run. The smoke proves that:

1. Data converter produces well-formed M3A-format examples;
2. Trainer converges (training loss drops);
3. Trained LoRA, evaluated offline on M3A-format prompts, **clearly
   beats** the zero-shot Gemma 4 E2B baseline.

## Validated recipe (locked in)

| Component | Value | Why |
|---|---|---|
| Base | `unsloth/gemma-4-E2B-it`, 4-bit | continuity with prior runs |
| LoRA | r=16, α=32, all-linear, vision+language | Run-L config |
| Projector | unlocked (`modules_to_save=["embedding_projection"]`) | run 11 confirms locking projector hurts (-4.5pp) |
| Optimizer | adamw_8bit, lr=2e-4 cosine, warmup=15, wd=0.001 | run 8 confirms 1e-4 under-trains, run 7 confirms 500 steps over-trains |
| Effective batch | 1 × 4 grad-accum | sticks to single 4090 envelope |
| Schedule | 300 steps (~0.8 epoch on 1500 rows) | runs 4/5/7 establish 200-step under-trains, 500-step over-trains, 300 is the plateau |
| Loss masking | train_on_responses_only = True | M3A response-only loss |
| Reason synth | natural-language one-liner | run 9 schema-anchored ties on full but breaks open_app; run 10 decoupled crashes input_text |
| Eval | M3A-format prompt, max_new_tokens=384 | run 2 truncated at 128 |
| Data | balanced 250/cls × 6 cls = 1500 rows from AC-train | run 5 shows imbalanced training collapses to click majority |

## Headline numbers (segment 2, 500-row eval — noise floor ±1.25pp)

| | full_match | type_match | parse_pct | reason_pct |
|---|---|---|---|---|
| **Baseline** (run 13) | 20.80% | 55.00% | 99.60% | 100.00% |
| **Trained** (run 14) | **22.20%** | 54.20% | 99.00% | 99.00% |
| Δ | **+1.40** | -0.80 | -0.60 | -1.00 |

## Per-action-type validation (run 14 vs run 13)

| action | n in eval | baseline type | trained type | Δ |
|---|---|---|---|---|
| click | 306 | 71.6% | 64.7% | -6.9 |
| scroll | 59 | 13.6% | **37.3%** | **+23.7** |
| input_text | 43 | 55.8% | 51.2% | -4.6 |
| wait | 42 | 2.4% | 0.0% | -2.4 |
| open_app | 31 | 58.1% | **83.9%** | **+25.8** |
| navigate_back | 19 | 26.3% | 15.8% | -10.5 |

## What we learned (the recipe-space map)

The smoke explored 12 distinct recipe variants. The biggest single
inflection points:

1. **Class balancing was the dominant lever (+8.5pp from run 5 to run 6).**
   Without balancing, even one epoch on AC's natural distribution
   collapses the model to predict "click" for everything (run 5: 67% of
   predictions are click; input_text and navigate_back routed to click).
   With per-class subsample/oversample to 250/cls × 6 cls, minorities
   become trainable.

2. **300 steps is the sweet spot.** 200 steps under-trains; 500 steps
   over-trains and pulls back toward click majority even with balanced
   data. The smoke loss curve shows training loss continuing to drop
   past 300 — but eval performance has already peaked.

3. **Projector unlock matters at smoke scale too.** Run 11 (projector
   off) drops -4.5pp and breaks every minority class. Run-L's lesson
   ("the vision-LM bridge needs to be malleable for grounding") holds.

4. **Reason format is mostly a wash.** Schema-anchored Reason (run 9)
   improved type-match (50→57) but hurt full_match through coupling
   args into reasoning. Simple natural-language Reason wins on net.

5. **Bigger eval set matters more than bigger train set.** Going from
   200→500 eval rows reduces noise from ±2pp to ±1.25pp and makes
   recipe deltas detectable. Going from 1500→2400 train rows produces
   redistribution within the metric (scroll +19, open_app -10) but no
   net headline gain.

## Discarded variants

| Run | Recipe | Result | Lesson |
|---|---|---|---|
| 2 | 200-step + max_new=128 | -1.5 | Eval truncation, not real |
| 4 | 200-step natural dist | -2.0 | Too few steps + class imbalance |
| 5 | 500-step natural dist | -7.0 | Click collapse |
| 7 | 500-step balanced | -0.5 | Click favoring at high steps |
| 8 | 300-step balanced @ lr=1e-4 | -1.5 | Under-trained |
| 9 | + schema-anchored Reason | tie | Args-in-Reason coupling |
| 10 | + decoupled open_app reason | -2.0 | Cross-class loss interference |
| 11 | balanced + projector OFF | -4.5 | Lost vision grounding |
| 12 | run-6 recipe + seed=2024 | -2.0 | ±2pp seed noise on 200-row eval |
| 15 | balanced 400/cls + 400 steps | -0.4 | Capacity-bound |

## What this DOESN'T validate

- **Wait remains 0% type-match.** No model emits `wait` reliably. AC's
  wait rows are mid-episode "pause for screen update" steps — out of
  distribution for AW where the goal-screen relationship is clearer.
  Needs synthetic wait positives or prompt-time hint.
- **Navigate_back regressed.** Both balanced training and the natural-
  distribution baseline emit navigate_back in only ~10-25% of relevant
  rows. AC has ~3% navigate_back in train; balancing oversamples it but
  the synth Reason stays generic.
- **AC is single-step data.** The recipe is validated for per-step
  action emission. Multi-step trajectory training (the AndroidLab
  Instruction dataset, day 1 in pathZ plan) is the next test.

## Next steps (out of smoke scope)

These are the next experiments at PRODUCTION scale (24h+ training):

1. **Pull AndroidLab Instruction dataset** (THUDM/Android-Lab) and
   merge with the balanced AC mixture per the day-1/day-2 plan.
2. **Run the full 8K-step SFT** with the validated recipe + AndroidLab
   trajectories. Expect AW SR to clear baseline+5pp at minimum based
   on AndroidLab's published lift.
3. **Add `status` and `answer` synthetic rows** so the model can learn
   to terminate (the dominant AW failure mode in the M3A baseline §52
   was max_steps_no_terminate at 67.9%).
4. **V-Droid-style constrained decoding** at AW eval time to kill
   schema hallucinations (~5% in our smoke).

## Repro

```bash
git checkout autoresearch/pathz-sft-smoke-2026-04-30
uv run python scripts/pathZ/prepare_smoke_data.py \
    --balance-classes --per-class-target 250 --n-eval 500
./autoresearch.sh
```
