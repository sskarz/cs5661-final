# Autoresearch Dashboard: pathZ-sft-smoke

**Runs:** 20 | **Kept:** 8 | **Discarded:** 12 | **Crashed:** 0
**Best (segment 3, AW SR primary):** **20.0% (#19, +20pp vs baseline 0%)** — first positive live AW signal
**Best (segment 2, AC offline):** 23.40% (#16, +2.6 vs floor)

## Segment 0 (max_new=128 eval, 200-row eval)

| # | commit | full_match | type_match | parse_pct | status | description |
|---|--------|-----------|------------|-----------|--------|-------------|
| 1 | ce09f38 | 20.00% | 51.50% | 92.50% | keep | baseline @ max_new=128 |
| 2 | ce09f38 | 18.50% (-1.5) | 44.00% | 78.50% | discard | 200-step QLoRA, lr=2e-4, projector on; max_new=128 truncation |

## Segment 1 (max_new=384 eval, 200-row eval — noise floor ±2pp)

| # | commit | full_match | type_match | parse_pct | status | description |
|---|--------|-----------|------------|-----------|--------|-------------|
| 3 | 12e0490 | 20.50% | 54.50% | 100.00% | keep | baseline @ max_new=384 (segment floor) |
| 4 | 12e0490 | 18.50% (-2.0) | 54.50% | 98.50% | discard | 200-step natural dist; open_app/navigate_back regress |
| 5 | 12e0490 | 13.50% (-7.0) | 48.50% | 99.50% | discard | 500-step natural dist; class-collapse to click |
| 6 | e41a94c | 22.00% (+1.5) | 50.00% | 98.50% | keep | 300-step BALANCED; first positive |
| 7 | 9051e7c | 21.50% (-0.5) | 53.50% | 99.00% | discard | 500-step balanced; click-favoring shift |
| 8 | 9051e7c | 20.50% (-1.5) | 54.50% | 99.00% | discard | 300-step balanced @ lr=1e-4; under-trained |
| 9 | 9051e7c | 22.00% (=) | 57.00% | 98.00% | discard | + schema-anchored Reason; ties on full, type +7 |
| 10 | 9051e7c | 20.00% (-2.0) | 46.00% | 99.00% | discard | + decoupled open_app reason; input_text crashed |
| 11 | 9051e7c | 17.50% (-4.5) | 56.00% | 99.50% | discard | balanced + projector OFF; minorities regress |
| 12 | 9051e7c | 20.00% (-2.0) | 58.50% | 98.00% | discard | run-6 recipe + seed=2024; reveals ±2pp seed noise |

## Segment 2 (max_new=384 eval, 500-row eval — noise floor ±1.25pp)

| # | commit | full_match | type_match | parse_pct | status | description |
|---|--------|-----------|------------|-----------|--------|-------------|
| 13 | 9051e7c | 20.80% | 55.00% | 99.60% | keep | baseline @ 500-row eval (segment floor) |
| 14 | 9051e7c | 22.20% (+1.4) | 54.20% | 99.00% | keep | run-6 recipe at 500 rows; lift confirmed |
| 15 | dea04e6 | 21.80% (+1.0) | 57.40% | 100.00% | discard | 400/cls × 6 + 400 steps; over-balanced |
| 16 | dea04e6 | **23.40% (+2.6)** | 52.80% | 98.80% | **KEEP** | lora_r=32, alpha=64 + 300 steps; **NEW BEST** |
| 17 | b18d9a0 | 20.00% (-0.8) | 50.20% | 97.80% | discard | r=32 + 400 steps; click drift returns |
| 18 | 73a20ec | 21.20% (+0.4) | 54.20% | 98.80% | discard | r=32 + 250 steps post-AL-merge; under-trained by 50 steps |

## Best recipe (segment 2, run 16)

| Component | Value |
|---|---|
| Base | `unsloth/gemma-4-E2B-it`, 4-bit |
| LoRA | **r=32, α=64**, all-linear, vision+language |
| Projector | unlocked (modules_to_save=["embedding_projection"]) |
| Optimizer | adamw_8bit, lr=2e-4 cosine, warmup=12, wd=0.001 |
| Effective batch | 1 × 4 grad-accum |
| Schedule | 300 steps (~0.8 epoch on 1500-row balanced data) |
| Loss masking | train_on_responses_only = True |
| Data | balanced 250/cls × 6 cls = 1500 rows from AC-train |
| Reason format | natural-language one-liner |
| Eval | M3A-format prompt, max_new_tokens=384, 500 AC-val rows |

## Per-action-type, run 16 (best) vs baseline 13

| action | baseline type | run-16 type | delta |
|---|---|---|---|
| scroll | 13.6% | 47.0% | **+33.4pp** |
| open_app | 58.1% | 61.0% | +2.9 |
| navigate_back | 26.3% | 26.0% | ≈0 |
| click | 71.6% | 64.0% | -7.6 |
| input_text | 55.8% | 33.0% | -22.8 |
| wait | 2.4% | 0.0% | -2.4 |

## Segment 3 (live AW smoke slice — 10 curated tasks, primary metric: aw_success_rate)

| # | commit | aw_SR | aw_n_ok/total | ac_full | al_full | status | description |
|---|--------|-------|---------------|---------|---------|--------|-------------|
| 19 | 73a20ec | **20.00%** | 2/10 | 19.40 | 5.58 | **KEEP** | AC+AL mix (1000 AC + 500 AL); CameraTakePhoto + OpenAppTaskEval ✅ |
| 20 | 73a20ec | 0.00% | 0/10 | — | — | keep | M3A baseline floor — confirmed 0/10 |

**Key insight from run 19**: the AC+AL mix produces +20pp live AW lift even
though it REGRESSES on AC offline action-match (-1.4pp vs run 16) and
fails AL action-match (full=5.58, status type-match=0%). Offline action-match
on AC is a misleading proxy. AW success is the right north star.

## Phase 2: AndroidLab integration

Pivoted after run 18 — AC-only smoke had plateaued at +2.6pp on AC offline.
Added 6053 AndroidLab SoM trajectory rows, 716 `status` (AC has zero).
Run 19 mixed AC+AL @ 50/50 per overlapping class → first AW lift.
