# Autoresearch Dashboard: pathZ-sft-smoke

**Runs:** 14 | **Kept:** 5 | **Discarded:** 9 | **Crashed:** 0
**Best:** full_match: **22.20% (#14, +1.40 vs segment-2 baseline)** ← validated lift

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
| 6 | e41a94c | **22.00% (+1.5)** | 50.00% | 98.50% | keep | 300-step BALANCED; first positive |
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
| 14 | 9051e7c | **22.20% (+1.4)** | 54.20% | 99.00% | **KEEP** | run-6 recipe at 500 rows; **lift confirmed** |

## Validated recipe (segment 2 best)

| Component | Value |
|---|---|
| Base | `unsloth/gemma-4-E2B-it`, 4-bit |
| LoRA | r=16, α=32, all-linear, vision+language |
| Projector | unlocked (modules_to_save=["embedding_projection"]) |
| Optimizer | adamw_8bit, lr=2e-4 cosine, warmup=15, wd=0.001 |
| Effective batch | 1 × 4 grad-accum |
| Schedule | 300 steps (~0.8 epoch on 1500-row balanced data) |
| Loss masking | train_on_responses_only = True |
| Data | balanced 250/cls × 6 cls = 1500 rows from AC-train |
| Reason format | natural-language one-liner, e.g. `Click element 5 ("Search").` |
| Eval | M3A-format prompt, max_new_tokens=384 |

## Per-action-type confirmed gains (run 14 vs baseline 13)

| action | baseline type | trained type | delta |
|---|---|---|---|
| open_app | 58.1% | 83.9% | **+25.8pp** |
| scroll | 13.6% | 37.3% | **+23.7pp** |
| input_text | 55.8% | 51.2% | -4.6 |
| navigate_back | 26.3% | 15.8% | -10.5 |
| click | 71.6% | 64.7% | -6.9 |
| wait | 2.4% | 0.0% | -2.4 |
