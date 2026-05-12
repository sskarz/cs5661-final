# Autoresearch Dashboard: pathZ-sft-smoke

**Runs:** 44 | **Kept:** 12 | **Discarded:** 32 | **Crashed:** 0
**Best (recipe): r43 — E4B + plan distillation + drop status, 10% AW-20 with NEW task solved (CameraTakeVideo, first time across r19-r44)**.
**True base capability**: 10% AW-20 SR remains the ceiling. Across r35→r44 trying data fix, harness changes, bigger model, plan distillation, status drop, and no-op removal, AW-20 has not exceeded 2/20. Trajectory analysis (r43): real failure modes are model grounding errors (wrong app selection, out-of-range indices, loop-clicking) — model-capacity issues, not harness. Next: r42-style teacher rollouts on AL tasks for higher-quality on-distribution multi-step training data.
**Best (segment 3, AW SR primary):** **50.00% (#22)** — but 4-sample variance study (r22/r26/r28/r29: 50/10/20/0) shows recipe true mean ≈20%, σ≈21pp; r22 was upward outlier
**Latest 20-task slice (pure-a11y stack):** r31=10%, r32=5%, r33=10% — all 3 land within σ≈7pp.
**Best (segment 2, AC offline):** 23.40% (#16, +2.6 vs floor)

## ⚠️ r34 finding: AC training data is 96% off-distribution from AW

Teacher distillation (Gemma 4 31B 4-bit) on 1500 harness-parity AC rows surfaced a structural data problem that invalidates not just r34 but the assumed root cause for r19-r33 plateau:

- **240/250 AC `open_app` rows target apps that DO NOT EXIST in the AW emulator** (Amazon, Khan Academy, Maps, Drive, Gmail, eBay, Booking.com, ZARA, Decathlon, Vimeo, Edmunds, Myntra, …). AW's harness inventory is 19 apps (Files/Markor/Joplin/Broccoli/Pro Expense/Camera/Clock/Simple Calendar Pro/Simple SMS Messenger/Contacts/Audio Recorder/Chrome/Settings/OsmAnd/Retro Music/Simple Draw Pro/Tasks/VLC/OpenTracks). Only 4% of AC `open_app` rows match.
- **281/329 erroneous teacher status emissions were correct infeasibility refusals** — the teacher correctly identified the missing-from-inventory apps. Discarding those as "mismatches" would have forced the student to learn rationalizations of actions even the 31B model refused.
- **Teacher per-class match: click 34.5%, scroll 11.2%, navigate_back 2.4%, open_app 10.0%, input_text 0%, wait 3.4%** — teacher fails on the same classes as student, for the same reason (text-only loses spatial / state / goal-substep signal).

**Implication**: every r19-r33 model was trained to call non-existent apps. r35+ pivots to filtering AC to AW-compatible rows and re-including the 6053 AndroidLab rows (which were collected on the AW emulator and are naturally on-distribution).

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
| 19 | 2590f55 | 20.00% | 2/10 | 19.40 | 5.58 | keep | AC+AL mix (1000 AC + 500 AL); CameraTakePhoto + OpenAppTaskEval ✅ |
| 20 | 2590f55 | 0.00% | 0/10 | — | — | keep | M3A baseline floor — confirmed 0/10 |
| 21 | 2590f55 | 10.00% (-10) | 1/10 | 16.00 | 5.98 | discard | + status class (250 rows from AL); status type-match still 0% |
| 22 | 057dd31 | **50.00%** (+30) | 5/10 | 19.60 | 1.99 | **KEEP** | r19 mix + 400 steps; ClockStopWatchRunning + RecipeDeleteSingleRecipe new wins |
| 23 | 057dd31 | 30.00% (-20) | 3/10 | 21.20 | 3.59 | discard | r22 + 500 steps; over-trained |
| 24 | 057dd31 | 30.00% (-20) | 3/10 | 18.40 | 3.59 | discard | r22 + 350/cls (2100 rows, 0.76 epoch); under-trained |
| 25 | 057dd31 | 20.00% (-30) | 2/10 | 17.20 | 3.19 | discard | r22 + 350/cls + 525 steps (1.0 epoch); lost both Clock tasks |
| 26 | 057dd31 | 10.00% (-40) | 1/10 | 16.20 | 4.38 | discard | r22 verbatim + seed=2024; **REVEALS HUGE SEED VARIANCE** |
| 27 | 057dd31 | 10.00% (-40) | 1/10 | 20.00 | 5.98 | discard | PURE-AL ablation; killed OpenAppTaskEval — AC mixing load-bearing |
| 28 | 8a3e4b8 | 20.00% (-30) | 2/10 | 21.80 | 5.58 | discard | r22 verbatim + seed=4242 (3rd sample); confirms r22 was outlier |
| 34 | b096ce4 | n/a (no train) | n/a | n/a | n/a | discard | Teacher-distillation analysis: 96% of AC open_app rows target non-AW apps; teacher's "mismatches" are mostly correct infeasibility refusals. Distillation premise invalidated; pivoting to data filter (r35) |
| 35 | 56002a5 | 10.00% (=) | 2/20 | 12.00 | 1.20 | discard | AW-distribution data rebuild (--aw-apps-only, --synthesize-open-app 250, AL re-included). 1750 balanced rows. **IDENTICAL tasks succeeded as r31/r33** (Clock+OpenApp). M3AA11Y eval prompt already prepends AW app inventory → contamination wasn't load-bearing for AW SR even though structurally real. Bottleneck is harness context, not data distribution. Pivoting r36 to reason-preserving history |
| 36 | b4d87b8 | 5.00% (-5) | 1/20 | =r35 | =r35 | discard | Reason-preserving history (HARNESS-only, r35 adapter unchanged). Append model prior `Reason:` text on `  reason:` line under each `Step N:` summary. **LOST ClockStopWatchRunning** (succeeded r31/r33/r35). Net-negative: prompt distribution shift dilutes attention. To unlock the harness-context lever properly, train-time format must change in lockstep. Pivoting r37 to candidate-action shortlist header (additive, no history restructure) |
| 37 | 328c176 | 5.00% (=r36) | 1/20 | 11.00 | 1.20 | discard | r37 = r36 + matching train. prepare_smoke_data --history-reasons emits same `  reason:` line under each `Step N:` in train prompts. Aligns train+eval distribution. **IDENTICAL FAILURE as r36** — RULES OUT distribution shift hypothesis. Real cause: model uses prior reasons as distraction signal, can't disambiguate own-prior-reasoning from user goal, gets stuck looping (max-steps on stopwatch). At 2B+text-only scale, in-context reasoning has a sharp toxicity cliff. Pivoting r38 to compact a11y tree (drop empty-label elements) — prompt-budget hypothesis without touching reasoning |
| 38 | 11c0a21 | 5.00% (=r36) | 1/20 | 10.00 | 4.38 | discard | History cap at 3 (eval-only, matches train's prior_strs[-3:]). 3rd consecutive harness change to land at 5% (lost ClockStopWatchRunning). **TRUE FLOOR REVEALED: 5% (1 robust task = OpenAppTaskEval)**. r35's 10% was 1 robust + 1 fragile (Clock brittle to ANY history-block change). Structural ceiling for Gemma 4 E2B + SFT-only + text-only. r39 needs to step up to bigger base model OR re-enable vision OR step up multi-step training data volume |
| 39 | 50188ff | 10.00% (=r35) | 2/20 | 13.50 | 1.59 | discard (metric); KEPT (fix) | Discovered + fixed pre-existing bug in render_m3a_prompt (joined element lines with empty string instead of newline). Train/eval format mismatch present since r33. Bug fix recovers Clock+OpenApp baseline; doesn't lift past r35 ceiling. Bug fix is keeper-tier infra; metric is discard. r36/r37/r38's 5% was driven by harness changes (not data bugs); r39 disambiguated. Pivoting r40 to Gemma 4 E4B 4-bit base model upgrade |

**Key insight from run 19**: the AC+AL mix produces +20pp live AW lift even
though it REGRESSES on AC offline action-match (-1.4pp vs run 16) and
fails AL action-match (full=5.58, status type-match=0%). Offline action-match
on AC is a misleading proxy. AW success is the right north star.

**Key insight from runs 26+28**: r22's 50% AW was a **positive outlier** of a
high-variance distribution. Three samples of the *exact same recipe* with
different seeds yield 50/10/20 → mean=26.7%, σ≈21pp. The 10-task slice is
too small to distinguish 25% from 35% reliably. **Variance reduction
(20-task slice) is the next priority before further recipe tuning.**

**Key insight from run 27**: pure-AL training drops AW to 10% by losing
the AC-only `open_app` action class. AC's open_app coverage (61% type-match
in r22 vs 0% in r27) is required for the AW slice tasks that begin with
"Open the X app". AndroidLab alone can't replace it.

## Phase 2: AndroidLab integration

Pivoted after run 18 — AC-only smoke had plateaued at +2.6pp on AC offline.
Added 6053 AndroidLab SoM trajectory rows, 716 `status` (AC has zero).
Run 19 mixed AC+AL @ 50/50 per overlapping class → first AW lift.
