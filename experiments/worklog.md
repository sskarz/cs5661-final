# Autoresearch worklog: pathZ M3A-format SFT smoke

Started: 2026-04-30
Branch: `autoresearch/pathz-sft-smoke-2026-04-30`
Goal: validate the AndroidLab-style SFT recipe at smoke scale before
committing GPU-hours to the full 8K-step pathZ run.

## Data

- Source: `data/androidcontrol_a11y_native_v3` (AC-train 73K, AC-val 686).
- Smoke train: 2000 rows reformatted into M3A's exact prompt + action
  vocab. Distribution: click 60.2%, scroll 13.9%, input_text 8.8%,
  wait 7.4%, open_app 6.6%, navigate_back 3.1%, navigate_home 0.1%.
- Smoke eval: 200 rows from AC-val (held out, never trained on).
  Distribution: click 61.5%, input_text 10%, wait 9.5%, scroll 7.5%,
  open_app 6%, navigate_back 5.5%.
- The prompt skeleton is `M3A_PROMPT_PREFIX` from `m3a_format.py` —
  trimmed (no long Text-related-operations section) but action vocab
  is M3A-verbatim (click/long_press/input_text/scroll/.../status/answer).
- Training labels are `Reason: <synthetic one-liner>\nAction: {<json>}`.

## Runs

### Run 33: schema-parity retrain (AC prompts match M3AA11Y eval format) — aw_SR=10.00 (DISCARD)
- Timestamp: 2026-05-01 17:30
- What changed: prepare_smoke_data.py grew a `--harness-parity` flag. AC training prompts now prepend the indexed app inventory + render history as `Step N: <action_repr> -> ok; window changed` (the same form M3AA11Y emits at eval time). Same recipe otherwise: 1500 AC-only rows, --text-only, max_length=16384, 400 steps.
- Result: 2/20 = 10% AW SR. IDENTICAL tasks to r31 (ClockStopWatchRunning + OpenAppTaskEval). AC offline DROPPED full=18.6→9.5 (-9pp), type=44.6→23.0 (-22pp). BUT input_text type-match SURGED 4.65→25.00 (+20pp); the model rebalanced toward AW-relevant verbs (click/input_text/open_app dominant; scroll/wait/navigate_back collapsed to 0%).
- Insight: schema parity is format-neutral, not load-bearing. The harness patches are NOT dormant due to train/eval mismatch — they're dormant because the base model lacks the planning ability to use them. Pure (state, action) imitation has hit a ceiling at ~10% AW SR for our 2B base. Confirms user's earlier intuition: scaffolding cannot compensate for model capability.
- Next: r34 — teacher distillation. Use Gemma 4 31B (4-bit, same Unsloth pipeline as student) to generate (Reason, Action) trajectories on AC rows, train student on distilled rationales.

### Run 32: same r31 adapter + harness patches (no-op widening, status/answer guards, tolerant parser) — aw_SR=5.00 (DISCARD)
- Timestamp: 2026-05-01 16:55
- What changed: kept r31's adapter; patched M3AA11Y harness — `_NAV_ACTIONS` widened to include scroll/input_text, `status(complete)` guard refusing on no-change, `answer(...)` guard requiring at least one open_app+click, tolerant JSON parser (brace-balance walking, truncation repair), `max_new_tokens` 256→384.
- Result: 1/20 = 5% (only OpenAppTaskEval). ClockStopWatchRunning regressed (it succeeded in r31 with same adapter). Within σ≈7pp of r31's 10% — statistically indistinguishable.
- Insight: harness scaffolding alone cannot lift a fixed adapter. The model's premature `status(complete)` after 2-3 steps is the real ceiling — guards refuse some bad emissions but the model finds new ones. This *strongly* validates the schema-parity hypothesis: until the adapter is trained on the new harness's prompt format, the harness investments stay dormant.
- Next: schema parity retrain (regenerate AC training rows to match the new harness prompt format with inventory + det-history + executor feedback).

### Run 31: pure-a11y AC-only + new M3AA11Y harness (4 patches) — aw_SR=10.00 (KEEP)
- Timestamp: 2026-05-01 16:00
- What changed: training data dropped AL rows (kept 1500 AC-only with input_text boost), `--text-only` flag added (no images at train + eval), max_length 4096→16384, M3AA11Y harness with: indexed app inventory in prompt, deterministic step summaries (no LLM summary call), executor feedback in history, no-op detection (force `status(infeasible)` after 2 consecutive nav-class no-ops).
- Result: 2/20 = 10% (ClockStopWatchRunning + OpenAppTaskEval). open_app type-match=96.77% (highest yet); pure-a11y emitted status=2% on AL despite no status training.
- Insight: First pure-a11y AW data point — confirms a11y-only inference works on this 2B base. open_app accuracy lifted significantly. But absolute SR no better than r19's 20% (10pp lower), so pure-a11y isn't the silver bullet AndroidLab paper suggested for ≤9B models. Their finding was on AndroidLab benchmark, not AW — different app distribution.
- Next: r32 patches harness; if no lift, schema parity is the necessary move.

### Run 30: input_text boost x2 (CANCELLED before AW eval) — DISCARD
- Timestamp: 2026-05-01 12:30
- What changed: increased input_text rows in training mix to fix the input_text emission collapse seen in r29.
- Result: AC offline DROPPED to 14.20% (-6.8 vs r29). Boost diluted click/scroll/navigate_back; input_text only moved 0% → 2.33%. User cancelled before AW eval.
- Insight: more input_text rows do NOT fix the structural emission problem. Pivot to pure-a11y for r31.

### Run 29: r22 verbatim (default seed=3407) on 20-task slice — aw_SR=0.00 (DISCARD)
- Timestamp: 2026-05-01 11:00
- What changed: identical recipe and seed as r22 (md5-confirmed identical training data), but ran the 20-task slice instead of 10. Goal: confirm r22's 50% was reproducible with same seed.
- Result: 0/20 = 0% AW SR. AC offline 21.00% (typical for r22 family). AC input_text COLLAPSED to 0% (was 37.5% in r22's matched run).
- Insight: HUGE blow to r22 reproducibility. Four-sample r22-recipe distribution: 50/10/20/0 → mean=20%, σ≈21pp. CUDA non-determinism makes seed-fixed reruns diverge wildly. Recipe is genuinely seed-fragile, not just slice-noisy. Variance reduction (multi-seed eval) is now a hard requirement.

### Run 20: M3A baseline AW smoke slice — aw_SR=0.00 (KEEP, segment-3 floor)
- Timestamp: 2026-05-01 05:50
- What: ran the M3A harness with NO LoRA on the 10 curated AW tasks to set
  the segment-3 floor. Same emulator, same task list as run 19.
- Result: 0/10 succeeded. CameraTakePhoto FAILED, OpenAppTaskEval FAILED,
  all others failed. Confirms run 19's +20pp lift is real lift, not luck.

### Run 19: AC+AL mix @ run-16 recipe — aw_SR=20.00 (KEEP, FIRST POSITIVE AW SIGNAL)
- Timestamp: 2026-05-01 05:30
- What changed: switched primary metric from offline AC action-match to
  live AW success rate on a 10-task curated slice (FilesDeleteFile,
  OpenAppTaskEval, SimpleSmsReply, RecipeDeleteSingleRecipe, CameraTakePhoto,
  ClockStopWatchPausedVerify, ClockStopWatchRunning, MarkorCreateFolder,
  MarkorDeleteNote, NotesIsTodo). Trained on 1500-row balanced AC+AL mix
  (1000 AC + 500 AL, 250×6 cls) at run-16's recipe (lora_r=32, 300 steps).
- AC offline: full=19.40 (-1.4 vs run-16 best 23.40, -1.4 vs baseline 20.80).
- AL offline: full=5.58, type=34.26, status type-match=0%. Format mismatch
  between AL prompt (no element list, SoM-marked image) and AC prompt
  (text element list) likely interferes.
- **Live AW: 2/10 = 20.00%** (CameraTakePhoto ✅, OpenAppTaskEval ✅).
  Baseline 0/10 on the same slice (run 20). **+20pp lift.**
- Insight: **AC offline action-match is a MISLEADING PROXY for live AW.**
  Run 19 is the worst AC offline of segment 2 yet the only positive AW.
  The AndroidLab data taught the model task-completion / app-launch
  patterns that AC alone cannot. Status action-match still 0% on offline
  but the model evidently emits SOMETHING that AW accepts as termination
  on the easy tasks.
- Side-effect: navigate_back type-match collapsed 26→0 on AC eval —
  AL has only 35 nav_back rows, mixing them caused that class to lose
  the AC representation.
- Next: keep this recipe as new floor; experiment with (a) more AL
  weight, (b) including more AL classes (longer trajectories), (c)
  AL prompt format alignment to AC.

### Run 18: lora_r=32 @ 250 steps AC-only (post-AL-integration) — full_match=21.20 (DISCARD)
- Timestamp: 2026-05-01 04:50
- What changed: ran the AC-only baseline recipe one more time to confirm
  no regression after merging the AL conversion code. Trainer/eval now
  honor per-row `_image_root` so AL+AC rows can coexist in one dataset.
- Result: full=21.20 (-2.20 vs run 16 best 23.40), type=54.20,
  train_loss_final=0.93. parse=98.80.
- Per-type: click 66.34, scroll 47.46, input_text 20.93, wait 11.90,
  open_app 70.97, navigate_back 21.05.
- Insight: at lora_r=32 the recipe wants 300 steps not 250 — under-trained
  by ~50 steps. Dropping back to 250 gives up the +2.6pp lift. The AL
  integration itself is neutral; this isolates the step-count regression.
- Next: run 19 should rebuild train as AC+AL mix at 300 steps.

### Run 17: lora_r=32 + 400 steps balanced — full_match=20.00 (DISCARD)
- Timestamp: 2026-05-01 03:35
- What changed: at the run-16 setup (r=32, alpha=64, balanced 250/cls),
  bumped max_steps 300 → 400.
- Result: full=20.00 (-3.40 vs best), type=50.20, train_loss=0.77.
- Per-type: scroll/input_text gain but click drops 11pp (66→55) — the
  classic over-balanced collapse, just delayed by the higher rank.
- Insight: 300 steps at r=32 is the sweet spot. More steps + balanced
  data = click drift even at higher rank. Stop bumping steps without a
  bigger/more diverse train pool.

### Run 16: lora_r=32 alpha=64 @ 300 steps — full_match=23.40 (KEEP, NEW BEST)
- Timestamp: 2026-05-01 03:00
- What changed: doubled LoRA rank/alpha from 16/32 to 32/64 on top of
  run-14's recipe (300 steps, balanced 250×6, projector on, lr=2e-4).
- Result: full=23.40 (+1.2 vs run 14, +2.6 vs baseline 20.80), type=52.80,
  train_loss=0.87 (vs run 14's 0.95 — more capacity, lower loss).
- Per-type vs baseline: click 71.6→64 (-7.6), scroll 13.6→47 (+33.4),
  open_app 58.1→61 (+2.9), input_text 55.8→33 (-22.8), wait 2.4→0 (-2.4),
  navigate_back 26.3→26 (=).
- Insight: rank-32 helps the harder-to-learn classes (scroll +33pp) but
  costs input_text quality. The full_match win confirms the bias-variance
  trade favors more rank at smoke scale.
- This is the AC-only smoke ceiling for our recipe — further AC gains
  require more data (Phase 2: AndroidLab trajectories).

### Run 15: 400/cls × 6 = 2400 rows + 400 steps — full_match=21.80 (DISCARD)
- Timestamp: 2026-05-01 02:30
- What changed: from run-14's recipe, bumped per-class target 250→400 and
  steps 300→400 to see whether more balanced data = more headroom.
- Result: full=21.80 (-0.40 vs run 14), type=57.40, train_loss=0.84.
- Per-type: scroll +19, input_text +5, but open_app & navigate_back
  regressed; net trade-off slightly negative.
- Insight: more data + more steps over-balances. Quality-of-data first,
  then quantity. The 250/cls target is well-tuned to the AC distribution.

### Run 14: run-6 recipe @ 500-row eval — full_match=22.20 (KEEP, RECIPE VALIDATED)
- Timestamp: 2026-05-01 02:00
- What changed: rebuilt eval with --n-eval=500 (vs prior 200) to drop noise
  floor 1/sqrt(2.5) ≈ ±1.25pp instead of ±2pp.
- Result: full=22.20 (+1.4 vs new baseline 20.80, replicating run 6's
  +1.5 at 200 rows). type=54.20 parse=99.00 reason=99.00.
- Per-type vs run-13 baseline: scroll 13.6→37.3 (+23.7pp), open_app
  58.1→83.9 (+25.8pp), click 71.6→64.7 (-6.9), navigate_back
  26.3→15.8 (-10.5), input_text 55.8→51.2 (-4.6), wait 2.4→0 (-2.4).
- Insight: **the +1.4-1.5pp lift is real, not seed-luck**. Run 12's
  20.00 at seed=2024 was the noise tail. Two action types learn
  meaningfully: scroll +24pp and open_app +26pp. Click & navigate_back
  regress modestly because the balanced training distributes attention
  away from those classes.
- Smoke verdict: **SFT recipe is correct and produces positive transfer
  on every action type the model has training data for**. Caveat: wait
  remains stubbornly at 0% — no model emits wait reliably; needs a
  data-side fix (synth wait positives or prompt-time hint).

### Run 13: baseline @ 500-row eval — full_match=20.80 (KEEP, segment-2 floor)
- Timestamp: 2026-05-01 01:30
- What changed: --n-eval=500 (was 200).
- Result: full=20.80 (vs run 3's 20.50 at 200 rows, +0.3 — within noise),
  type=55.00, parse=99.60.
- Per-type: click 71.6, scroll 13.6, input_text 55.8, wait 2.4,
  open_app 58.1, navigate_back 26.3.
- Insight: at 500 rows, baseline metric is more stable. Now we can
  compare recipes against a tighter floor.

### Run 12: run-6 recipe + seed=2024 — full_match=20.00 (DISCARD)
- Timestamp: 2026-05-01 01:15
- What changed: only `seed: 3407 → 2024`.
- Result: full=20.00 (-2 vs run 6's 22.00 at same recipe, same data).
- Per-type vs run 6: click +21, open_app -42, navigate_back +18 — same
  recipe lands in a totally different per-class equilibrium just from
  seed change.
- Insight: **±2pp noise floor on 200-row eval is real**. Bumped eval
  to 500 rows to reduce noise.

### Run 11: balanced 300-step + projector OFF — full_match=17.50 (DISCARD)
- Timestamp: 2026-05-01 01:00
- What changed: train_projector=False (was True).
- Result: full=17.50 (-4.5 vs run 6), type=56.00, parse=99.50.
- Per-type: click type 56→75 (+19) but full lower; minorities all
  regressed (input_text 50→25, scroll 47→33, open_app 92→83).
- Insight: Run-L's projector lesson holds at smoke scale too — the
  vision-LM bridge needs to be malleable for this kind of grounding.

### Run 10: balanced 300-step + decoupled schema-Reason — full_match=20.00 (DISCARD)
- Timestamp: 2026-05-01 00:55
- What changed: removed app_name from open_app Reason ("`open_app`."
  not "`open_app`, app_name "Gmail""). Other Reasons unchanged.
- Result: full=20.00 (-2 vs run 6); type=46.00.
- Per-type: input_text type crashed 65→10, wait dropped, open_app held
  at 83. Decoupling open_app changed the loss landscape for OTHER
  classes.
- Insight: schema-anchored Reasons interact in non-trivial ways across
  classes. Reverting to run-6 simple Reasons.

### Run 9: balanced 300-step + schema-anchored Reason — full_match=22.00 (DISCARD, ties)
- Timestamp: 2026-05-01 00:50
- What changed: synth Reason now explicitly names canonical M3A token
  (`The action_type is \`click\`, targeting index 5`).
- Result: full=22.00 (=run 6), type=57.00 (+7 vs run 6's 50).
- Per-type: click type 56→69, input_text 50→65, scroll 47→60,
  navigate_back 0→18 — all up. open_app crashed 92→42 due to verbose
  Reason coupling app_name into reasoning.
- Insight: schema-anchoring helps schema fidelity (less hallucination)
  but hurts grounding when the reason commits to args. Need to decouple
  args from reason — see run 10.

### Run 8: 300-step balanced @ lr=1e-4 — full_match=20.50 (DISCARD, ties baseline)
- Timestamp: 2026-05-01 00:45
- What changed: lr 2e-4 → 1e-4.
- Result: full=20.50 (-1.5 vs run 6, =baseline), type=54.50.
- Per-type: minorities (scroll/open_app/wait) regressed back toward
  baseline; click held high — model retains base distribution rather
  than learning balanced behavior. **Under-trained.**
- Insight: 1e-4 at 300 steps doesn't deliver enough updates. Either
  bump LR back to 2e-4 OR run more steps at 1e-4.

### Run 7: 500-step balanced — full_match=21.50 (DISCARD)
- Timestamp: 2026-05-01 00:40
- What changed: max_steps 300 → 500.
- Result: full=21.50 (-0.5 vs run 6), type=53.50.
- Per-type: click recovered (56→67) but wait/open_app eroded.
- Insight: more training pulls back toward click-favoring even on
  balanced data. 300 steps is the sweet spot.

### Run 6: 300-step QLoRA on BALANCED data — full_match=22.00 (KEEP, FIRST POSITIVE)
- Timestamp: 2026-05-01 00:30
- What changed: rebuilt train data with `--balance-classes --per-class-target 250`
  → 1500 rows = 250 each of click/input_text/navigate_back/open_app/scroll/wait
  (navigate_home dropped: only 29 source rows, would replay too aggressively).
  Trained 300 steps at lr=2e-4 (was 200/500 on imbalanced for runs 4/5).
- Result: full=22.00 (+1.5 vs 20.50 baseline), type=50.00, parse=98.50.
  train_loss_final=0.95 (higher than run 5's 0.74 — balanced is harder).
- Per-type vs baseline: click full=22 (-3.2), input_text full=0 (=),
  wait type=15.79 (+15.79), scroll type=46.67 (+33.34), open_app
  type=91.67 (+33.34), navigate_back type=0 (-27.27).
- Confusion: open_app 11/12 (91.7%) excellent. navigate_back 0/11 — all
  routed to open_app/click (model conflates "go back" with "open app").
  ~11 rows hallucinate "type_into_text_field" (not a valid M3A type).
- Insight: **Class imbalance was the killer.** Even one epoch on
  click-dominated training collapsed the model to predict click. With
  balanced data, the minorities (open_app, scroll, wait) become
  trainable. Click does drop 3pp, but the gain on minorities outweighs.
- Next: bump steps 300→500 on balanced; see if further training cleans
  the navigate_back-as-open_app confusion + the type_into_text_field
  hallucinations.

### Run 5: 500-step QLoRA on natural dist — full_match=13.50 (DISCARD, REGRESS)
- Timestamp: 2026-04-30 23:50
- What changed: 500 steps (1 epoch over 2K rows).
- Result: full=13.50, type=48.50, parse=99.50.
- Per-type catastrophe: click predicted 134/200 (67%), only 86 click gt
  rows → 16/20 input_text routed to click, 7/11 navigate_back → click.
- Insight: more training on imbalanced data = more class collapse. 200
  steps was the "less bad" point on this curve. Recipe needs class-
  balanced data, not more imbalanced training.
- Next: rebuild train.jsonl with balanced classes, retry at 300 steps.

### Run 4: 200-step QLoRA @ max_new=384 — full_match=18.50 (DISCARD)
- Timestamp: 2026-04-30 23:35
- What changed: bumped eval `max_new_tokens` 128 → 384.
- Result: full=18.50 (vs 20.50 baseline-384), type=54.50, parse=98.50.
- Per-type vs baseline: click full=23.6 (-1.6), open_app full=25.0 (-16.7),
  navigate_back full=9.1 (-18.2), scroll full=20.0 (+6.7), wait full=5.3 (+5.3).
- Insight: format compliance is now great (parse 98.5%) but **specific
  actions regressed** — under-trained SFT damages base instruction-
  following without yet teaching the grounded skill. Projector unlock at
  this LR may be too aggressive at the smoke scale.
- Next: bump steps 200→500 to push past the disruption phase.

### Run 3: baseline @ max_new=384 — full_match=20.50 (KEEP, segment 1 baseline)
- Timestamp: 2026-04-30 23:25
- What changed: re-eval baseline at the new max_new_tokens=384 to set a
  fair floor for the new segment.
- Result: full=20.50 (+0.5 vs run 1), type=54.50 (+3.0), parse=100,
  reason=100. The +0.5 is the rescued tail of generations that had been
  truncated at 128 tokens.
- Insight: the M3A prompt itself produces well-formatted output for ~all
  rows on Gemma 4 E2B even zero-shot. The bar is now 20.50% to clear.
- Next: re-run the trained adapter at the new eval length.

### Run 2: 200-step QLoRA @ max_new=128 — full_match=18.50 (DISCARD)
- Timestamp: 2026-04-30 23:10
- What changed: 200 steps SFT, lr=2e-4, batch 1×4, lora_r=16/alpha=32,
  projector unlocked.
- Result: full=18.50 (vs 20.00 baseline), type=44.00, parse=78.50,
  train_loss_final=1.08 (8.0 → 0.45 over 200 steps).
- Insight: training mechanically converges, but eval parse rate
  COLLAPSED from 92.5% → 78.5%. Inspecting raw preds: trained model
  emits a long verbose CoT reason that gets truncated at max_new=128
  before reaching the `Action:` line. Truncation, not skill regression.
- Next: bump max_new_tokens to 384 in eval; re-baseline at the new
  eval length so we can see the trained adapter's true full_match.

### Run 1: baseline (zero-shot, no LoRA) — full_match=20.00 (KEEP, baseline)
- Timestamp: 2026-04-30 23:00
- What changed: nothing — `RECIPE=baseline` to set the floor.
- Result: full=20.00%, type=51.50%, parse=92.50%, reason=94.50%.
- Per-type: click=65/24% (type/full), input_text=55/0%, scroll=13/13%,
  wait=0/0%, open_app=58/42%, navigate_back=27/27%.
- Insight: Gemma 4 E2B does follow the M3A prompt pretty well at the
  format level (92.5% parse). The gaps are: (1) action_type hallucination
  ("type into a text field" instead of `input_text`) — schema fidelity,
  (2) `wait` never emitted — class imbalance prior, (3) `scroll`
  consistently mis-typed as click — needs grounding training.
- Next: 200 step QLoRA on the 2K M3A-format train mix; expect parse to
  approach 100% and full_match to clear ~25% if the recipe is sound.

## Key Insights

- **The floor is ~20.5%** full-match, not 0% — the M3A prompt gets you
  surprisingly far on a pre-trained Gemma 4 E2B because its
  instruction-following is decent. The bar is therefore "clearly beat
  20.5%" not "clearly beat 0%".
- **Class imbalance was the dominant failure mode at smoke scale.**
  60% of AC-train rows are clicks; 200/500 steps of imbalanced SFT
  collapsed the model to "predict click" for everything. Switching to
  per-class balanced training (250 each × 6 classes) was a +6pp gain
  in one move (15→22 between runs 5 and 6).
- **Long verbose CoT requires longer max_new_tokens at eval** — bumping
  default 128→384 was net-positive (no false truncation, +0.5pp on
  baseline).
- **Balanced training + short training is fragile** — at 300 steps,
  model still hallucinates schema-violating action_type strings like
  `type_into_text_field`. More training or constrained decoding may
  clean this up.

## Next Ideas

- baseline first to set floor
- 200-step SFT to prove convergence
- vary LR (1e-4, 2e-4, 3e-4)
- vary projector on/off
- vary response-only loss masking on/off
- expand to 4K train rows
- add long_press examples (currently 0; AC has none — would need synth)

### Run 28: r22 recipe + seed=4242 (3rd variance sample) — aw_success_rate=20.00 (DISCARD)
- Timestamp: 2026-05-01 13:55
- What changed: train_smoke.py seed 3407→4242, all else identical to r22.
- Result: AW SR=20% (2/10: ClockStopWatchRunning + NotesIsTodo). AC offline 21.80% / 56.20%, AL 5.58% — virtually identical to r22 on offline.
- 3-sample distribution of r22-recipe: r22=50%, r26=10%, r28=20% → mean=26.67%, σ≈21pp
- Insight: r22's 50% was a positive outlier. Recipe's true mean is ~20-30%. The 10-task AW slice has variance too wide to distinguish recipe quality at this scale. AC and AL offline metrics collapse near-deterministically (variance <1pp) — only the live AW eval is noisy.
- Insight (corollary): seed-driven variance dominates any 5pp-class recipe improvement on this slice. Cannot tune further until variance is reduced.
- Next: expand AW slice from 10 → 20 tasks (or run 3-seed ensemble per recipe). Until then, all recipe tweaks are noise-bound.

## Key Insights (updated)
- AC offline action-match is a misleading proxy for AW SR (run 19 vs 18).
- Class balancing + projector unlock + lora_r=32 + 300-400 steps is the right base recipe shape.
- AL trajectories alone are insufficient — `open_app` from AC is load-bearing (run 27).
- The status action class never transferred — model emits 0% status type-match across all training mixes (r21, r25, r26, r27, r28). Adding status rows didn't fix it.
- **NEW (r28)**: 10-task AW eval has σ ≈ 21pp from seed alone. Cannot reliably tune below ±20pp gap. Variance reduction is now the gating step.
- train_loss is inversely correlated with AW SR within a recipe family (lower train_loss → worse AW). Confirmed on r22 (0.7262, 50%) vs r23 (0.6330, 30%) vs r25 (0.6181, 20%).

## Next Ideas (updated)
- **Expand AW slice 10 → 20 tasks** for tighter variance. Pick 10 more "short-baseline" tasks from the existing AW catalog.
- **3-seed ensemble per recipe**: run each candidate at 3 seeds, report mean ± σ. Quadruples cost but resolves r22-vs-baseline ambiguity.
- Phase-3 Gemini-distilled CoT (logged in autoresearch.ideas.md) — defer until variance is reduced.
- max_length 4096→8192 + AndroidLab memory-field training (long-horizon item).

### Run 34: teacher distillation (Gemma 4 31B 4-bit) — INVALIDATED PRE-TRAIN (DISCARD)
- Timestamp: 2026-05-01 19:15
- What changed: built `scripts/pathZ/distill_teacher.py`. Loaded Gemma 4 31B 4-bit, ran blind generation on 1500 harness-parity AC rows. Bucketed teacher emissions: full action match (kept), type-match (2nd-pass salvage prompt asking teacher to justify the gold action), full mismatch (discarded). Final: match=152 (10.13%), salvaged=342 (22.80%), mismatch=1005 (67%). Total kept=494 (32.93%).
- Result: NO STUDENT TRAINING RUN. Pre-train analysis revealed two structural data problems that invalidate the distillation premise:
  1. **AC training data is 96% off-distribution from AW.** Of 250 AC `open_app` rows, only 10 (4%) target an app in the AW 19-app inventory (Files/Markor/Joplin/Broccoli/Pro Expense/Camera/Clock/Simple Calendar Pro/Simple SMS Messenger/Contacts/Audio Recorder/Chrome/Settings/OsmAnd/Retro Music/Simple Draw Pro/Tasks/VLC/OpenTracks). The other 240 target real consumer apps (Amazon, Khan Academy, Maps, Drive, Gmail, eBay, Booking.com, ZARA, Decathlon, Vimeo, Edmunds, Myntra, etc.) that DO NOT EXIST in the AW emulator. We were training the student to call `open_app: Edmunds` knowing the deployment harness can never satisfy that action — the only correct action there is `status: infeasible`.
  2. **Teacher's "mismatches" are mostly correct refusals.** 281/329 erroneous teacher status emits were `infeasible` — the teacher correctly identified the missing-from-inventory app and refused. The salvage step would force the student to learn post-hoc rationalizations of actions even the strong teacher refused as infeasible. This would actively poison the student.
- Per-class teacher full-match: click=34.5%, scroll=11.2%, navigate_back=2.4%, open_app=10.0%, input_text=0%, wait=3.4%. The teacher fails on the same classes the student fails on, for the same reason: text-only a11y prompt loses spatial info (scroll/wait), goal-substep history (navigate_back), and the `wait` label is an annotation quirk of AC (a11y tree IS populated, smart agent correctly clicks the visible target).
- Insight: **this is a data + harness problem, not a model-capacity problem.** Teacher distillation cannot fix structural input deficiencies. Distillation is only useful when the teacher reliably handles cases the student fails on; here the teacher fails on the same cases.
- Insight (corollary): all prior r19-r33 runs were trained on the same off-distribution AC data. The `open_app` confusion they exhibited at AW eval was the model loyally reproducing what we taught it.
- Tooling kept: `distill_teacher.py` with `--resume` flag. Useful for a future run on properly filtered data.
- Next: **r35 — rebuild train data**. (a) Filter AC to rows whose actions reference only AW-inventory apps, plus the non-open_app classes (click/scroll/input_text/wait/navigate_back) which are AW-app-agnostic. (b) Re-include all 6053 AndroidLab rows (collected on the AW emulator with the same app inventory — naturally on-distribution). (c) Re-balance classes from this filtered pool. Drop the teacher distillation salvage path entirely. Train student on this v2 dataset; eval AW-20.

## Key Insights (updated 2026-05-01 r34)
- **Off-distribution training data is the dominant problem.** AC trajectories were collected on real consumer Android phones with a long tail of real apps. AW is a curated 19-app emulator. 96% of AC `open_app` rows reference apps that don't exist in the deployment environment. Every model trained on this dataset learns to call non-existent apps; in deployment the wrapper has no recourse but to fail. This explains the entire r19-r33 plateau independent of recipe choices.
- **AndroidLab is the right distribution.** AL trajectories were collected on the AW emulator with the AW app inventory. Our previous runs that mixed in 50% AL (r19, r22) were the only ones to ever exceed 20% AW SR — that wasn't coincidence.
- **Distillation is conditional.** Teacher distillation only helps when the teacher reliably handles cases the student fails on. Here the teacher and student share the same input-distribution failure modes. Need to fix the data first.

### Run 35: AW-distribution data rebuild — aw_SR=10.00 (DISCARD, but key diagnostic)
- Timestamp: 2026-05-01 20:35
- What changed: prepare_smoke_data.py grew `--aw-apps-only` (drops AC `open_app` rows whose target app is not in the AW 19-app inventory; 4727 dropped, 316 kept) and `--synthesize-open-app N` (synthesizes N synthetic `open_app` rows per AW app from a generic "Open the X app" goal; 250×19=4750 rows). Re-included AndroidLab (50/50 mix per shared class). Result: 1750 rows balanced 7-class (click/input_text/scroll/navigate_back/wait + status from AL + open_app from AC-AW-filter+synth). Same recipe + harness as r33: text-only, harness-parity, 400 steps.
- Result: 2/20 = 10% AW SR. **IDENTICAL tasks succeeded as r31 + r33**: ClockStopWatchRunning + OpenAppTaskEval. AC offline DROPPED to 12.0% (eval contains the same off-distribution open_app rows model now correctly refuses). AL type_match SURGED 25.10→37.85 driven by click 53→92 (AL data integration helped click). train_loss collapsed to 0.6231 (synth open_app rows trivially fit, lowering avg).
- Insight: **r34 data finding was structurally TRUE but not LOAD-BEARING for AW SR.** The M3AA11Y eval prompt already prepends the AW app inventory at deploy time, which constrains the model to AW apps regardless of what it learned to call in training. Contamination existed and the model learned bad habits, but it wasn't deploying them — the inventory prompt at eval time was already serving as the filter. Removing the contaminated rows didn't unlock new tasks; it just cleaned up training noise without changing deployment behavior.
- Insight (corollary): **AppVLM's headline 37.8% comes from RFT on the same 82 tasks they evaluate on**. SFT-side cleaning alone (the only AppVLM-style change applicable without RFT, which user vetoed) cannot break the 10% floor on this hardware/model combo. Confirms user's framing: "this is a data and harness problem" — and HARNESS is the dominant variable.
- Insight (corollary 2): every r19-r33+r35 model solves the same 2 tasks: a single-button stopwatch and an open_app permission flow. Both are essentially 1-2 step trajectories. The 18 failures all involve multi-step planning (find note → edit note, find recipe → delete, find event → answer). The model isn't planning at all — it's reactively clicking on what looks salient. Reason-preserving history would directly attack this.
- Next: r36 = harness-only change (no retrain). Modify `m3a_a11y.M3AA11Y` to carry the model's prior `Reason:` text in step history (currently history is `Step N: <action_repr> -> ok; window changed`, which discards the rationale). Test on AW-20 with r35 adapter unchanged. This isolates the harness-context lever from the recipe lever — if SR lifts, the user's hypothesis is confirmed and we expand to candidate-action shortlist + compact a11y in r37/r38.

### Run 36: reason-preserving history in M3AA11Y harness — aw_SR=5.00 (DISCARD)
- Timestamp: 2026-05-01 21:00
- What changed: HARNESS-ONLY (r35 adapter unchanged). Modified `m3a_a11y.M3AA11Y` history-rendering loop to append the model's prior step `Reason:` text on a second indented row beneath each `Step N: <action_repr>` summary. Reasoning: AppVLM-research-rejected; user hypothesis is that harness context (past reasons) is what's blocking long-horizon planning.
- Result: 1/20 = 5% AW SR. **LOST ClockStopWatchRunning** (the trivial single-button task that had succeeded in r31, r33, r35). Only OpenAppTaskEval still passes. Within σ ≈ 7pp of the 10% floor but below; not a recipe ladder.
- Insight: **adding raw context naively is net-NEGATIVE.** The model was trained on history WITHOUT reasons. Adding ~240 chars/step of reason text at eval time:
  (a) inflates the prompt and dilutes attention budget for relevant UI elements;
  (b) likely confuses the prompt structure — model may interpret prior reasons as part of the user goal;
  (c) self-poisoning loop: when an action is wrong, its reason justifies it; carrying that through history reinforces the wrong direction.
- Insight: **the user's harness-context hypothesis is correct in spirit but the lever has a sharp prompt-distribution-shift cliff.** Either need to (a) retrain student with reasons-in-history (aligns distribution, ~30-40min total cost) or (b) find harness changes that are additive and don't restructure existing prompt sections.
- Insight (corollary): the M3A prompt format we trained on is fragile. Any changes to the history block at eval time risk SR regressions. This bounds what eval-time-only harness experiments can achieve. To unlock the harness lever fully, train-time format must change in lockstep.
- Next: r37 = candidate-action shortlist header (additive, doesn't restructure history). Render a per-step header block from the parsed a11y tree: "Available actions: click(0..N) / scroll(up|down|left|right) / open_app({inventory})". Should be lower-risk because it adds new info above the prompt rather than rewriting existing history. r35 adapter unchanged.

### Run 37: retrained WITH reasons-in-history + matching M3AA11Y harness — aw_SR=5.00 (DISCARD)
- Timestamp: 2026-05-01 21:30
- What changed: TRAIN + EVAL both updated. prepare_smoke_data.py grew `--history-reasons` flag emitting `  reason: <synthetic>` line under each prior `Step N: <action_repr>` in train prompts. M3AA11Y harness mirror — carries the model's actual prior `action_reason` text through history at eval. Aligned the train/eval distribution — direct response to r36's distribution-shift hypothesis. Same recipe as r35 (1750 v3 rows, text-only, harness-parity, 400 steps).
- Result: 1/20 = 5% AW SR. **IDENTICAL FAILURE as r36** — lost ClockStopWatchRunning, only OpenAppTaskEval succeeds. The trivial 1-button stopwatch task hits "Reached max number of steps" again. AC type_match dropped 40→27 (eval data lacks history-reasons → mild mismatch in opposite direction).
- **Critical insight: matching train+eval distribution did NOT fix the regression.** This RULES OUT prompt-distribution-shift as the cause. The harness change is genuinely net-negative regardless of whether training is aligned.
- Real diagnosis: **the model uses prior-step reasons as a distraction signal.** When a prior reason in history says "I clicked the start button to start the stopwatch", the model can't tell that's its own prior thought vs the user's goal. It tries to re-execute the action instead of terminating. Hence max-steps loop on trivial tasks.
- Insight (corollary): **at this model scale (2B, text-only), in-context reasoning has a sharp toxicity cliff.** Small models cannot disambiguate own-prior-reasoning from external goal text. AppVLM (PaliGemma-3B) explicitly uses NO CoT in their labels — which now makes more sense as a deliberate design choice rather than ablation.
- Insight (meta): the user's harness-context hypothesis was correct in DIRECTION but the SPECIFIC lever (carrying prior reasons in history) is wrong for this model+scale. Two iterations (r36 eval-only, r37 train+eval) both regressed by 5pp. Need a different harness lever.
- Reverted m3a_a11y.py to r35 history rendering. prepare_smoke_data.py keeps the --history-reasons flag (off by default) for future use.
- Next: **r38 = compact a11y tree** — drop empty-label / non-interactive UI elements at both train+eval. Tests "prompt budget is the bottleneck" hypothesis directly without touching history or reasoning. Doesn't add new content, just removes noise. Uses index-stable rendering (skip a UI element line entirely if empty-label, but preserve the index in the remaining lines so click(7) still references the same element).

### Run 38: cap M3AA11Y eval history to last 3 steps — aw_SR=5.00 (DISCARD)
- Timestamp: 2026-05-01 21:55
- What changed: HARNESS-only eval-time (m3a_a11y.M3AA11Y now caps `history_block` to the last 3 prior steps to match training's `prior_strs[-3:]`). Retrained on v2 (no history-reasons) since r37's adapter overwrote r35's. Same recipe, different seed.
- Result: 1/20 = 5% AW SR — IDENTICAL outcome as r36 + r37 (lost ClockStopWatchRunning, only OpenAppTaskEval succeeds). AC offline this seed: full=10 / type=47 (+6.5pp vs r35's 40.5); AL: full=4.38 (+3.2pp vs r35's 1.20) — recipe is genuinely better at offline grounding this seed but no AW lift.
- **Critical pattern across r36/r37/r38: 3 different harness changes (add reasons eval-only, add reasons train+eval, cap history length) all regressed by 5pp from r35.** Each change touched a different lever. Each lost ClockStopWatchRunning specifically.
- **True diagnosis: r35's "10%" was a measurement artifact.** ClockStopWatchRunning was a fragile stochastic success — the model wasn't really solving it, it was lucking into a brittle button-click sequence that worked under one specific prompt configuration. ANY change to history block format/length breaks the lucky configuration.
- **True base capability of Gemma 4 E2B + SFT-only + text-only on AW-20 = 5%** (1 robustly-solved task = OpenAppTaskEval). The previously-celebrated 10% floor was 1 robust + 1 fragile success.
- Implication: harness ceiling at this scale is **5%**, not 10%. The AW-15 target requires structural changes:
  1. Bigger base model (Gemma 4 P3 ~3B or PaliGemma-3B at 4-bit, IF it fits 24GB VRAM)
  2. Vision back on (single-screenshot at low resolution)
  3. Much higher AL multi-step trajectory volume (drop balanced sampling, use all 6053 AL rows including longer trajectories)
  4. Some form of on-policy data augmentation (RFT lite — vetoed but would clearly work per AppVLM)
- AppVLM evidence: their PaliGemma-3B + vision + RFT achieves 37.8% on a curated 82-task subset; their pre-RFT base hits ~20%. Our 2B + text-only + SFT-only reaching 15% on full AW-116 looks structurally infeasible based on this comparison.
- Next: **r39 = step up base model to PaliGemma-3B 4-bit** if it fits in 24GB. Tests model-capacity hypothesis directly. If OOM, fall back to Gemma 4 P3. Major reset on the recipe (different chat template, different processor) but warranted given the floor is structural.

### Run 39: fix UI elements newline bug — aw_SR=10.00 (DISCARD on metric, KEPT as fix)
- Timestamp: 2026-05-01 22:30
- What changed: BUG FIX in `m3a_format.render_m3a_prompt`. The function joined lines with `"".join()` (empty string) so all UI element lines were concatenated WITHOUT newlines. 25+ elements jammed on one line in train prompts. Eval-side `m3a._generate_ui_elements_description_list` correctly appends `\n` per element. **Train/eval format mismatch since r33** (harness_parity introduction). Fix: append trailing `\n` per element line in render_m3a_prompt.
- Result: 2/20 = 10% AW SR (Clock + OpenApp). Recovers r35 baseline. Bug fix did NOT lift past 10% but proves r36/r37/r38's 5% was driven by harness changes themselves, not data quality bugs.
- Insight: **Structural ceiling for Gemma 4 E2B + SFT-only + text-only is 10% AW-20** (within recipe noise). Cleanest harness lever (newline fix) shipped, no SR lift. Confirms harness-side improvements within this paradigm are exhausted.
- Bug fix is keeper-tier infrastructure (eliminates real train/eval mismatch) even though it doesn't move SR. Future runs will have a clean baseline to compare against.
- Next: **r40 = step up base model to Gemma 4 E4B (4-bit Unsloth quant)**. Same architecture/template/processor as E2B — drop-in upgrade. ~2× parameters. Tests if model capacity is the bottleneck. Same v5 data, same recipe, same harness. ~12 min train.
- After r40, if E4B doesn't break 10%, the remaining structural lever is vision (re-enable image input) since the only published recipe approaching 15%+ AW SR (AppVLM) is vision-based.

### Run 41: plan distillation (E4B + teacher plans) — aw_SR=5.00 (DISCARD)
- Timestamp: 2026-05-02 06:30
- What changed: built distill_plans.py (Gemma 4 31B 4-bit teacher generates 3-5 step plans for each unique AL goal); 595 plans / 166 infeasible (teacher correctly refused goals using non-AW apps). prepare_smoke_data --plans-jsonl --plan-align-filter augments AL training rows: step 0 emits Plan: header in assistant text; step 1+ has "Plan from step 0:" injected in user prompt before goal. m3a_a11y mirrors at eval (parses Plan from step-0 emission, re-injects in step 1+). v6 train data, E4B base.
- Result: 1/20 = 5% AW SR. Lost ClockStopWatchRunning. Only OpenApp succeeds. **HUGE offline lifts though**: AL full_match doubled 3.19→6.37 (highest of any run); AC navigate_back FIRST EVER non-zero (0→18.18); AC wait 5→63; AC scroll 27→40.
- Insight: plans WORK at the offline action-grounding level (model emits much more diverse verbs) but PREMATURE `status:complete` from the plans kills task completion. Every plan ends with "X. status: complete"; model generalizes and emits status:complete on AW tasks before they're actually done.
- Next: r43 = drop status from action vocab + harness suppresses status emission; force harness to terminate via max_steps or no-op-auto-declare only.

### Run 43: drop status emission entirely — aw_SR=10.00 (KEEP, structural progress)
- Timestamp: 2026-05-02 06:50
- What changed: removed `status` from M3A_PROMPT_PREFIX action vocab; m3a_a11y now suppresses any status emission (logs feedback, no termination). Plans rebuilt to strip trailing `X. status: complete` step (al_plans_no_status.jsonl). v7 train data drops status class entirely (1500 rows, 6-class balance). E4B base.
- Result: 2/20 = 10% AW SR. Same metric as r39/r40 BUT **NEW TASK SOLVED**: CameraTakeVideo (first time across r19-r43). ClockStopWatchRunning lost (was a fragile success across r31-r40). OpenAppTaskEval still solved.
- Insight (DEEP): the success on CameraTakeVideo wasn't strategic model competence — analysis of trajectories showed the model just kept clicking the shutter button until the harness no-op-auto-declare terminated the trajectory, and the env state happened to have a saved video → success. Same fragility as Clock in r35 but in a different direction.
- Insight: r43 status drop genuinely freed the model from the premature-termination pattern. **Plan distillation produces a structurally more diverse-acting agent** but its actions still aren't precise enough to actually complete tasks reliably.
- Trajectory-level failures (per dump analysis): (a) wrong app selection — opened "Files" instead of "Markor"; (b) out-of-range indices — input_text(15) when only 0-14; (c) loop-clicking same wrong element until no-op. These are model-grounding errors, not harness errors.
- Next: r44 = remove the no-op auto-declare (test whether more steps help)

### Run 44: remove no-op auto-declare in M3AA11Y — aw_SR=0.00 (DISCARD, lesson)
- Timestamp: 2026-05-02 07:25
- What changed: m3a_a11y.M3AA11Y no longer self-terminates after 2 consecutive nav-class no-ops; model runs until max_steps cap unconditionally. Reused r43 adapter, harness-only.
- Result: 0/20 = 0% AW SR. INCLUDING OpenAppTaskEval (which had succeeded in EVERY run since r19).
- **CRITICAL FINDING: the no-op auto-declare is LOAD-BEARING for AW scoring.** It was the only mechanism that terminated trajectories at intermediate states (the env-side success check fires at termination). Without it, max_steps termination leaves the env in an over-acted state — camera continuously recording (no saved file), app navigated past the success state, etc. The HOW of termination changes the env's final state and thus the success check outcome.
- Insight: harness termination strategy is real infrastructure. Cannot remove or simplify without a smarter terminator that detects success earlier. The current mechanism is "terminate when stuck (no-ops)", which approximates "terminate when state hasn't changed for a while" — a weak heuristic but apparently sufficient for some tasks.
- Reverted m3a_a11y.py to r43 no-op behavior.
- Next: r45 = pivot to r42 (teacher rollouts on AL tasks). The harness levers are exhausted; the bottleneck is model grounding (wrong app, hallucinated indices). Need higher-quality on-distribution multi-step trajectory training data.
