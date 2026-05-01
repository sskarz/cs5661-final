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
