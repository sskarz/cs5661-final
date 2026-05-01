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

- The floor is **20%** full-match, not 0% — the M3A prompt gets you
  surprisingly far on a pre-trained Gemma 4 E2B because its
  instruction-following is decent. The bar is therefore "clearly beat
  20%" not "clearly beat 0%".
- Top failure modes are schema hallucination (4.5% emit a verbose
  action_type string) and `wait`/`scroll` class imbalance.

## Next Ideas

- baseline first to set floor
- 200-step SFT to prove convergence
- vary LR (1e-4, 2e-4, 3e-4)
- vary projector on/off
- vary response-only loss masking on/off
- expand to 4K train rows
- add long_press examples (currently 0; AC has none — would need synth)
