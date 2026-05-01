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

(populated per iteration)

## Key Insights

(populated as we discover them)

## Next Ideas

- baseline first to set floor
- 200-step SFT to prove convergence
- vary LR (1e-4, 2e-4, 3e-4)
- vary projector on/off
- vary response-only loss masking on/off
- expand to 4K train rows
- add long_press examples (currently 0; AC has none — would need synth)
