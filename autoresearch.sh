#!/usr/bin/env bash
# autoresearch.sh — single-iteration runner for the pathZ SFT recipe smoke.
#
# Usage:  ./autoresearch.sh
#
# Behavior:
#   - If RECIPE=baseline   → run eval against zero-shot Gemma 4 E2B (no LoRA)
#   - else                 → train (using current train_smoke.py recipe), then eval the resulting LoRA
#
# Outputs `METRIC <name>=<value>` lines for the autoresearch loop to parse.
set -euo pipefail

REPO=/home/sanskar/Documents/Github/cs5661-final
cd "$REPO"

# --- 1) Ensure smoke data is built ---
TRAIN=${TRAIN:-data/pathZ/smoke/train.jsonl}
EVAL=data/pathZ/smoke/eval.jsonl
if [[ ! -f "$TRAIN" || ! -f "$EVAL" ]]; then
  echo "[autoresearch] smoke data not built; running prepare_smoke_data.py"
  uv run python scripts/pathZ/prepare_smoke_data.py
fi

OUT=outputs/pathZ_smoke
RECIPE=${RECIPE:-train}
TEXT_ONLY=${TEXT_ONLY:-0}
TEXT_ONLY_FLAG=""
if [[ "$TEXT_ONLY" == "1" ]]; then
  TEXT_ONLY_FLAG="--text-only"
  echo "[autoresearch] mode=text-only (a11y-only, no images)"
fi

if [[ "$RECIPE" == "baseline" ]]; then
  echo "[autoresearch] phase=baseline (no training)"
  rm -rf "$OUT"
  uv run python scripts/pathZ/eval_smoke.py \
      --eval-jsonl "$EVAL" \
      --save-preds outputs/pathZ_smoke_eval/baseline.jsonl
  exit 0
fi

# --- 2) Train ---
echo "[autoresearch] phase=train"
rm -rf "$OUT"
MODEL_ARG=""
if [[ -n "${MODEL:-}" ]]; then
  MODEL_ARG="--model $MODEL"
  echo "[autoresearch] base model: $MODEL"
fi
MAX_STEPS_ARG=""
if [[ -n "${MAX_STEPS:-}" ]]; then
  MAX_STEPS_ARG="--max-steps $MAX_STEPS"
  echo "[autoresearch] max_steps: $MAX_STEPS"
fi
uv run python scripts/pathZ/train_smoke.py \
    --train-jsonl "$TRAIN" \
    --output-dir "$OUT" \
    $MODEL_ARG \
    $MAX_STEPS_ARG \
    $TEXT_ONLY_FLAG

# --- 3+4) Parallel eval: AC-val + AL-val. Both load E4B 4-bit (~5GB) +
# adapter; together comfortably fit in 24GB of VRAM. Cuts wall-clock ~50%.
echo "[autoresearch] phase=eval-ac+eval-al (parallel, trained adapter)"
LOG_AC=/tmp/autoresearch-eval-ac.log
LOG_AL=/tmp/autoresearch-eval-al.log

uv run python scripts/pathZ/eval_smoke.py \
    --adapter "$OUT/checkpoint-final" \
    --eval-jsonl "$EVAL" \
    --save-preds outputs/pathZ_smoke_eval/trained_ac.jsonl \
    $TEXT_ONLY_FLAG \
    > "$LOG_AC" 2>&1 &
PID_AC=$!

EVAL_AL=data/pathZ/smoke/eval_al.jsonl
PID_AL=""
if [[ -f "$EVAL_AL" ]]; then
  uv run python scripts/pathZ/eval_smoke.py \
      --adapter "$OUT/checkpoint-final" \
      --eval-jsonl "$EVAL_AL" \
      --save-preds outputs/pathZ_smoke_eval/trained_al.jsonl \
      $TEXT_ONLY_FLAG \
      > "$LOG_AL" 2>&1 &
  PID_AL=$!
fi

wait "$PID_AC"
EXIT_AC=$?
grep '^METRIC ' "$LOG_AC" | sed 's/^METRIC \([a-zA-Z_]*\)=/METRIC ac_\1=/' || true
if [[ $EXIT_AC -ne 0 ]]; then
  echo "[autoresearch] eval-ac failed (exit $EXIT_AC); tail:"
  tail -40 "$LOG_AC"
fi

if [[ -n "$PID_AL" ]]; then
  wait "$PID_AL"
  EXIT_AL=$?
  grep '^METRIC ' "$LOG_AL" | sed 's/^METRIC \([a-zA-Z_]*\)=/METRIC al_\1=/' || true
  if [[ $EXIT_AL -ne 0 ]]; then
    echo "[autoresearch] eval-al failed (exit $EXIT_AL); tail:"
    tail -40 "$LOG_AL"
  fi
fi
