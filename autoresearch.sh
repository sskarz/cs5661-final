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
TRAIN=data/pathZ/smoke/train.jsonl
EVAL=data/pathZ/smoke/eval.jsonl
if [[ ! -f "$TRAIN" || ! -f "$EVAL" ]]; then
  echo "[autoresearch] smoke data not built; running prepare_smoke_data.py"
  uv run python scripts/pathZ/prepare_smoke_data.py
fi

OUT=outputs/pathZ_smoke
RECIPE=${RECIPE:-train}

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
uv run python scripts/pathZ/train_smoke.py \
    --train-jsonl "$TRAIN" \
    --output-dir "$OUT"

# --- 3) Eval the trained LoRA on AC-val (grounding proxy) ---
echo "[autoresearch] phase=eval-ac (trained adapter)"
uv run python scripts/pathZ/eval_smoke.py \
    --adapter "$OUT/checkpoint-final" \
    --eval-jsonl "$EVAL" \
    --save-preds outputs/pathZ_smoke_eval/trained_ac.jsonl \
    | sed 's/^METRIC \([a-zA-Z_]*\)=/METRIC ac_\1=/'

# --- 4) Eval on AL-val (trajectory proxy, closer to AW distribution) ---
EVAL_AL=data/pathZ/smoke/eval_al.jsonl
if [[ -f "$EVAL_AL" ]]; then
  echo "[autoresearch] phase=eval-al (trained adapter)"
  uv run python scripts/pathZ/eval_smoke.py \
      --adapter "$OUT/checkpoint-final" \
      --eval-jsonl "$EVAL_AL" \
      --save-preds outputs/pathZ_smoke_eval/trained_al.jsonl \
      | sed 's/^METRIC \([a-zA-Z_]*\)=/METRIC al_\1=/'
fi
