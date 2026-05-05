#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="outputs/r64_amex_gemma_sft_subset"
LOG_DIR="outputs/logs"
LOG_FILE="$LOG_DIR/r64_amex_gemma_sft_subset.log"
TRAIN_JSONL="data/pathZ/amex_sft_subset/train.jsonl"

mkdir -p "$OUT_DIR" "$LOG_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[r64-amex-gemma-sft] start $(date -Is)"
echo "[r64-amex-gemma-sft] root=$ROOT"
echo "[r64-amex-gemma-sft] train_jsonl=$TRAIN_JSONL"
echo "[r64-amex-gemma-sft] output_dir=$OUT_DIR"
echo "[r64-amex-gemma-sft] log_file=$LOG_FILE"
git status --short
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

uv run python scripts/pathZ/train_smoke.py \
  --train-jsonl "$TRAIN_JSONL" \
  --data-dir "$ROOT" \
  --model unsloth/gemma-4-E4B-it \
  --output-dir "$OUT_DIR" \
  --epochs 1 \
  --lr 5e-5 \
  --batch-size 1 \
  --grad-accum 8 \
  --lora-r 16 \
  --lora-alpha 32 \
  --max-length 8192 \
  --no-train-vision-head

echo "[r64-amex-gemma-sft] end $(date -Is)"
