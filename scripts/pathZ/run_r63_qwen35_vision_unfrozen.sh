#!/usr/bin/env bash
set -euo pipefail

REPO=/home/sanskar/Documents/Github/cs5661-final
LOGDIR="$REPO/outputs/logs"
LOG="$LOGDIR/r63_train_qwen35_r57b_plus_genesis_vision_unfrozen_e2.log"

mkdir -p "$LOGDIR"
cd "$REPO"

{
  echo "[$(date)] === r63 Qwen3.5 r57b + Genesis vision unfrozen continuation start ==="
  echo "base_adapter=outputs/r57b/checkpoint-final"
  echo "train_jsonl=data/pathZ/genesis_vision_rebuild/train.expanded.jsonl"
  echo "output_dir=outputs/r63_qwen35_r57b_plus_genesis_vision_unfrozen_e2"
  echo "vision_head=unfrozen"
} | tee "$LOG"

uv run python scripts/pathZ/train_smoke.py \
  --model outputs/r57b/checkpoint-final \
  --train-jsonl data/pathZ/genesis_vision_rebuild/train.expanded.jsonl \
  --data-dir data/pathZ/genesis_vision_rebuild \
  --output-dir outputs/r63_qwen35_r57b_plus_genesis_vision_unfrozen_e2 \
  --epochs 2 \
  --lr 5e-5 \
  --batch-size 1 \
  --grad-accum 4 \
  --warmup-steps 4 \
  --lora-r 32 \
  --lora-alpha 64 \
  --train-vision-head \
  2>&1 | tee -a "$LOG"

echo "[$(date)] === r63 training done ===" | tee -a "$LOG"
