#!/usr/bin/env bash
# Overnight: wait for Run I training to finish, then run eval sweep on every
# checkpoint and post-analysis. Sends one final WAKE_UP_SUMMARY.md to the
# repo root.
set -e
cd "$(dirname "$0")/.."

TRAIN_PID=${TRAIN_PID:-}
ADAPTER_BASE=outputs/gemma4-e2b-pathW-lora-runI
EVAL_OUT_DIR=outputs/eval
LOG_DIR=outputs/eval_logs
mkdir -p "$EVAL_OUT_DIR" "$LOG_DIR"

if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "[$(date)] waiting for training PID $TRAIN_PID to exit..."
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        sleep 60
    done
fi
echo "[$(date)] starting eval sweep"

# Discover checkpoints (numeric sort)
shopt -s nullglob
mapfile -t CKPTS < <(ls -d "$ADAPTER_BASE"/checkpoint-* 2>/dev/null | sort -V)
if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "[$(date)] FATAL: no checkpoints found in $ADAPTER_BASE" | tee outputs/runI_logs/postrun.log
    exit 1
fi
echo "[$(date)] discovered ${#CKPTS[@]} checkpoints"

# Eval each ckpt sequentially. 200 samples / seed 3407 / save predictions.
for ckpt_path in "${CKPTS[@]}"; do
    n=$(basename "$ckpt_path" | sed 's/checkpoint-//')
    out="$EVAL_OUT_DIR/runI_ckpt${n}.json"
    if [ -f "$out" ]; then
        echo "[$(date)] [skip] $out exists"
        continue
    fi
    echo "[$(date)] === eval ckpt-$n ==="
    .venv/bin/python -u scripts/eval_a11y_native.py \
        --data-dir data/androidcontrol_a11y_native \
        --adapter "$ckpt_path" \
        --num-samples 200 --seed 3407 \
        --save-all-predictions \
        --output "$out" \
        > "$LOG_DIR/runI_eval_ckpt${n}.log" 2>&1 || \
        echo "[$(date)] [warn] eval ckpt-$n failed; continuing"
done

echo "[$(date)] eval sweep done; running post-analysis"
.venv/bin/python -u scripts/runI_postanalysis.py \
    --eval-glob "outputs/eval/runI_ckpt*.json" \
    --baseline outputs/eval/native_baseline.json \
    --out-summary WAKE_UP_SUMMARY.md \
    --training-log TRAINING_LOG.md \
    > outputs/runI_logs/postanalysis.log 2>&1

echo "[$(date)] PIPELINE DONE — see WAKE_UP_SUMMARY.md"
