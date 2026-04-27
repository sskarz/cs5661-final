#!/usr/bin/env bash
# Eval all D2 checkpoints at 200 samples each.
set -e
mkdir -p outputs/eval
for ckpt in 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 5966; do
    out="outputs/eval/d2_ckpt${ckpt}.json"
    if [ -f "$out" ]; then
        echo "skip ${ckpt} (exists)"
        continue
    fi
    echo "=== eval D2 checkpoint-${ckpt} ==="
    uv run python scripts/eval_androidcontrol.py \
        --adapter "outputs/gemma4-e2b-androidcontrol-lora-runD/checkpoint-${ckpt}" \
        --num-samples 200 --seed 3407 --save-all-predictions \
        --output "$out" 2>&1 | tee "outputs/eval/d2_ckpt${ckpt}.log"
done
echo "=== D2 sweep done ==="
