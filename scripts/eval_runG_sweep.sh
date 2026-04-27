#!/usr/bin/env bash
# Eval all Run G checkpoints at 200 samples each (discrete coords + sqrt-inverse weights).
set -e
mkdir -p outputs/eval
DATA_DIR=data/androidcontrol_disc1024
RUN_DIR=outputs/gemma4-e2b-androidcontrol-lora-runG
# Initial sweep at 5 evenly spaced points; can fill in densities later.
for ckpt in 500 1500 3000 4500 5500 5966; do
    out="outputs/eval/runG_ckpt${ckpt}.json"
    if [ -f "$out" ]; then
        echo "skip ${ckpt} (exists)"
        continue
    fi
    if [ ! -d "${RUN_DIR}/checkpoint-${ckpt}" ]; then
        echo "skip ${ckpt} (checkpoint not yet saved)"
        continue
    fi
    echo "=== eval Run G checkpoint-${ckpt} ==="
    uv run python scripts/eval_androidcontrol.py \
        --data-dir "$DATA_DIR" \
        --adapter "${RUN_DIR}/checkpoint-${ckpt}" \
        --num-samples 200 --seed 3407 --save-all-predictions \
        --coord-encoding discrete --grid-size 1024 \
        --output "$out" 2>&1 | tee "outputs/eval/runG_ckpt${ckpt}.log"
done
echo "=== Run G sweep done ==="
