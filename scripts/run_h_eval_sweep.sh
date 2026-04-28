#!/usr/bin/env bash
# Sequential eval sweep across all Run H checkpoints.
# 200 samples / seed 3407 / save_all_predictions for spatial diagnostics.
set -e
cd "$(dirname "$0")/.."
mkdir -p outputs/eval

CKPTS=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 5966)
ADAPTER_BASE=outputs/gemma4-e2b-androidcontrol-lora-runH

for n in "${CKPTS[@]}"; do
    out=outputs/eval/runH_ckpt${n}.json
    if [ -f "$out" ]; then
        echo "[skip] $out already exists"
        continue
    fi
    echo "=== eval ckpt-$n ==="
    .venv/bin/python -u scripts/eval_androidcontrol.py \
        --adapter "$ADAPTER_BASE/checkpoint-$n" \
        --num-samples 200 --seed 3407 \
        --save-all-predictions \
        --output "$out" 2>&1 | tail -40
    echo "[done] ckpt-$n -> $out"
done

echo "=== sweep summary ==="
.venv/bin/python -c "
import json, glob
rows = []
for p in sorted(glob.glob('outputs/eval/runH_ckpt*.json'), key=lambda x: int(x.split('ckpt')[1].split('.')[0])):
    m = json.load(open(p))['metrics']
    n = int(p.split('ckpt')[1].split('.')[0])
    rows.append((n, m['full_match'], m['parse_rate']))
print(f'{\"step\":>6}  {\"full_match\":>10}  {\"parse_rate\":>10}')
for n, fm, pr in rows:
    print(f'{n:>6}  {fm:>10.4f}  {pr:>10.4f}')
print()
print(f'baseline 0.288  (HF-test, 200 samples, seed 3407)')
print(f'Run E best      0.210  (frozen projector)')
"
