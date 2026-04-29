#!/usr/bin/env bash
# Run L — prior-action history (Path W + CAP + 1-step history).
#
# Single-variable change vs Run K: input data only. v3 dataset has
# "Previous action: <action_v2_json>" prepended to each user prompt;
# step-0 rows get "<none>". Same lr=5e-5, r=16/alpha=32, save-steps=300,
# save-total-limit=10, no cui, no coord loss, no oversampling. v2 output
# schema unchanged (CAP). Inherits §35.5 dataloader speedup.
#
# Epoch budget: 0.3 epoch (~2740 steps) — Run K's best ckpt was ckpt-2700
# at exactly 30% epoch. If history-conditioning is going to help wait/scroll,
# we should see it well within this window. Cuts cost vs full-epoch runs.
#
# Pipeline:
#   1. Train (~1h 15min on RTX 4090, ~2740 steps)
#   2. Sequential val sweep across 9 ckpts (~1h, 686 val rows / ckpt)
#   3. Pick best ckpt by val element-accuracy
#   4. Best-ckpt full-test (~80 min, 8217 test rows)
#   5. Auto-append §44 + §45 to TRAINING_LOG.md, write FINAL_SUMMARY_runL.md
set -e
cd "$(dirname "$0")/.."

RUNL_DIR=outputs/gemma4-e2b-pathW-lora-runL
LOG=outputs/runL_logs/run_l_chain.log
mkdir -p outputs/runL_logs outputs/eval outputs/eval_logs

echo "[$(date)] === run_l_chain start (prior-action history, 0.3 epoch CAP) ===" | tee "$LOG"

# Wipe any partial runL output from a prior aborted launch
rm -rf "$RUNL_DIR" 2>/dev/null || true
rm -f outputs/runL_logs/runL.log 2>/dev/null || true

# ========================================================================
# Step 1: Run L training (0.3 epoch on v3 = v2 + prior-action prefix)
# ========================================================================
echo "[$(date)] launching Run L training..." | tee -a "$LOG"
.venv/bin/python -u scripts/train_sft.py \
    --data-dir data/androidcontrol_a11y_native_v3 \
    --output-dir "$RUNL_DIR" \
    --epochs 0.3 \
    --lora-r 16 --lora-alpha 32 --lr 5e-5 \
    --save-steps 300 --save-total-limit 10 \
    > outputs/runL_logs/runL.log 2>&1 || \
    { echo "[$(date)] Run L training FAILED" | tee -a "$LOG"; exit 1; }

echo "[$(date)] Run L training done; starting val sweep" | tee -a "$LOG"

# ========================================================================
# Step 2: Run L val sweep (sequential)
# ========================================================================
shopt -s nullglob
mapfile -t LCKPTS < <(ls -d "$RUNL_DIR"/checkpoint-* 2>/dev/null | sort -V)
echo "[$(date)] discovered ${#LCKPTS[@]} Run L checkpoints" | tee -a "$LOG"
for ckpt_path in "${LCKPTS[@]}"; do
    n=$(basename "$ckpt_path" | sed 's/checkpoint-//')
    out="outputs/eval/runL_val_ckpt${n}.json"
    [ -f "$out" ] && { echo "[$(date)] [skip] $out exists" | tee -a "$LOG"; continue; }
    echo "[$(date)] === Run L val eval ckpt-$n ===" | tee -a "$LOG"
    .venv/bin/python -u scripts/eval_a11y_native.py \
        --data-dir data/androidcontrol_a11y_native_v3 \
        --split val \
        --adapter "$ckpt_path" \
        --num-samples 9999 --seed 3407 \
        --save-all-predictions \
        --output "$out" \
        > "outputs/eval_logs/runL_val_ckpt${n}.log" 2>&1 || \
        echo "[$(date)] [warn] Run L val eval ckpt-$n failed" | tee -a "$LOG"
done

# ========================================================================
# Step 3: Pick best Run L ckpt by val element-accuracy
# ========================================================================
BEST_L_CKPT=$(.venv/bin/python -c "
import json, glob, re, sys
sys.path.insert(0, 'scripts')
from rescore_native_element import element_match
best = None; best_acc = -1
for p in sorted(glob.glob('outputs/eval/runL_val_ckpt*.json')):
    m = re.search(r'ckpt(\d+)', p)
    if not m: continue
    n = int(m.group(1))
    raw = json.load(open(p))
    preds = raw.get('all_predictions') or []
    if not preds: continue
    ok = sum(1 for p_ in preds if element_match(p_.get('pred') or {}, p_.get('gt') or {}))
    acc = ok / max(len(preds), 1)
    if acc > best_acc:
        best_acc = acc; best = n
print(best)
" 2>>"$LOG")
echo "[$(date)] best Run L val ckpt = $BEST_L_CKPT" | tee -a "$LOG"

# ========================================================================
# Step 4: Run L full-test on best
# ========================================================================
if [ -n "$BEST_L_CKPT" ]; then
    out="outputs/eval/runL_ckpt${BEST_L_CKPT}_fulltest.json"
    if [ -f "$out" ]; then
        echo "[$(date)] [skip] $out exists" | tee -a "$LOG"
    else
        echo "[$(date)] running full-test on Run L ckpt-${BEST_L_CKPT}" | tee -a "$LOG"
        .venv/bin/python -u scripts/eval_a11y_native.py \
            --data-dir data/androidcontrol_a11y_native_v3 \
            --adapter "$RUNL_DIR/checkpoint-${BEST_L_CKPT}" \
            --num-samples 9999 --seed 3407 \
            --save-all-predictions \
            --output "$out" \
            > "outputs/eval_logs/runL_ckpt${BEST_L_CKPT}_fulltest.log" 2>&1 || \
            echo "[$(date)] [warn] Run L best-val-ckpt full-test failed" | tee -a "$LOG"
    fi
fi

# ========================================================================
# Step 5: Append §44 + §45 to TRAINING_LOG.md
# ========================================================================
{
    echo ""
    echo "## 44. Run L — prior-action history (Path W + CAP + 1-step history)"
    echo ""
    echo "_Appended automatically by \`run_l_chain.sh\` at $(date '+%Y-%m-%d %H:%M %Z')._"
    echo ""
    echo "Run L = Run K recipe + v3 dataset (v2 prompt with \"Previous action: <action_v2_json>\" prepended)."
    echo "Single-variable A/B vs Run K: input only — same lr (5e-5), r=16/α=32, save-steps=300, save-total-limit=10."
    echo "Epoch budget: 0.3 (Run K's best ckpt was at 30% epoch; longer wouldn't have helped)."
    echo ""
    echo "Best Run L val ckpt by val element-accuracy: \`ckpt-${BEST_L_CKPT:-NA}\`."
    echo ""
    echo "### Run L val sweep (element-accuracy)"
    echo ""
    echo '```'
    .venv/bin/python -u scripts/rescore_native_element.py outputs/eval/runL_val_ckpt*.json --label "Run L val" 2>&1
    echo '```'
    if [ -n "$BEST_L_CKPT" ] && [ -f "outputs/eval/runL_ckpt${BEST_L_CKPT}_fulltest.json" ]; then
        echo ""
        echo "### Run L best-ckpt full-test"
        echo ""
        echo '```'
        .venv/bin/python -c "
import json
m = json.load(open('outputs/eval/runL_ckpt${BEST_L_CKPT}_fulltest.json'))['metrics']
print(f'full_match (tap-radius, legacy): {m[\"full_match\"]:.4f}')
print(f'parse_rate:                     {m[\"parse_rate\"]:.4f}')
print(f'tap_oracle_reachability:        {m.get(\"tap_oracle_reachability\", 0):.4f}')
print(f'n_samples:                      {m[\"num_samples\"]}')
print()
print('per-action-type:')
for k, v in m['per_type'].items():
    print(f'  {k:>16}: n={v[\"n\"]:>4} acc={v[\"accuracy\"]:.3f}')
"
        echo '```'
        echo ""
        .venv/bin/python -u scripts/rescore_native_element.py "outputs/eval/runL_ckpt${BEST_L_CKPT}_fulltest.json" --label "Run L full-test" 2>&1 | sed 's/^/    /'
    fi
} >> TRAINING_LOG.md

# ========================================================================
# Write FINAL_SUMMARY_runL.md (full lineage under element-accuracy)
# ========================================================================
.venv/bin/python -c "
import json, glob, re, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from rescore_native_element import element_match

def element_acc(p):
    raw = json.loads(Path(p).read_text())
    preds = raw.get('all_predictions') or []
    if not preds: return None, 0
    ok = sum(1 for p_ in preds if element_match(p_.get('pred') or {}, p_.get('gt') or {}))
    return ok / len(preds), len(preds)

def fm(p):
    return json.loads(Path(p).read_text())['metrics']['full_match']

lines = ['# Final summary — baseline + Run I + Run J + Run K + Run L', '',
         '## Headline (full test, 8217 rows)', '',
         '| run | element-acc | tap-radius (legacy) | n |',
         '|---|---|---|---|']
files = [
    ('baseline', 'outputs/eval/native_baseline_fulltest.json'),
    ('Run I ckpt-7800', 'outputs/eval/runI_ckpt7800_fulltest.json'),
]
for p in sorted(glob.glob('outputs/eval/runJ_ckpt*_fulltest.json')):
    n = re.search(r'ckpt(\d+)', p).group(1)
    files.append((f'Run J ckpt-{n}', p))
for p in sorted(glob.glob('outputs/eval/runK_ckpt*_fulltest.json')):
    n = re.search(r'ckpt(\d+)', p).group(1)
    files.append((f'Run K ckpt-{n}', p))
for p in sorted(glob.glob('outputs/eval/runL_ckpt*_fulltest.json')):
    n = re.search(r'ckpt(\d+)', p).group(1)
    files.append((f'Run L ckpt-{n}', p))
for name, path in files:
    if not Path(path).exists():
        lines.append(f'| {name} | (missing) | — | — |')
        continue
    ea, n = element_acc(path)
    if ea is None:
        lines.append(f'| {name} | (no preds) | — | — |')
    else:
        lines.append(f'| {name} | {ea:.4f} | {fm(path):.4f} | {n} |')
Path('FINAL_SUMMARY_runL.md').write_text('\n'.join(lines) + '\n')
print('Wrote FINAL_SUMMARY_runL.md')
" >> "$LOG" 2>&1 || echo "[$(date)] FINAL_SUMMARY_runL write failed" | tee -a "$LOG"

# Append §45 marker
{
    echo ""
    echo "## 45. Run L chain complete"
    echo ""
    echo "_Appended automatically by \`run_l_chain.sh\` at $(date '+%Y-%m-%d %H:%M %Z')._"
    echo ""
    echo "Run L (prior-action history) ablation complete. See \`FINAL_SUMMARY_runL.md\` for headline numbers across baseline, Run I, Run J, Run K, and Run L under element-accuracy."
} >> TRAINING_LOG.md

echo "[$(date)] === RUN L CHAIN DONE — see FINAL_SUMMARY_runL.md ===" | tee -a "$LOG"
