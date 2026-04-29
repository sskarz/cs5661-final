#!/usr/bin/env bash
# Run K — two-stage decoding (CoCo-Agent CAP-style) head-to-head against Run I.
#
# Single-variable change vs Run I: the JSON output schema. Same lr=5e-5,
# r=16/alpha=32, 1.0 epoch, save-steps=300, save-total-limit=30, no cui,
# no coord loss, no oversampling. Inherits the §35.5 dataloader speedup.
#
# Pipeline:
#   1. Train (~4h 13min on RTX 4090, 9132 steps, dataloader workers active)
#   2. Sequential val sweep across 30 ckpts (~3.5h, 686 val rows / ckpt)
#   3. Pick best ckpt by val element-accuracy
#   4. Best-ckpt full-test (~80 min, 8217 test rows)
#   5. Auto-append §40 + §41 to TRAINING_LOG.md, write FINAL_SUMMARY_runK.md
set -e
cd "$(dirname "$0")/.."

RUNK_DIR=outputs/gemma4-e2b-pathW-lora-runK
LOG=outputs/runI_logs/run_k_chain.log
mkdir -p outputs/runK_logs outputs/runI_logs outputs/eval outputs/eval_logs

echo "[$(date)] === run_k_chain start (CAP-style two-stage decoding) ===" | tee "$LOG"

# Wipe any partial runK output from a prior aborted launch
rm -rf "$RUNK_DIR" 2>/dev/null || true
rm -f outputs/runK_logs/runK.log 2>/dev/null || true

# ========================================================================
# Step 1: Run K training
# ========================================================================
echo "[$(date)] launching Run K training..." | tee -a "$LOG"
.venv/bin/python -u scripts/train_sft.py \
    --data-dir data/androidcontrol_a11y_native_v2 \
    --output-dir "$RUNK_DIR" \
    --epochs 1.0 \
    --lora-r 16 --lora-alpha 32 --lr 5e-5 \
    --save-steps 300 --save-total-limit 30 \
    > outputs/runK_logs/runK.log 2>&1 || \
    { echo "[$(date)] Run K training FAILED" | tee -a "$LOG"; exit 1; }

echo "[$(date)] Run K training done; starting val sweep" | tee -a "$LOG"

# ========================================================================
# Step 2: Run K val sweep (sequential)
# ========================================================================
shopt -s nullglob
mapfile -t KCKPTS < <(ls -d "$RUNK_DIR"/checkpoint-* 2>/dev/null | sort -V)
echo "[$(date)] discovered ${#KCKPTS[@]} Run K checkpoints" | tee -a "$LOG"
for ckpt_path in "${KCKPTS[@]}"; do
    n=$(basename "$ckpt_path" | sed 's/checkpoint-//')
    out="outputs/eval/runK_val_ckpt${n}.json"
    [ -f "$out" ] && { echo "[$(date)] [skip] $out exists" | tee -a "$LOG"; continue; }
    echo "[$(date)] === Run K val eval ckpt-$n ===" | tee -a "$LOG"
    .venv/bin/python -u scripts/eval_a11y_native.py \
        --data-dir data/androidcontrol_a11y_native_v2 \
        --split val \
        --adapter "$ckpt_path" \
        --num-samples 9999 --seed 3407 \
        --save-all-predictions \
        --output "$out" \
        > "outputs/eval_logs/runK_val_ckpt${n}.log" 2>&1 || \
        echo "[$(date)] [warn] Run K val eval ckpt-$n failed" | tee -a "$LOG"
done

# ========================================================================
# Step 3: Pick best Run K ckpt by val element-accuracy
# ========================================================================
BEST_K_CKPT=$(.venv/bin/python -c "
import json, glob, re, sys
sys.path.insert(0, 'scripts')
from rescore_native_element import element_match
best = None; best_acc = -1
for p in sorted(glob.glob('outputs/eval/runK_val_ckpt*.json')):
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
echo "[$(date)] best Run K val ckpt = $BEST_K_CKPT" | tee -a "$LOG"

# ========================================================================
# Step 4: Run K full-test on best
# ========================================================================
if [ -n "$BEST_K_CKPT" ]; then
    out="outputs/eval/runK_ckpt${BEST_K_CKPT}_fulltest.json"
    if [ -f "$out" ]; then
        echo "[$(date)] [skip] $out exists" | tee -a "$LOG"
    else
        echo "[$(date)] running full-test on Run K ckpt-${BEST_K_CKPT}" | tee -a "$LOG"
        .venv/bin/python -u scripts/eval_a11y_native.py \
            --data-dir data/androidcontrol_a11y_native_v2 \
            --adapter "$RUNK_DIR/checkpoint-${BEST_K_CKPT}" \
            --num-samples 9999 --seed 3407 \
            --save-all-predictions \
            --output "$out" \
            > "outputs/eval_logs/runK_ckpt${BEST_K_CKPT}_fulltest.log" 2>&1 || \
            echo "[$(date)] [warn] Run K best-val-ckpt full-test failed" | tee -a "$LOG"
    fi
fi

# ========================================================================
# Step 5: Append §40 + §41 to TRAINING_LOG.md
# ========================================================================
{
    echo ""
    echo "## 40. Run K — two-stage decoding (CAP-style) head-to-head against Run I"
    echo ""
    echo "_Appended automatically by \`run_k_chain.sh\` at $(date '+%Y-%m-%d %H:%M %Z')._"
    echo ""
    echo "Run K = Run I recipe + v2 hierarchical action schema (\`{\"action_type\": ..., \"action_args\": {...}}\`)."
    echo "Single-variable A/B: same lr (5e-5), r=16/α=32, 1.0 epoch, save-steps=300, save-total-limit=30; only the JSON output schema differs."
    echo "Inherits §35.5 dataloader speedup (8 workers, prefetch 4, persistent, pin_memory)."
    echo "See §39 for design rationale + literature backing (Ma et al. ACL Findings 2024, arXiv:2402.11941)."
    echo ""
    echo "Best Run K ckpt by val element-accuracy: \`ckpt-${BEST_K_CKPT:-NA}\`."
    echo ""
    echo "### Run K val sweep (element-accuracy)"
    echo ""
    echo '```'
    .venv/bin/python -u scripts/rescore_native_element.py outputs/eval/runK_val_ckpt*.json --label "Run K val" 2>&1
    echo '```'
    if [ -n "$BEST_K_CKPT" ] && [ -f "outputs/eval/runK_ckpt${BEST_K_CKPT}_fulltest.json" ]; then
        echo ""
        echo "### Run K best-ckpt full-test"
        echo ""
        echo '```'
        .venv/bin/python -c "
import json
m = json.load(open('outputs/eval/runK_ckpt${BEST_K_CKPT}_fulltest.json'))['metrics']
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
        .venv/bin/python -u scripts/rescore_native_element.py "outputs/eval/runK_ckpt${BEST_K_CKPT}_fulltest.json" --label "Run K full-test" 2>&1 | sed 's/^/    /'
    fi
} >> TRAINING_LOG.md

# ========================================================================
# Write FINAL_SUMMARY_runK.md (Run I + Run J + Run K headline)
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

lines = ['# Final summary — Run I + Run J + Run K', '', '## Headline (full test, 8217 rows)', '']
lines.append('| run | element-acc | tap-radius (legacy) | n |')
lines.append('|---|---|---|---|')
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
for name, path in files:
    if not Path(path).exists():
        lines.append(f'| {name} | (missing) | — | — |')
        continue
    ea, n = element_acc(path)
    if ea is None:
        lines.append(f'| {name} | (no preds) | — | — |')
    else:
        lines.append(f'| {name} | {ea:.4f} | {fm(path):.4f} | {n} |')
Path('FINAL_SUMMARY_runK.md').write_text('\n'.join(lines) + '\n')
print('Wrote FINAL_SUMMARY_runK.md')
" >> "$LOG" 2>&1 || echo "[$(date)] FINAL_SUMMARY_runK write failed" | tee -a "$LOG"

# Append §41 marker
{
    echo ""
    echo "## 41. Run K chain complete"
    echo ""
    echo "_Appended automatically by \`run_k_chain.sh\` at $(date '+%Y-%m-%d %H:%M %Z')._"
    echo ""
    echo "Run K (CAP-style two-stage decoding) ablation complete. See \`FINAL_SUMMARY_runK.md\` for headline numbers across baseline, Run I, Run J, and Run K under element-accuracy."
} >> TRAINING_LOG.md

echo "[$(date)] === RUN K CHAIN DONE — see FINAL_SUMMARY_runK.md ===" | tee -a "$LOG"
