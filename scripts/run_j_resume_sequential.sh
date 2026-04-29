#!/usr/bin/env bash
# Resume Run J post-training chain after the batched-eval smoke test FAILED
# (7/200 row-level mismatches — likely RoPE-position drift from left-padding).
# Training is already done; 4 val ckpts (600/900/1200/1500) already evaluated.
# This script does the remaining sequential val sweep + best-ckpt full-test +
# TRAINING_LOG/FINAL_SUMMARY bookkeeping.
set -e
cd "$(dirname "$0")/.."

RUNJ_DIR=outputs/gemma4-e2b-pathW-lora-runJ
LOG=outputs/runI_logs/run_j_resume_sequential.log
mkdir -p outputs/runJ_logs outputs/runI_logs outputs/eval outputs/eval_logs

echo "[$(date)] === run_j_resume_sequential start (sequential eval, batched smoke FAILED) ===" | tee "$LOG"

shopt -s nullglob
mapfile -t JCKPTS < <(ls -d "$RUNJ_DIR"/checkpoint-* 2>/dev/null | sort -V)
echo "[$(date)] ${#JCKPTS[@]} Run J checkpoints discovered" | tee -a "$LOG"

# ========================================================================
# Run J val sweep (sequential, skip already-done ckpts)
# ========================================================================
for ckpt_path in "${JCKPTS[@]}"; do
    n=$(basename "$ckpt_path" | sed 's/checkpoint-//')
    out="outputs/eval/runJ_val_ckpt${n}.json"
    if [ -f "$out" ]; then
        echo "[$(date)] [skip] $out exists" | tee -a "$LOG"
        continue
    fi
    echo "[$(date)] === Run J val eval ckpt-$n ===" | tee -a "$LOG"
    .venv/bin/python -u scripts/eval_a11y_native.py \
        --data-dir data/androidcontrol_a11y_native \
        --split val \
        --adapter "$ckpt_path" \
        --num-samples 9999 --seed 3407 \
        --save-all-predictions \
        --output "$out" \
        > "outputs/eval_logs/runJ_val_ckpt${n}.log" 2>&1 || \
        echo "[$(date)] [warn] Run J val eval ckpt-$n failed" | tee -a "$LOG"
done

# ========================================================================
# Pick best Run J ckpt by val element-accuracy
# ========================================================================
BEST_J_CKPT=$(.venv/bin/python -c "
import json, glob, re, sys
sys.path.insert(0, 'scripts')
from rescore_native_element import element_match
best = None; best_acc = -1
for p in sorted(glob.glob('outputs/eval/runJ_val_ckpt*.json')):
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
echo "[$(date)] best Run J val ckpt = $BEST_J_CKPT" | tee -a "$LOG"

# ========================================================================
# Run J full-test on best
# ========================================================================
if [ -n "$BEST_J_CKPT" ]; then
    out="outputs/eval/runJ_ckpt${BEST_J_CKPT}_fulltest.json"
    if [ -f "$out" ]; then
        echo "[$(date)] [skip] $out exists" | tee -a "$LOG"
    else
        echo "[$(date)] running full-test on Run J ckpt-${BEST_J_CKPT}" | tee -a "$LOG"
        .venv/bin/python -u scripts/eval_a11y_native.py \
            --data-dir data/androidcontrol_a11y_native \
            --adapter "$RUNJ_DIR/checkpoint-${BEST_J_CKPT}" \
            --num-samples 9999 --seed 3407 \
            --save-all-predictions \
            --output "$out" \
            > "outputs/eval_logs/runJ_ckpt${BEST_J_CKPT}_fulltest.log" 2>&1 || \
            echo "[$(date)] [warn] Run J best-val-ckpt full-test failed" | tee -a "$LOG"
    fi
fi

# ========================================================================
# Append §35 to TRAINING_LOG.md
# ========================================================================
{
    echo ""
    echo "## 36. Run J — clean cui ablation against Run I"
    echo ""
    echo "_Appended automatically by \`run_j_resume_sequential.sh\` at $(date '+%Y-%m-%d %H:%M %Z')._"
    echo ""
    echo "_Eval used **sequential** generation. The batched eval (\`eval_a11y_native_batched.py\`) was smoke-tested against the proven sequential reference on Run I ckpt-7800 (200 rows, seed 3407) and produced **7/200 row-level mismatches** despite identical aggregate metrics (\`full_match=0.5750\` both). Likely cause: left-padding interacts with Gemma 4's RoPE positions, producing small attention-score deltas that flip greedy decisions on borderline tokens. Sequential is correct; batched is held for future investigation (explicit \`position_ids\` or attention-mask-aware RoPE)._"
    echo ""
    echo "Run J = Run I recipe + class-balanced loss (\`--action-weight-scheme cui\`). Single-variable ablation: same lr (5e-5), same 1.0 epoch budget, same save-steps (300) as Run I; only the cui rebalancing differs. Goal: isolate whether class-balanced loss lifts \`wait\` (Run I 0.000) and \`open_app\` (Run I 0.167) off the floor without harming the strong classes (\`tap\`/\`type\`/\`navigate_back\`)."
    echo ""
    echo "Best Run J ckpt by val element-accuracy: \`ckpt-${BEST_J_CKPT:-NA}\`."
    echo ""
    echo "### Run J val sweep (element-accuracy)"
    echo ""
    echo '```'
    .venv/bin/python -u scripts/rescore_native_element.py outputs/eval/runJ_val_ckpt*.json --label "Run J val" 2>&1
    echo '```'
    if [ -n "$BEST_J_CKPT" ] && [ -f "outputs/eval/runJ_ckpt${BEST_J_CKPT}_fulltest.json" ]; then
        echo ""
        echo "### Run J best-ckpt full-test"
        echo ""
        echo '```'
        .venv/bin/python -c "
import json
m = json.load(open('outputs/eval/runJ_ckpt${BEST_J_CKPT}_fulltest.json'))['metrics']
print(f'full_match (tap-radius): {m[\"full_match\"]:.4f}')
print(f'parse_rate:              {m[\"parse_rate\"]:.4f}')
print(f'tap_oracle_reachability: {m.get(\"tap_oracle_reachability\", 0):.4f}')
print(f'n_samples:               {m[\"num_samples\"]}')
print()
print('per-action-type:')
for k, v in m['per_type'].items():
    print(f'  {k:>16}: n={v[\"n\"]:>4} acc={v[\"accuracy\"]:.3f}')
"
        echo '```'
        echo ""
        .venv/bin/python -u scripts/rescore_native_element.py "outputs/eval/runJ_ckpt${BEST_J_CKPT}_fulltest.json" --label "Run J full-test" 2>&1 | sed 's/^/    /'
    fi
} >> TRAINING_LOG.md

# ========================================================================
# FINAL_SUMMARY.md
# ========================================================================
.venv/bin/python -u scripts/lifts_chain_summary.py \
    --out FINAL_SUMMARY.md \
    > outputs/runI_logs/final_summary.log 2>&1 || \
    echo "[$(date)] FINAL_SUMMARY write failed" | tee -a "$LOG"

# Append §36 marker
{
    echo ""
    echo "## 37. Lifts chain complete"
    echo ""
    echo "_Appended automatically by \`run_j_resume_sequential.sh\` at $(date '+%Y-%m-%d %H:%M %Z')._"
    echo ""
    echo "Run J ablation complete. See \`FINAL_SUMMARY.md\` for headline numbers comparing Run I and Run J under both metrics."
} >> TRAINING_LOG.md

echo "[$(date)] === RUN J RESUME-SEQUENTIAL DONE — see FINAL_SUMMARY.md ===" | tee -a "$LOG"
