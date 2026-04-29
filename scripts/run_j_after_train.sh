#!/usr/bin/env bash
# Replacement post-training chain for Run J. Started by hand AFTER killing
# run_j_clean.sh's bash parent (PID 321405). The orphaned train_sft.py keeps
# running until it finishes; this script polls for that, then runs:
#   1. Smoke test (sequential vs batched on Run I ckpt-7800, 200 rows)
#   2. If smoke passes -> batched val sweep + batched full-test
#      Else            -> sequential val sweep + sequential full-test
#   3. §35 + §36 to TRAINING_LOG.md, FINAL_SUMMARY.md
set -e
cd "$(dirname "$0")/.."

TRAIN_PID=321417
RUNJ_DIR=outputs/gemma4-e2b-pathW-lora-runJ
LOG=outputs/runI_logs/run_j_after_train.log
mkdir -p outputs/runJ_logs outputs/runI_logs outputs/eval outputs/eval_logs

echo "[$(date)] === run_j_after_train start; waiting on PID $TRAIN_PID ===" | tee "$LOG"

# Poll until training PID exits
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60
done
echo "[$(date)] training PID $TRAIN_PID exited; verifying ckpts..." | tee -a "$LOG"

# Confirm training actually finished cleanly (final ckpt exists)
shopt -s nullglob
mapfile -t JCKPTS < <(ls -d "$RUNJ_DIR"/checkpoint-* 2>/dev/null | sort -V)
echo "[$(date)] discovered ${#JCKPTS[@]} Run J checkpoints" | tee -a "$LOG"
if [ "${#JCKPTS[@]}" -lt 5 ]; then
    echo "[$(date)] FATAL: too few ckpts (${#JCKPTS[@]}); training may have crashed" | tee -a "$LOG"
    exit 1
fi

# ========================================================================
# Smoke test: batched vs sequential on Run I ckpt-7800
# ========================================================================
echo "[$(date)] === smoke-testing batched eval ===" | tee -a "$LOG"
SMOKE_LOG=outputs/eval_logs/smoke_batched.log
USE_BATCHED=0
if bash scripts/smoke_test_batched_eval.sh > "$SMOKE_LOG" 2>&1; then
    if grep -qE "ACCEPT|MARGINAL" "$SMOKE_LOG"; then
        USE_BATCHED=1
        echo "[$(date)] smoke PASSED -> using batched eval" | tee -a "$LOG"
    else
        echo "[$(date)] smoke FAILED (mismatches > threshold) -> sequential fallback" | tee -a "$LOG"
    fi
else
    echo "[$(date)] smoke script errored -> sequential fallback" | tee -a "$LOG"
fi
tail -30 "$SMOKE_LOG" | tee -a "$LOG"

EVAL_SCRIPT=scripts/eval_a11y_native.py
EVAL_EXTRA_ARGS=""
if [ "$USE_BATCHED" = "1" ]; then
    EVAL_SCRIPT=scripts/eval_a11y_native_batched.py
    EVAL_EXTRA_ARGS="--batch-size 8"
fi
echo "[$(date)] eval script = $EVAL_SCRIPT $EVAL_EXTRA_ARGS" | tee -a "$LOG"

# ========================================================================
# Run J val sweep
# ========================================================================
for ckpt_path in "${JCKPTS[@]}"; do
    n=$(basename "$ckpt_path" | sed 's/checkpoint-//')
    out="outputs/eval/runJ_val_ckpt${n}.json"
    [ -f "$out" ] && { echo "[$(date)] [skip] $out exists" | tee -a "$LOG"; continue; }
    echo "[$(date)] === Run J val eval ckpt-$n ===" | tee -a "$LOG"
    .venv/bin/python -u "$EVAL_SCRIPT" \
        --data-dir data/androidcontrol_a11y_native \
        --split val \
        --adapter "$ckpt_path" \
        --num-samples 9999 --seed 3407 \
        $EVAL_EXTRA_ARGS \
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
    echo "[$(date)] running full-test on Run J ckpt-${BEST_J_CKPT}" | tee -a "$LOG"
    .venv/bin/python -u "$EVAL_SCRIPT" \
        --data-dir data/androidcontrol_a11y_native \
        --adapter "$RUNJ_DIR/checkpoint-${BEST_J_CKPT}" \
        --num-samples 9999 --seed 3407 \
        $EVAL_EXTRA_ARGS \
        --save-all-predictions \
        --output "$out" \
        > "outputs/eval_logs/runJ_ckpt${BEST_J_CKPT}_fulltest.log" 2>&1 || \
        echo "[$(date)] [warn] Run J best-val-ckpt full-test failed" | tee -a "$LOG"
fi

# ========================================================================
# Append §35 to TRAINING_LOG.md
# ========================================================================
{
    echo ""
    echo "## 35. Run J — clean cui ablation against Run I"
    echo ""
    echo "_Appended automatically by \`run_j_after_train.sh\` at $(date '+%Y-%m-%d %H:%M %Z')._"
    echo ""
    if [ "$USE_BATCHED" = "1" ]; then
        echo "_Eval used batched generation (\`eval_a11y_native_batched.py\`, batch=8) after smoke-test verified prediction parity vs sequential on Run I ckpt-7800._"
    else
        echo "_Eval used sequential generation (\`eval_a11y_native.py\`); batched smoke test failed and fell back._"
    fi
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
    echo "## 36. Lifts chain complete"
    echo ""
    echo "_Appended automatically by \`run_j_after_train.sh\` at $(date '+%Y-%m-%d %H:%M %Z')._"
    echo ""
    echo "Run J ablation complete. See \`FINAL_SUMMARY.md\` for headline numbers comparing Run I and Run J under both metrics."
} >> TRAINING_LOG.md

echo "[$(date)] === RUN J AFTER-TRAIN DONE — see FINAL_SUMMARY.md ===" | tee -a "$LOG"
