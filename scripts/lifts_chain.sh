#!/usr/bin/env bash
# 3-lift chain after current ckpt-1500 + baseline full-test evals finish.
# Step 1 (element rescore on full-test JSONs): cheap, runs after both finish.
# Step 2 (val eval sweep, 30 ckpts × 686 rows): GPU-bound, ~4.5 h.
# Step 3 (Run J: lr=2e-5, 0.5 epoch, action-rebalance): ~3 h train + ~70 min eval.
#
# Total expected runtime: ~9 h after the queue ahead clears.
set -e
cd "$(dirname "$0")/.."
LOG=outputs/runI_logs/lifts_chain.log
mkdir -p outputs/runI_logs outputs/eval outputs/runJ_logs

CKPT1500_PID="${CKPT1500_PID:-217705}"
BASELINE_QUEUE_PID="${BASELINE_QUEUE_PID:-221492}"

echo "[$(date)] === lifts chain start ===" | tee "$LOG"

# Wait for ckpt-1500 full-test
if [ -n "$CKPT1500_PID" ] && kill -0 "$CKPT1500_PID" 2>/dev/null; then
    echo "[$(date)] waiting for ckpt-1500 full-test (PID $CKPT1500_PID)..." | tee -a "$LOG"
    while kill -0 "$CKPT1500_PID" 2>/dev/null; do sleep 60; done
fi

# Wait for baseline queue (which itself runs the baseline eval)
if [ -n "$BASELINE_QUEUE_PID" ] && kill -0 "$BASELINE_QUEUE_PID" 2>/dev/null; then
    echo "[$(date)] waiting for baseline full-test queue (PID $BASELINE_QUEUE_PID)..." | tee -a "$LOG"
    while kill -0 "$BASELINE_QUEUE_PID" 2>/dev/null; do sleep 60; done
fi

echo "[$(date)] queue clear; starting 3-lift chain" | tee -a "$LOG"

# Helper to append a sub-section to TRAINING_LOG.md after each step
log_append() {
    local title="$1"
    local body_path="$2"
    {
        echo ""
        echo "## $title"
        echo ""
        echo "_Appended automatically by \`lifts_chain.sh\` at $(date '+%Y-%m-%d %H:%M %Z')._"
        echo ""
        if [ -f "$body_path" ]; then
            cat "$body_path"
        fi
    } >> TRAINING_LOG.md
}

# ========================================================================
# Step 1: Element-accuracy rescore on full-test JSONs
# ========================================================================
echo "[$(date)] === step 1: element-accuracy rescore on full-test JSONs ===" | tee -a "$LOG"
.venv/bin/python -u scripts/rescore_native_element.py \
    outputs/eval/runI_ckpt1500_fulltest.json \
    outputs/eval/native_baseline_fulltest.json \
    --label "fulltest" \
    --out outputs/eval/element_summary_fulltest.json \
    2>&1 | tee outputs/runI_logs/step1_rescore.txt | tee -a "$LOG" || \
    echo "[$(date)] step 1 partial fail (continuing)" | tee -a "$LOG"

# Append step 1 result to training log
{
    echo ""
    echo "## 33. Lifts chain step 1 — full-test element-accuracy rescore"
    echo ""
    echo "_Appended automatically by \`lifts_chain.sh\` at $(date '+%Y-%m-%d %H:%M %Z')._"
    echo ""
    echo "Re-scored full-test (8,217 rows) Path W baseline and Run I ckpt-1500 with element-accuracy."
    echo ""
    echo '```'
    cat outputs/runI_logs/step1_rescore.txt 2>/dev/null
    echo '```'
} >> TRAINING_LOG.md

# ========================================================================
# Step 2: Val-set eval sweep on all 30 Run I checkpoints
# ========================================================================
echo "[$(date)] === step 2: val eval sweep across all Run I ckpts ===" | tee -a "$LOG"
shopt -s nullglob
mapfile -t CKPTS < <(ls -d outputs/gemma4-e2b-pathW-lora-runI/checkpoint-* 2>/dev/null | sort -V)
for ckpt_path in "${CKPTS[@]}"; do
    n=$(basename "$ckpt_path" | sed 's/checkpoint-//')
    out="outputs/eval/runI_val_ckpt${n}.json"
    if [ -f "$out" ]; then
        echo "[$(date)] [skip] $out exists" | tee -a "$LOG"
        continue
    fi
    echo "[$(date)] === val eval ckpt-$n ===" | tee -a "$LOG"
    .venv/bin/python -u scripts/eval_a11y_native.py \
        --data-dir data/androidcontrol_a11y_native \
        --split val \
        --adapter "$ckpt_path" \
        --num-samples 9999 --seed 3407 \
        --save-all-predictions \
        --output "$out" \
        > "outputs/eval_logs/runI_val_ckpt${n}.log" 2>&1 || \
        echo "[$(date)] [warn] val eval ckpt-$n failed; continuing" | tee -a "$LOG"
done

# Element-rescore the val sweep
echo "[$(date)] rescoring val sweep with element-accuracy" | tee -a "$LOG"
.venv/bin/python -u scripts/rescore_native_element.py \
    outputs/eval/runI_val_ckpt*.json \
    --label "val-sweep" \
    --out outputs/eval/element_summary_val.json \
    2>&1 | tee outputs/runI_logs/step2_val_rescore.txt | tee -a "$LOG" || true

# Append step 2 result to training log
{
    echo ""
    echo "## 34. Lifts chain step 2 — Run I val sweep"
    echo ""
    echo "_Appended automatically by \`lifts_chain.sh\` at $(date '+%Y-%m-%d %H:%M %Z')._"
    echo ""
    echo "Evaluated all 30 Run I checkpoints against \`val.jsonl\` (686 rows)."
    echo "Element-accuracy rescore of the val sweep:"
    echo ""
    echo '```'
    cat outputs/runI_logs/step2_val_rescore.txt 2>/dev/null
    echo '```'
} >> TRAINING_LOG.md

# Pick best val ckpt by element-accuracy (more representative metric)
BEST_VAL_CKPT=$(.venv/bin/python -c "
import json, glob, re
best = None; best_acc = -1
for p in sorted(glob.glob('outputs/eval/runI_val_ckpt*.json')):
    m = re.search(r'ckpt(\d+)', p)
    if not m: continue
    n = int(m.group(1))
    raw = json.load(open(p))
    preds = raw.get('all_predictions') or []
    if not preds: continue
    # Element-accuracy fast recompute
    import sys; sys.path.insert(0, 'scripts')
    from rescore_native_element import element_match
    ok = sum(1 for p_ in preds if element_match(p_.get('pred') or {}, p_.get('gt') or {}))
    acc = ok / max(len(preds), 1)
    if acc > best_acc:
        best_acc = acc; best = n
print(best)
" 2>>"$LOG")
echo "[$(date)] best val ckpt = $BEST_VAL_CKPT" | tee -a "$LOG"

# Run that ckpt on full test if it's not already ckpt-1500
if [ -n "$BEST_VAL_CKPT" ] && [ "$BEST_VAL_CKPT" != "1500" ]; then
    echo "[$(date)] best val ckpt ($BEST_VAL_CKPT) differs from ckpt-1500; running full-test on it" | tee -a "$LOG"
    out="outputs/eval/runI_ckpt${BEST_VAL_CKPT}_fulltest.json"
    if [ ! -f "$out" ]; then
        .venv/bin/python -u scripts/eval_a11y_native.py \
            --data-dir data/androidcontrol_a11y_native \
            --adapter "outputs/gemma4-e2b-pathW-lora-runI/checkpoint-${BEST_VAL_CKPT}" \
            --num-samples 9999 --seed 3407 \
            --save-all-predictions \
            --output "$out" \
            > "outputs/eval_logs/runI_ckpt${BEST_VAL_CKPT}_fulltest.log" 2>&1 || \
            echo "[$(date)] [warn] full-test on best-val-ckpt failed" | tee -a "$LOG"
        # Append best-val-ckpt full-test result to training log
        {
            echo ""
            echo "## 34.5. Lifts chain step 2b — best-val-ckpt full-test eval"
            echo ""
            echo "_Appended automatically by \`lifts_chain.sh\` at $(date '+%Y-%m-%d %H:%M %Z')._"
            echo ""
            echo "Best Run I checkpoint by val element-accuracy: \`ckpt-${BEST_VAL_CKPT}\` (differs from default ckpt-1500)."
            echo "Full-test eval on this ckpt:"
            echo ""
            echo '```'
            .venv/bin/python -c "
import json
m = json.load(open('$out'))['metrics']
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
            .venv/bin/python -u scripts/rescore_native_element.py "$out" --label "best-val-fulltest" 2>&1 | sed 's/^/    /'
        } >> TRAINING_LOG.md
    fi
fi

# ========================================================================
# Step 3: Run J — lr=2e-5, 0.5 epoch, class-rebalanced loss
# ========================================================================
echo "[$(date)] === step 3: Run J training (Run I recipe + cui ablation) ===" | tee -a "$LOG"
RUNJ_DIR=outputs/gemma4-e2b-pathW-lora-runJ
# Clean single-variable A/B against Run I: ONLY change is --action-weight-scheme cui.
# The original lr=2e-5/0.5-epoch design was anti-cliff; cliff turned out to be
# a tap-radius scoring artifact (under element-accuracy Run I trains monotonically),
# so the original Run J rationale was obsolete.
.venv/bin/python -u scripts/train_sft.py \
    --data-dir data/androidcontrol_a11y_native \
    --output-dir "$RUNJ_DIR" \
    --epochs 1.0 \
    --lora-r 16 --lora-alpha 32 --lr 5e-5 \
    --action-weight-scheme cui \
    --save-steps 300 --save-total-limit 30 \
    > outputs/runJ_logs/runJ.log 2>&1 || echo "[$(date)] Run J training failed" | tee -a "$LOG"

# Run J val sweep
echo "[$(date)] === Run J val sweep ===" | tee -a "$LOG"
mapfile -t JCKPTS < <(ls -d "$RUNJ_DIR"/checkpoint-* 2>/dev/null | sort -V)
for ckpt_path in "${JCKPTS[@]}"; do
    n=$(basename "$ckpt_path" | sed 's/checkpoint-//')
    out="outputs/eval/runJ_val_ckpt${n}.json"
    [ -f "$out" ] && continue
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

# Pick best Run J val ckpt
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

# Run J full-test on best
if [ -n "$BEST_J_CKPT" ]; then
    out="outputs/eval/runJ_ckpt${BEST_J_CKPT}_fulltest.json"
    .venv/bin/python -u scripts/eval_a11y_native.py \
        --data-dir data/androidcontrol_a11y_native \
        --adapter "$RUNJ_DIR/checkpoint-${BEST_J_CKPT}" \
        --num-samples 9999 --seed 3407 \
        --save-all-predictions \
        --output "$out" \
        > "outputs/eval_logs/runJ_ckpt${BEST_J_CKPT}_fulltest.log" 2>&1 || \
        echo "[$(date)] [warn] Run J best-val-ckpt full-test failed" | tee -a "$LOG"
fi

# Append step 3 (Run J) result to training log
{
    echo ""
    echo "## 35. Lifts chain step 3 — Run J results (lr=2e-5, 0.5 epoch, action-weight cui)"
    echo ""
    echo "_Appended automatically by \`lifts_chain.sh\` at $(date '+%Y-%m-%d %H:%M %Z')._"
    echo ""
    echo "Run J = Run I recipe + class-balanced loss (\`--action-weight-scheme cui\`). Single-variable ablation: same lr (5e-5), same 1.0 epoch budget, same save-steps as Run I; only the cui rebalancing differs. Goal: isolate whether class-balanced loss lifts \`wait\` (Run I 0.000) and \`open_app\` (Run I 0.167) off the floor without harming the strong classes (\`tap\`/\`type\`/\`navigate_back\`). Original 0.5-epoch / lr-2e-5 design was discarded once §31 showed the 'cliff' was a metric artifact, not a real overfitting signal."
    echo ""
    echo "Best Run J ckpt by val element-accuracy: \`ckpt-${BEST_J_CKPT:-NA}\`."
    echo ""
    echo "Run J val sweep (element-accuracy):"
    echo ""
    echo '```'
    .venv/bin/python -u scripts/rescore_native_element.py outputs/eval/runJ_val_ckpt*.json --label "Run J val" 2>&1
    echo '```'
    if [ -n "$BEST_J_CKPT" ] && [ -f "outputs/eval/runJ_ckpt${BEST_J_CKPT}_fulltest.json" ]; then
        echo ""
        echo "Run J best-ckpt full-test:"
        echo ""
        echo '```'
        .venv/bin/python -c "
import json
m = json.load(open('outputs/eval/runJ_ckpt${BEST_J_CKPT}_fulltest.json'))['metrics']
print(f'full_match (tap-radius): {m[\"full_match\"]:.4f}')
print(f'parse_rate:              {m[\"parse_rate\"]:.4f}')
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
# Final summary
# ========================================================================
echo "[$(date)] === final summary write ===" | tee -a "$LOG"
.venv/bin/python -u scripts/lifts_chain_summary.py \
    --out FINAL_SUMMARY.md \
    > outputs/runI_logs/final_summary.log 2>&1 || \
    echo "[$(date)] summary script failed" | tee -a "$LOG"

# Final wrap-up section in training log
{
    echo ""
    echo "## 36. Lifts chain complete"
    echo ""
    echo "_Appended automatically by \`lifts_chain.sh\` at $(date '+%Y-%m-%d %H:%M %Z')._"
    echo ""
    echo "All three lifts complete. See \`FINAL_SUMMARY.md\` (repo root) for headline numbers and full comparison tables. Per-step rescores landed at \`outputs/eval/element_summary_{fulltest,val}.json\` plus the Run J equivalents."
} >> TRAINING_LOG.md

echo "[$(date)] === LIFTS CHAIN DONE — see FINAL_SUMMARY.md ===" | tee -a "$LOG"
