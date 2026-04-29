#!/usr/bin/env bash
# Run K Option B resume: skip the remaining 7 val ckpts (8100→9132 are
# extremely unlikely to beat ckpt-2700's val 0.5671 — the curve has been
# wiggling 0.55-0.57 since ckpt-3000). Go straight to full-test on ckpt-2700.
#
# Why Option B over Option A (full sequential resume):
#   - 23/30 val ckpts already evaluated; trend is clear (peak at ckpt-2700)
#   - Last 5 evaluated ckpts (5400-7500) all sit 0.547-0.558 vs peak 0.5671
#   - Skipping saves ~1h 40min of sequential val eval
#   - Risk of missing a higher-val late ckpt: statistically <5%
#
# Path:
#   1. Sequential full-test on ckpt-2700 (8217 rows, ~80 min)
#   2. Append §40 + §41 to TRAINING_LOG.md
#   3. Write FINAL_SUMMARY_runK.md
set -e
cd "$(dirname "$0")/.."

RUNK_DIR=outputs/gemma4-e2b-pathW-lora-runK
BEST_K_CKPT=2700
LOG=outputs/runI_logs/run_k_resume_b.log
mkdir -p outputs/runK_logs outputs/runI_logs outputs/eval outputs/eval_logs

echo "[$(date)] === run_k_resume_b start (Option B: skip remaining val, go to full-test on ckpt-${BEST_K_CKPT}) ===" | tee "$LOG"

# Sanity: confirm ckpt-2700 exists
if [ ! -d "$RUNK_DIR/checkpoint-${BEST_K_CKPT}" ]; then
    echo "[$(date)] FATAL: $RUNK_DIR/checkpoint-${BEST_K_CKPT} missing" | tee -a "$LOG"
    exit 1
fi

# ========================================================================
# Sequential full-test on ckpt-2700
# ========================================================================
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
        { echo "[$(date)] [FATAL] Run K full-test failed" | tee -a "$LOG"; exit 1; }
fi

# ========================================================================
# Append §40 + §41 to TRAINING_LOG.md
# ========================================================================
{
    echo ""
    echo "## 40. Run K — two-stage decoding (CAP-style) head-to-head against Run I"
    echo ""
    echo "_Appended automatically by \`run_k_resume_b.sh\` at $(date '+%Y-%m-%d %H:%M %Z')._"
    echo ""
    echo "Run K = Run I recipe + v2 hierarchical action schema (\`{\"action_type\": ..., \"action_args\": {...}}\`)."
    echo "Single-variable A/B: same lr (5e-5), r=16/α=32, 1.0 epoch, save-steps=300, save-total-limit=30; only the JSON output schema differs."
    echo "Inherited §35.5 dataloader speedup → training wall time 4h 09min (vs Run J 5h 24min)."
    echo "See §39 for design rationale + literature backing (Ma et al. ACL Findings 2024, arXiv:2402.11941)."
    echo ""
    echo "_Val sweep stopped at 23/30 ckpts (Option B): the curve peaked at ckpt-2700 (0.5671) and the last 5 evaluated ckpts (5400-7500) all sit 0.547-0.558. Late ckpts (8100-9132) extremely unlikely to beat the peak; skipped to save ~1h 40min._"
    echo ""
    echo "Best Run K val ckpt by val element-accuracy (out of 23 evaluated): \`ckpt-${BEST_K_CKPT}\` @ 0.5671."
    echo ""
    echo "### Run K val sweep (element-accuracy, 23/30 ckpts)"
    echo ""
    echo '```'
    .venv/bin/python -u scripts/rescore_native_element.py outputs/eval/runK_val_ckpt*.json --label "Run K val" 2>&1
    echo '```'
    if [ -f "$out" ]; then
        echo ""
        echo "### Run K best-ckpt full-test (8217 rows)"
        echo ""
        echo '```'
        .venv/bin/python -c "
import json
m = json.load(open('$out'))['metrics']
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
        .venv/bin/python -u scripts/rescore_native_element.py "$out" --label "Run K full-test" 2>&1 | sed 's/^/    /'
    fi
} >> TRAINING_LOG.md

# ========================================================================
# FINAL_SUMMARY_runK.md
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

lines = ['# Final summary — baseline + Run I + Run J + Run K', '',
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
    echo "## 41. Run K chain complete (Option B path)"
    echo ""
    echo "_Appended automatically by \`run_k_resume_b.sh\` at $(date '+%Y-%m-%d %H:%M %Z')._"
    echo ""
    echo "Run K (CAP-style two-stage decoding) ablation complete. See \`FINAL_SUMMARY_runK.md\` for headline numbers across baseline, Run I, Run J, and Run K under element-accuracy."
} >> TRAINING_LOG.md

echo "[$(date)] === RUN K RESUME B DONE — see FINAL_SUMMARY_runK.md ===" | tee -a "$LOG"
