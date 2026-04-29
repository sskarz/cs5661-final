#!/usr/bin/env bash
# Smoke-test eval_a11y_native_batched.py against eval_a11y_native.py on the same
# ckpt+seed+rows. Verifies the batched implementation gives identical predictions
# to the proven sequential one before we use it for real.
#
# Run this AFTER Run J training finishes (when GPU is free).
set -e
cd "$(dirname "$0")/.."

# Pick the same ckpt the prior chain already evaluated (so we have a sequential
# reference to diff against): Run I ckpt-7800.
ADAPTER=outputs/gemma4-e2b-pathW-lora-runI/checkpoint-7800
N_ROWS=200
SEED=3407

REF=outputs/eval/runI_ckpt7800_smoke_seq.json
TEST=outputs/eval/runI_ckpt7800_smoke_batched.json

echo "=== Sequential reference (200 rows) ==="
.venv/bin/python -u scripts/eval_a11y_native.py \
    --data-dir data/androidcontrol_a11y_native \
    --adapter "$ADAPTER" \
    --num-samples "$N_ROWS" --seed "$SEED" \
    --save-all-predictions \
    --output "$REF"

echo ""
echo "=== Batched (200 rows, batch=8) ==="
.venv/bin/python -u scripts/eval_a11y_native_batched.py \
    --data-dir data/androidcontrol_a11y_native \
    --adapter "$ADAPTER" \
    --num-samples "$N_ROWS" --seed "$SEED" \
    --batch-size 8 \
    --save-all-predictions \
    --output "$TEST"

echo ""
echo "=== Diff predictions ==="
.venv/bin/python -c "
import json
ref = json.load(open('$REF'))
tst = json.load(open('$TEST'))
ref_m = ref['metrics']; tst_m = tst['metrics']
print(f'Sequential: full_match={ref_m[\"full_match\"]:.4f} wall={ref_m[\"wall_seconds\"]/60:.1f} min')
print(f'Batched:    full_match={tst_m[\"full_match\"]:.4f} wall={tst_m[\"wall_seconds\"]/60:.1f} min')
print(f'Speedup:    {ref_m[\"wall_seconds\"]/tst_m[\"wall_seconds\"]:.2f}x')
print()
ref_preds = {(p['episode_id'], p['step_index']): p for p in ref.get('all_predictions', [])}
tst_preds = {(p['episode_id'], p['step_index']): p for p in tst.get('all_predictions', [])}
assert len(ref_preds) > 0, 'no sequential all_predictions saved — re-run with --save-all-predictions'
assert len(tst_preds) > 0, 'no batched all_predictions saved'
# Row-set equality — both scripts must have evaluated the SAME 200 rows for the
# diff to be meaningful. Same shuffle seed + same sort key should guarantee this.
ref_keys = set(ref_preds.keys()); tst_keys = set(tst_preds.keys())
assert ref_keys == tst_keys, (
    f'row sets differ: ref-only={list(ref_keys - tst_keys)[:5]}, '
    f'test-only={list(tst_keys - ref_keys)[:5]}'
)
mismatches = []
for k in ref_preds:
    rp = ref_preds[k]['pred']
    tp = tst_preds.get(k, {}).get('pred', {})
    # Strip resolver-injected coords for comparison
    rp_cmp = {kk: vv for kk, vv in rp.items() if not kk.startswith('_')}
    tp_cmp = {kk: vv for kk, vv in tp.items() if not kk.startswith('_')}
    if rp_cmp != tp_cmp:
        mismatches.append((k, rp_cmp, tp_cmp))
print(f'Mismatched predictions: {len(mismatches)} / {len(ref_preds)}')
for k, r, t in mismatches[:5]:
    print(f'  ep={k[0]} step={k[1]}')
    print(f'    seq:     {r}')
    print(f'    batched: {t}')
# Tightened thresholds: greedy decode should give 0 mismatches if implementation
# is correct. Allow 0-3 for batched-matmul numerical jitter on rare ties.
if len(mismatches) == 0:
    print('  -> ACCEPT batched eval for future runs.')
elif len(mismatches) <= 3:
    print('  -> MARGINAL — likely tie-breaking jitter on a few rows. Inspect printed mismatches.')
else:
    print('  -> FAIL — fall back to sequential.')
"
