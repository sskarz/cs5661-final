#!/usr/bin/env bash
set -u

REPO=/home/sanskar/Documents/Github/cs5661-final
AW=/home/sanskar/Documents/Github/android_world
ADAPTER="$REPO/outputs/pathZ_genesis_expanded_vision_r61_qwen35/checkpoint-final"
OUT_ROOT="$HOME/android_world/runs/r61_qwen35_genesis_vision_aw116"
LOGDIR="$REPO/outputs/androidworld_logs"
LOG="$LOGDIR/r61_qwen35_genesis_vision_aw116.log"
SUMMARY="$LOGDIR/r61_qwen35_genesis_vision_aw116.summary.txt"

export JAVA_HOME=$HOME/.jdks/jdk-17.0.13+11
export ANDROID_SDK_ROOT=$HOME/Android/Sdk
export ANDROID_HOME=$ANDROID_SDK_ROOT
export PATH=$JAVA_HOME/bin:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH

mkdir -p "$OUT_ROOT" "$LOGDIR"
{
  echo "[$(date)] === r61 Qwen3.5 Genesis vision AW-116 start ==="
  echo "adapter=$ADAPTER"
  echo "out_root=$OUT_ROOT"
  adb devices || true
} | tee "$LOG"

cd "$AW"
./.venv/bin/python run.py \
  --suite_family=android_world \
  --agent_name=m3a_gemma4_lora_a11y_vision \
  --adapter_path="$ADAPTER" \
  --output_path="$OUT_ROOT" \
  >> "$LOG" 2>&1
RUN_EXIT=$?

echo "[$(date)] === run.py exit=$RUN_EXIT ===" | tee -a "$LOG"

cd "$REPO"
AW_RUN_ROOT="$OUT_ROOT" AW_LOG="$LOG" RUN_EXIT="$RUN_EXIT" "$AW/.venv/bin/python" - <<'PY' | tee "$SUMMARY"
import gzip, json, os, pickle, time
from pathlib import Path

repo = Path('/home/sanskar/Documents/Github/cs5661-final')
out_root = Path(os.environ['AW_RUN_ROOT'])
log = Path(os.environ['AW_LOG'])
run_exit = int(os.environ.get('RUN_EXIT', '999'))
run_dirs = sorted([p for p in out_root.glob('run_*') if p.is_dir()], key=lambda p: p.stat().st_mtime)
run_dir = run_dirs[-1] if run_dirs else out_root
records = []
for p in sorted(run_dir.glob('*.pkl.gz')):
    try:
        with gzip.open(p, 'rb') as f:
            obj = pickle.load(f)
        if isinstance(obj, list):
            records.extend([x for x in obj if isinstance(x, dict)])
        elif isinstance(obj, dict):
            records.append(obj)
    except Exception as e:
        print(f'WARN failed to read {p}: {e}')

attempted = [r for r in records if r.get('exception_info') is None]
ok = [r for r in attempted if float(r.get('is_successful') or 0) > 0]
exceptions = [r for r in records if r.get('exception_info') is not None]
total = len(records)
att = len(attempted)
n_ok = len(ok)
rate = (100.0*n_ok/att) if att else 0.0
successes = sorted({r.get('task_template','?') for r in ok})
last_loss = '0.1078'
metrics = {
    'aw_n_ok': n_ok,
    'aw_n_total': total,
    'aw_n_attempted': att,
    'aw_success_rate': round(rate, 2),
    'aw_exceptions': len(exceptions),
    'successes': ','.join(successes),
    'train_loss_final': float(last_loss),
    'base_model': 'unsloth/Qwen3.5-4B',
    'adapter': str(repo / 'outputs/pathZ_genesis_expanded_vision_r61_qwen35/checkpoint-final'),
    'dataset': 'data/pathZ/genesis_vision_rebuild/train.expanded.jsonl',
    'train_rows': 107,
    'vision_training': True,
    'run_exit': run_exit,
    'run_dir': str(run_dir),
}
status = 'keep' if att and rate >= 8.62 else ('discard' if att else 'crash')
desc = f"r61 Qwen3.5-4B Genesis vision SFT on expanded filtered Genesis-only aligned screenshots. AW-116={n_ok}/{att}={rate:.2f}% attempted ({total} records, {len(exceptions)} exceptions). Successes: {', '.join(successes) if successes else 'none'}."
entry = {
    'run': '61-aw116',
    'commit': 'wip',
    'metric': round(rate, 2),
    'metrics': metrics,
    'status': status,
    'description': desc,
    'timestamp': int(time.time()),
    'segment': 4,
}
with (repo / 'autoresearch.jsonl').open('a') as f:
    f.write(json.dumps(entry, sort_keys=True) + '\n')

section = f'''
\n## r61 — Genesis vision SFT with Qwen3.5-4B + full AW-116\n\n- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n- Dataset cleanup: added `scripts/pathZ/filter_genesis_vision.py`; expanded Genesis-only filter kept 20/95 trajectories and 107 screenshot-aligned train rows. No AndroidControl/AndroidLab mixing.\n- Training command: `uv run python scripts/pathZ/train_smoke.py --train-jsonl data/pathZ/genesis_vision_rebuild/train.expanded.jsonl --data-dir data/pathZ/genesis_vision_rebuild/screenshots --model unsloth/Qwen3.5-4B --output-dir outputs/pathZ_genesis_expanded_vision_r61_qwen35 --max-steps 200 --lr 1e-4`\n- Training result: `train_loss_final=0.1078`; adapter `outputs/pathZ_genesis_expanded_vision_r61_qwen35/checkpoint-final`.\n- AW-116 command: `run.py --suite_family=android_world --agent_name=m3a_gemma4_lora_a11y_vision --adapter_path={metrics['adapter']} --output_path={out_root}`\n- AW-116 result: **{n_ok}/{att} = {rate:.2f}%** attempted; records={total}; exceptions={len(exceptions)}; run_exit={run_exit}.\n- Successes: {', '.join(successes) if successes else 'none'}.\n- Verdict: {status.upper()}. {'Beats prior 8.62% AW-116 best.' if rate > 8.62 else 'Does not beat prior 8.62% AW-116 best.'}\n- Artifacts: `{run_dir}`, `{log}`.\n'''
with (repo / 'TRAINING_LOG.md').open('a') as f:
    f.write(section)

print(json.dumps(entry, indent=2, sort_keys=True))
print(f'METRIC aw_success_rate={rate:.2f}')
print(f'METRIC aw_n_ok={n_ok}')
print(f'METRIC aw_n_attempted={att}')
print(f'METRIC aw_n_total={total}')
PY

cat "$SUMMARY" >> "$LOG"
echo "[$(date)] === r61 AW-116 summarize/log update done ===" | tee -a "$LOG"
exit "$RUN_EXIT"
