#!/usr/bin/env bash
set -u

REPO=/home/sanskar/Documents/Github/cs5661-final
AW=/home/sanskar/Documents/Github/android_world
ADAPTER="$REPO/outputs/r62_qwen35_r57b_plus_genesis_vision_e2/checkpoint-final"
OUT_ROOT="$HOME/android_world/runs/r62_qwen35_vision_cont_aw116"
LOGDIR="$REPO/outputs/androidworld_logs"
LOG="$LOGDIR/r62_qwen35_vision_cont_aw116.log"
SUMMARY="$LOGDIR/r62_qwen35_vision_cont_aw116.summary.txt"
ROWS_MD="$LOGDIR/r62_qwen35_vision_cont_aw116.rows.md"
ROWS_JSON="$LOGDIR/r62_qwen35_vision_cont_aw116.rows.json"

export JAVA_HOME=$HOME/.jdks/jdk-17.0.13+11
export ANDROID_SDK_ROOT=$HOME/Android/Sdk
export ANDROID_HOME=$ANDROID_SDK_ROOT
export PATH=$JAVA_HOME/bin:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH

mkdir -p "$OUT_ROOT" "$LOGDIR"
{
  echo "[$(date)] === r62 Qwen3.5 r57b+Genesis vision continuation AW-116 start ==="
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
AW_RUN_ROOT="$OUT_ROOT" AW_LOG="$LOG" RUN_EXIT="$RUN_EXIT" ROWS_MD="$ROWS_MD" ROWS_JSON="$ROWS_JSON" "$AW/.venv/bin/python" - <<'PY' | tee "$SUMMARY"
import gzip, json, os, pickle, time
from pathlib import Path

repo = Path('/home/sanskar/Documents/Github/cs5661-final')
out_root = Path(os.environ['AW_RUN_ROOT'])
log = Path(os.environ['AW_LOG'])
run_exit = int(os.environ.get('RUN_EXIT', '999'))
rows_md = Path(os.environ['ROWS_MD'])
rows_json = Path(os.environ['ROWS_JSON'])
run_dirs = sorted([p for p in out_root.glob('run_*') if p.is_dir()], key=lambda p: p.stat().st_mtime)
run_dir = run_dirs[-1] if run_dirs else out_root
records = []
for p in sorted(run_dir.glob('*.pkl.gz')):
    try:
        with gzip.open(p, 'rb') as f:
            obj = pickle.load(f)
        vals = obj if isinstance(obj, list) else [obj]
        for x in vals:
            if isinstance(x, dict):
                x = dict(x)
                x['_file'] = p.name
                records.append(x)
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
row_objs = []
for i, r in enumerate(records):
    row_objs.append({
        'idx': i,
        'task_template': r.get('task_template'),
        'goal': r.get('goal'),
        'is_successful': float(r.get('is_successful') or 0),
        'episode_length': r.get('episode_length'),
        'exception_info': str(r.get('exception_info')) if r.get('exception_info') is not None else None,
        'file': r.get('_file'),
    })
rows_json.write_text(json.dumps(row_objs, indent=2, sort_keys=True))
md = ['# r62 AW-116 rows', '', f'Run dir: `{run_dir}`', '', '| # | task | ok | steps | goal | exception |', '|---:|---|---:|---:|---|---|']
for x in row_objs:
    goal = str(x['goal'] or '').replace('|','\\|')[:180]
    exc = str(x['exception_info'] or '').replace('|','\\|')[:80]
    md.append(f"| {x['idx']} | {x['task_template']} | {int(x['is_successful']>0)} | {x['episode_length']} | {goal} | {exc} |")
rows_md.write_text('\n'.join(md)+'\n')

metrics = {
    'aw_n_ok': n_ok,
    'aw_n_total': total,
    'aw_n_attempted': att,
    'aw_success_rate': round(rate, 2),
    'aw_exceptions': len(exceptions),
    'successes': ','.join(successes),
    'train_loss_final': 0.3321,
    'base_model': 'unsloth/Qwen3.5-4B',
    'base_adapter': str(repo / 'outputs/r57b/checkpoint-final'),
    'adapter': str(repo / 'outputs/r62_qwen35_r57b_plus_genesis_vision_e2/checkpoint-final'),
    'dataset': 'data/pathZ/genesis_vision_rebuild/train.expanded.jsonl',
    'train_rows': 107,
    'vision_training': True,
    'epochs': 2.0,
    'run_exit': run_exit,
    'run_dir': str(run_dir),
    'rows_md': str(rows_md),
    'rows_json': str(rows_json),
}
status = 'keep' if att and rate >= 8.62 else ('discard' if att else 'crash')
desc = f"r62 Qwen3.5 r57b+Genesis low-epoch vision continuation. AW-116={n_ok}/{att}={rate:.2f}% attempted ({total} records, {len(exceptions)} exceptions). Successes: {', '.join(successes) if successes else 'none'}."
entry = {
    'run': '62-aw116',
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

## r62 — Qwen3.5 r57b + low-epoch Genesis vision continuation full AW-116

- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}
- Training: continued from `outputs/r57b/checkpoint-final` on 107 aligned Genesis vision rows for 2 epochs / 54 steps; final train loss `0.3321`.
- AW-20 smoke before full: 3/20 = 15.00%; user requested full AW-116 anyway.
- AW-116 command: `run.py --suite_family=android_world --agent_name=m3a_gemma4_lora_a11y_vision --adapter_path={metrics['adapter']} --output_path={out_root}`
- AW-116 result: **{n_ok}/{att} = {rate:.2f}%** attempted; records={total}; exceptions={len(exceptions)}; run_exit={run_exit}.
- Successes: {', '.join(successes) if successes else 'none'}.
- Verdict: {status.upper()}. {'Beats/ties prior 8.62% AW-116 best.' if rate >= 8.62 else 'Does not beat prior 8.62% AW-116 best.'}
- Row artifacts: `{rows_md}`, `{rows_json}`.
- Run artifacts: `{run_dir}`, `{log}`.
'''
for name in ['TRAINING_LOG.md', 'experiments/worklog.md']:
    p = repo / name
    if p.exists():
        with p.open('a') as f:
            f.write(section)

print(json.dumps(entry, indent=2, sort_keys=True))
print(f'METRIC aw_success_rate={rate:.2f}')
print(f'METRIC aw_n_ok={n_ok}')
print(f'METRIC aw_n_attempted={att}')
print(f'METRIC aw_n_total={total}')
PY

cat "$SUMMARY" >> "$LOG"
echo "[$(date)] === r62 AW-116 summarize/log update done ===" | tee -a "$LOG"
exit "$RUN_EXIT"
