#!/usr/bin/env bash
# Run the AndroidWorld smoke slice (10 curated tasks) under the standard
# M3A harness. Used by autoresearch.sh as the trajectory-faithful eval.
#
# Usage:
#   ADAPTER=outputs/pathZ_smoke/checkpoint-final ./run_aw_smoke_slice.sh
#   ADAPTER="" ./run_aw_smoke_slice.sh   # baseline (no LoRA)
#
# Outputs `METRIC aw_success_rate=<pct>` for autoresearch.

set -euo pipefail

export JAVA_HOME=$HOME/.jdks/jdk-17.0.13+11
export ANDROID_SDK_ROOT=$HOME/Android/Sdk
export ANDROID_HOME=$ANDROID_SDK_ROOT
export PATH=$JAVA_HOME/bin:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH

REPO=/home/sanskar/Documents/Github/cs5661-final
AW=/home/sanskar/Documents/Github/android_world
TASKS_FILE="$REPO/scripts/aw_smoke_tasks.txt"
LOGDIR="$REPO/outputs/androidworld_logs"
mkdir -p "$LOGDIR"

ADAPTER=${ADAPTER:-}
# Resolve to absolute path so the `cd "$AW"` below doesn't break it.
if [[ -n "$ADAPTER" && "$ADAPTER" != /* ]]; then
  ADAPTER="$REPO/$ADAPTER"
fi
TAG=${TAG:-aw_smoke}
LOG="$LOGDIR/${TAG}.log"
OUT_DIR="$HOME/android_world/runs/${TAG}"
rm -rf "$OUT_DIR" && mkdir -p "$OUT_DIR"

# Build comma-sep task list (skip blank/comment lines).
TASKS=$(grep -v '^[[:space:]]*\(#\|$\)' "$TASKS_FILE" | paste -sd,)
echo "[aw-smoke] tasks: $TASKS" | tee "$LOG"
echo "[aw-smoke] adapter: ${ADAPTER:-<none, baseline>}" | tee -a "$LOG"

cd "$AW"
if [[ -z "$ADAPTER" ]]; then
  AGENT=m3a_gemma4_baseline
  EXTRA_ARGS=()
else
  AGENT=m3a_gemma4_lora
  EXTRA_ARGS=(--adapter_path="$ADAPTER")
fi

./.venv/bin/python run.py \
    --suite_family=android_world \
    --agent_name="$AGENT" \
    "${EXTRA_ARGS[@]}" \
    --tasks="$TASKS" \
    --output_path="$OUT_DIR" \
    2>&1 | tee -a "$LOG"

# Aggregate: count success markers.
N_TOTAL=$(grep -v '^[[:space:]]*\(#\|$\)' "$TASKS_FILE" | wc -l)
N_OK=$(grep -c "Task Successful ✅" "$LOG" || true)
RATE=$(python3 -c "print(f'{100.0*$N_OK/$N_TOTAL:.2f}')")
echo "[aw-smoke] $N_OK / $N_TOTAL tasks succeeded"
echo "METRIC aw_success_rate=$RATE"
echo "METRIC aw_n_total=$N_TOTAL"
echo "METRIC aw_n_ok=$N_OK"
