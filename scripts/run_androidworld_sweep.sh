#!/usr/bin/env bash
# Sequential AndroidWorld full-sweep: baseline first, then LoRA (Run L ckpt-2100).
# All paths are absolute — script changes cwd into android_world before invoking run.py.
set -e

export JAVA_HOME=$HOME/.jdks/jdk-17.0.13+11
export ANDROID_SDK_ROOT=$HOME/Android/Sdk
export ANDROID_HOME=$ANDROID_SDK_ROOT
export PATH=$JAVA_HOME/bin:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH

REPO=/home/sanskar/Documents/Github/cs5661-final
AW=/home/sanskar/Documents/Github/android_world
LOGDIR=$REPO/outputs/androidworld_logs
mkdir -p "$LOGDIR"
SWEEP_LOG=$LOGDIR/sweep.log

# Baseline first
BASE_LOG=$LOGDIR/baseline_full_v2.log
BASE_OUT=$HOME/android_world/runs/baseline_full_v2
mkdir -p "$BASE_OUT"
echo "[$(date)] === BASELINE FULL SWEEP start ===" | tee -a "$SWEEP_LOG"
cd "$AW"
./.venv/bin/python run.py \
    --suite_family=android_world \
    --agent_name=gemma4_baseline \
    --output_path="$BASE_OUT" \
    > "$BASE_LOG" 2>&1 || \
    echo "[$(date)] baseline sweep exited non-zero" | tee -a "$SWEEP_LOG"
echo "[$(date)] === BASELINE FULL SWEEP done ===" | tee -a "$SWEEP_LOG"

# LoRA
LORA_LOG=$LOGDIR/lora_full_v2.log
LORA_OUT=$HOME/android_world/runs/lora_full_v2
mkdir -p "$LORA_OUT"
echo "[$(date)] === LORA FULL SWEEP start ===" | tee -a "$SWEEP_LOG"
cd "$AW"
./.venv/bin/python run.py \
    --suite_family=android_world \
    --agent_name=gemma4_lora \
    --output_path="$LORA_OUT" \
    > "$LORA_LOG" 2>&1 || \
    echo "[$(date)] lora sweep exited non-zero" | tee -a "$SWEEP_LOG"
echo "[$(date)] === LORA FULL SWEEP done ===" | tee -a "$SWEEP_LOG"

echo "[$(date)] === BOTH SWEEPS COMPLETE ===" | tee -a "$SWEEP_LOG"
