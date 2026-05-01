#!/usr/bin/env bash
# Full AndroidWorld sweep with the standard M3A harness on baseline
# Gemma 4 E2B (no LoRA). This produces the "score to beat" reference
# number for FUTURE_WORK.md.
set -e

export JAVA_HOME=$HOME/.jdks/jdk-17.0.13+11
export ANDROID_SDK_ROOT=$HOME/Android/Sdk
export ANDROID_HOME=$ANDROID_SDK_ROOT
export PATH=$JAVA_HOME/bin:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH

REPO=/home/sanskar/Documents/Github/cs5661-final
AW=/home/sanskar/Documents/Github/android_world
LOGDIR=$REPO/outputs/androidworld_logs
mkdir -p "$LOGDIR"

OUT=$HOME/android_world/runs/m3a_baseline_full
mkdir -p "$OUT"
LOG=$LOGDIR/m3a_baseline_full.log

echo "[$(date)] === M3A BASELINE FULL SWEEP start ===" | tee -a "$LOG"
cd "$AW"
./.venv/bin/python run.py \
    --suite_family=android_world \
    --agent_name=m3a_gemma4_baseline \
    --output_path="$OUT" \
    >> "$LOG" 2>&1
echo "[$(date)] === M3A BASELINE FULL SWEEP done ===" | tee -a "$LOG"
