#!/usr/bin/env bash
# Smoke test: 2 AndroidWorld tasks under the standard M3A harness with
# baseline Gemma 4 E2B (no LoRA). Verifies the wrapper emits parseable
# Reason/Action and the harness completes without crashing.
set -e

export JAVA_HOME=$HOME/.jdks/jdk-17.0.13+11
export ANDROID_SDK_ROOT=$HOME/Android/Sdk
export ANDROID_HOME=$ANDROID_SDK_ROOT
export PATH=$JAVA_HOME/bin:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH

REPO=/home/sanskar/Documents/Github/cs5661-final
AW=/home/sanskar/Documents/Github/android_world
LOGDIR=$REPO/outputs/androidworld_logs
mkdir -p "$LOGDIR"

OUT=$HOME/android_world/runs/m3a_baseline_smoke
mkdir -p "$OUT"
LOG=$LOGDIR/m3a_baseline_smoke.log

cd "$AW"
./.venv/bin/python run.py \
    --suite_family=android_world \
    --agent_name=m3a_gemma4_baseline \
    --tasks=SystemWifiTurnOn,ContactsAddContact \
    --output_path="$OUT" \
    > "$LOG" 2>&1
echo "smoke complete; log: $LOG"
