# AndroidWorld setup notes (cs5661-final)

Setup performed: 2026-04-29 by agent for sanskar1016pro@outlook.com on Linux 6.17.0-22-generic.

The cs5661-final repo was NOT modified other than this one file. AndroidWorld lives in a sibling directory with its own venv.

## 1. Source docs used

- Repo README (canonical install instructions):
  https://github.com/google-research/android_world/blob/main/README.md
  Repo last-pushed 2026-04-09; updated 2026-04-29 (active project).
- Repo `requirements.txt` (Python deps): https://github.com/google-research/android_world/blob/main/requirements.txt
- Repo `pyproject.toml`: https://github.com/google-research/android_world/blob/main/pyproject.toml
- `minimal_task_runner.py` smoke-test script:
  https://github.com/google-research/android_world/blob/main/minimal_task_runner.py
- Project website: https://google-research.github.io/android_world/ (high-level only; no Linux SDK steps).
- Android `avdmanager` CLI reference: https://developer.android.com/tools/avdmanager
- Android cmdline-tools download index: https://developer.android.com/studio#command-line-tools-only

The README uses the macOS SDK path; Linux uses `~/Android/Sdk` (also referenced in the auto-discovery code in `minimal_task_runner.py`).

## 2. Pinned versions / paths

| Component | Value |
|---|---|
| AndroidWorld repo path | `/home/sanskar/Documents/Github/android_world` |
| AndroidWorld git SHA | `d9c569f764b3a5629321858de03ff653d0f24056` (HEAD on `main`, 2026-02-24) |
| Python (AndroidWorld venv) | 3.11.8 (uv-managed; pinned by README) |
| AndroidWorld venv | `/home/sanskar/Documents/Github/android_world/.venv` |
| JDK | Temurin OpenJDK 17.0.13+11 at `~/.jdks/jdk-17.0.13+11` (user-local; sdkmanager rejects 1.8) |
| Android SDK root | `~/Android/Sdk` (= `$ANDROID_SDK_ROOT` = `$ANDROID_HOME`) |
| cmdline-tools | `13114758_latest` at `~/Android/Sdk/cmdline-tools/latest/` |
| platform-tools (adb) | installed at `~/Android/Sdk/platform-tools/adb` |
| emulator | installed at `~/Android/Sdk/emulator/emulator` |
| platform | `platforms;android-33` |
| system image | `system-images;android-33;google_apis;x86_64` (Tiramisu, API 33) |
| AVD name | `AndroidWorldAvd` (Pixel 6 device profile) |
| AVD path | `~/.android/avd/AndroidWorldAvd.avd` |
| Disk used by SDK | ~9.3 GB (`~/Android/Sdk`) |
| Disk free after install | 740 GB on `/` |
| KVM | `/dev/kvm` present (group `kvm`); user is in group, so HW accel works |

## 3. Environment variables (must be set per-shell)

```bash
export JAVA_HOME=$HOME/.jdks/jdk-17.0.13+11
export ANDROID_SDK_ROOT=$HOME/Android/Sdk
export ANDROID_HOME=$ANDROID_SDK_ROOT
export PATH=$JAVA_HOME/bin:$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH
```

Add to `~/.bashrc` if you want this persistent (NOT done automatically).

## 4. What's installed vs. what was already present

Already present:
- KVM (`/dev/kvm`, user in `kvm` group)
- `uv` 0.x (used to create the venv)
- `python3.12` (system); `python3.11.8` was downloaded by uv
- 750 GB free disk

Installed by this setup:
- Temurin JDK 17.0.13 (user-local, no sudo) — sdkmanager requires JDK 17+
- Android cmdline-tools 13114758 (user-local)
- Android platform-tools (adb, fastboot, etc.)
- Android emulator binary
- Android platform 33
- Android 13 (Tiramisu) Google APIs x86_64 system image
- AndroidWorld Python deps in `/home/sanskar/Documents/Github/android_world/.venv` (`pip install -r requirements.txt` then `python setup.py install`)

Note: `requirements.txt` does not pin `setuptools`. setuptools 82+ removed the bundled `pkg_resources`, which `setup.py` imports. This setup downgraded to `setuptools==79.0.1` inside the venv — keep that pin if you ever recreate the env.

## 5. What's MISSING / TODO

| Item | Required for | Suggested fix (commands NOT run) |
|---|---|---|
| `ffmpeg` | `pydub` audio handling (warning only at import time; required for some tasks) | `sudo apt-get update && sudo apt-get install -y ffmpeg` |
| `OPENAI_API_KEY` | Default `M3A` agent in `minimal_task_runner.py` (uses `gpt-4-turbo-2024-04-09`). NOT needed for our use case (Gemma 4 E2B + LoRA via custom agent), but the smoke test as-written will fail without it. | `export OPENAI_API_KEY=...` OR replace the agent in `minimal_task_runner.py` with our offline model. |
| `GCP_API_KEY` | optional, only if using Gemini agents | `export GCP_API_KEY=...` |
| Persistent shell exports | convenience | append the env block in section 3 to `~/.bashrc` |

For the Gemma 4 E2B + LoRA project goal: the OpenAI key block is irrelevant. We will need to add a custom agent class under `android_world/agents/` (subclass `EnvironmentInteractingAgent`, implement `step`) and register it in `run.py`'s `_get_agent` switch. Only AndroidWorld plumbing + emulator are needed from this setup; model inference is local.

## 6. Two-step "how to run it"

Open a fresh terminal. Make sure no other process is hogging CPU/GPU/RAM (the model training in cs5661-final can coexist; the emulator uses CPU+RAM, not the GPU).

### Step 1 - launch the emulator (long-running, leave open in terminal A)

```bash
export JAVA_HOME=$HOME/.jdks/jdk-17.0.13+11
export ANDROID_SDK_ROOT=$HOME/Android/Sdk
export ANDROID_HOME=$ANDROID_SDK_ROOT
export PATH=$JAVA_HOME/bin:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH

emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554
```

(For headless server use add `-no-window -no-audio`.) Wait until `adb devices` shows `emulator-5554  device`.

### Step 2 - run a benchmark task (terminal B)

```bash
cd /home/sanskar/Documents/Github/android_world
source .venv/bin/activate
export ANDROID_SDK_ROOT=$HOME/Android/Sdk
export ANDROID_HOME=$ANDROID_SDK_ROOT
export PATH=$ANDROID_SDK_ROOT/platform-tools:$PATH
# Optional - only needed by the default M3A agent:
# export OPENAI_API_KEY=...

# Smoke test (single task):
python minimal_task_runner.py --task=ContactsAddContact

# Full suite or subset (use after creating our own agent):
python run.py \
  --suite_family=android_world \
  --agent_name=t3a_gpt4 \
  --perform_emulator_setup \
  --tasks=ContactsAddContact,ClockStopWatchRunning
```

The very first time `run.py` is invoked you must pass `--perform_emulator_setup` (installs third-party APKs into the emulator). This is one-time per AVD.

## 7. Verification steps actually performed

- [x] Cloned repo to `/home/sanskar/Documents/Github/android_world` (SHA `d9c569f`).
- [x] Created `.venv` with Python 3.11.8 via `uv venv --python 3.11.8`.
- [x] `pip install -r requirements.txt` -> success.
- [x] `python setup.py install` -> success (after pinning `setuptools<80`).
- [x] `python -c "import android_world"` -> success.
- [x] `python minimal_task_runner.py --helpfull` -> flags listed; default `--adb_path` resolves to `~/Android/Sdk/platform-tools/adb` (auto-discovery works).
- [x] AVD `AndroidWorldAvd` created with Pixel 6 + `system-images;android-33;google_apis;x86_64`. `avdmanager list avd` confirms.
- [x] `adb` and `emulator` binaries present and executable.
- [ ] **Emulator was NOT started** (would block the GPU-adjacent training run for I/O and is not required for setup verification).
- [ ] End-to-end smoke task NOT executed (depends on running emulator + agent API key).

## 8. Resource notes

- Disk: ~9.3 GB consumed by `~/Android/Sdk` (system image is the biggest single item). 740 GB still free on `/`.
- RAM: emulator allocates ~2 GB by default (Pixel 6 profile). AndroidWorld README states "lightweight footprint (2 GB memory, 8 GB disk)".
- GPU: AndroidWorld emulator uses host GL via the host's OpenGL stack; it does NOT compete with CUDA/Triton. The training process at PID 525492 is unaffected.
- CPU: the emulator pegs 1-2 cores while idle and more during task execution.
- Network: required for `git clone`, `pip install`, `sdkmanager` downloads (already done). At runtime: only if a task needs it.

## 9. Open issues / TODOs for the user

1. Decide whether to install `ffmpeg` system-wide (sudo apt) or add a user-local static build to `PATH`. Right now `pydub` warns but still imports.
2. Implement custom Gemma-4-E2B-LoRA agent at `android_world/agents/gemma4_lora_agent.py`, subclass `EnvironmentInteractingAgent`, implement `step`. Wire into `run.py:_get_agent`. Then run `run.py --agent_name=gemma4_lora_agent ...`.
3. The AndroidWorld README mentions task step limits were updated 2024-11-18 to ~2x human-average; spreadsheet linked from README has per-task budgets. Use those when scoring.
4. Optional: persist the env exports to `~/.bashrc` (intentionally not done by this setup).
5. Optional: build the experimental Docker image (`docker build -t android_world:latest .`) if you ever want a sealed environment; not needed for native runs.
6. The default `M3A` agent in `minimal_task_runner.py` is hard-coded to `gpt-4-turbo-2024-04-09`. Edit the `model_name` constant if you want to point the smoke test at a cheaper or local model.
