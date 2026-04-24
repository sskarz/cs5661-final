# Mobile UI Agent: Benchmark → SFT + RL Plan

Fine-tune **Gemma 4 E2B** on mobile UI traversal using AndroidControl data,
train on NVIDIA GPU (LoRA QLoRA via Unsloth), deploy on Mac (MLX 8-bit).

## Plan Overview

| Step | Description | Status |
|---|---|---|
| 0 | Baseline eval on AndroidWorld (base model) | ⬜ |
| **1** | **Data: Download + format AndroidControl for SFT** | ✅ Implemented |
| 2 | Unsloth QLoRA fine-tuning on RTX 4090 | ⬜ |
| 3 | RL: DPO → GRPO on AndroidWorld rollouts | ⬜ |

## Step 1 — Data Preparation (✅ Done)

### What it does
- Downloads AndroidControl (~15,283 episodes) from HuggingFace (`smolagents/android-control`)
- Explodes episodes into step-level SFT samples (one image + one action per row)
- Normalizes coordinates per-image using actual PNG dimensions (handles heterogeneous resolutions)
- Writes PNG screenshots to disk + emits JSONL training data ready for Unsloth QLoRA

### Output structure
```
data/androidcontrol/
├── train.jsonl          # ~24,464 step-level samples (goal + step_instruction modes)
├── test.jsonl           # ~6,102 step-level samples (official split)
├── images/              # PNG screenshots: {episode_id:05d}_{step:02d}.png
└── ood_splits.json      # OOD split metadata (task_unseen, app_unseen, etc.)
```

### Usage (local data prep on Mac)
```bash
uv sync                    # install data-prep deps only (no CUDA needed)
uv run python scripts/prepare_androidcontrol.py \
    --output-dir data/androidcontrol

# Smoke test with subset:
uv run python scripts/prepare_androidcontrol.py \
    --output-dir data/androidcontrol_test \
    --max-episodes 100
```

### Usage (NVIDIA training box, full download + OOD splits)
```bash
uv sync --extra train      # install CUDA deps (unsloth, trl, bitsandbytes)
uv run python scripts/prepare_androidcontrol.py \
    --output-dir data/androidcontrol \
    --fetch-ood-splits       # fetch OOD split metadata from s3 + HF join
```

### Key design decisions (see `ACTION_SCHEMA.md` for details)
- **Exact HF repo**: `smolagents/android-control` (no fallback — reproducibility)
- **Step-level, not episode-level**: Each step = one row (avoids blowing context windows)
- **Two granularity modes per step**: "goal" + "step_instruction" (standard eval protocol)
- **Per-image coordinate normalization**: Each PNG's actual W×H used (handles heterogeneous resolutions)
- **Official split preserved**: No invented split ratios — train/test split matches HF card (~12,232 / ~3,051 episodes)

## Full Sequence (from cs5661 doc)
1. ✅ Confirm model = `google/gemma-4-e2b-it` + download AndroidControl data (Step 1)
2. ⬜ Set up AndroidWorld emulator → run eval on base model → record baseline (Step 0)
3. ⬜ SFT with Unsloth QLoRA on RTX 4090 (1–3 epochs, r=16) (Step 2)
4. ⬜ Re-run AndroidWorld eval — measure Δ vs baseline (Step 2)
5. ⬜ Generate rollout pairs → DPO with Unsloth → re-eval (Step 3)
6. ⬜ Merge LoRA, convert to MLX 8-bit for Mac inference (Step 2)
7. ⬜ (Optional) GRPO if DPO plateaus (Step 3)
