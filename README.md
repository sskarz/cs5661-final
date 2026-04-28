# Mobile UI Agent: Gemma 4 E2B → AndroidControl SFT

Fine-tune **Gemma 4 E2B** as an a11y-aware Android UI agent using AndroidControl,
train on RTX 4090 (Unsloth QLoRA), evaluate on the official AndroidControl test split.

Detailed run-by-run notes live in `TRAINING_LOG.md`. This README is the high-level summary.

## Status (2026-04-28)

| Step | Description | Status |
|---|---|---|
| 1 | Data: download + step-level format AndroidControl for SFT | ✅ |
| 2 | Unsloth QLoRA SFT on RTX 4090 | ✅ Run I clears baseline; Run J in flight |
| 3 | RL (DPO/GRPO on rollouts) | ⬜ deferred |

**Current headline (full test set, 8,217 rows):**

| run | element-acc | tap-radius (legacy) |
|---|---|---|
| Path W zero-shot baseline | 0.394 | 0.526 |
| **Run I ckpt-7800 (val-selected)** | **0.493** | 0.547 |

`+0.099` absolute / `+25%` relative on the metric we now use to gate ship/no-ship.

## Step 1 — Data preparation

`scripts/prepare_androidcontrol.py` downloads `smolagents/android-control` (~15,283
episodes), explodes each episode into step-level SFT rows, normalizes click coordinates
per-image using actual PNG dimensions, and writes JSONL + PNGs ready for Unsloth.

```
data/androidcontrol/
├── train.jsonl      # 159,082 rows (goal + step-instruction modes, 12,232 episodes)
├── val.jsonl        # 686 rows held out from train for ckpt selection
├── test.jsonl       # 39,162 rows (3,051 episodes, official split)
└── images/          # 99,122 PNGs (~48 GB)
```

Parallelized prep (14 workers): full dataset in ~13 min vs. ~108 min serial.

```bash
uv sync --extra train
uv run python scripts/prepare_androidcontrol.py \
    --output-dir data/androidcontrol \
    --fetch-ood-splits --num-workers 14
```

See `ACTION_SCHEMA.md` for the action JSON contract.

## Step 2 — SFT journey (what we tried, what worked)

### Runs A–G: coordinate-regression dead end

The first wave of runs (B, C, E, G, etc.) treated UI taps as a coordinate-regression
problem on raw screenshots. Each run added one more thing the literature said should
help: longer training (Run B → C), coord-aware Huber auxiliary loss (Run E),
discrete coord tokens + class-balanced action weights (Run G), small-rank early-stop
(Run D2). None of them cleared the zero-shot baseline on `tap` accuracy.

Diagnosis: **mode collapse on tap coordinates**. The model learned to output a
plausible mid-screen click regardless of the visual target. Root cause turned out
to be the **frozen multimodal projector** — every LoRA-only run kept Gemma's vision
adapter at its pretrained weights, so no amount of language-side fine-tuning could
teach it to ground taps in the screenshot.

### Run H: projector unlock

Made the multimodal projector trainable alongside LoRA. Tap accuracy improved but
the "coord cliff" (early peak then degradation) persisted, suggesting the regression
formulation itself was the wrong frame.

### Pivot: Path X (Set-of-Marks) + Path W (a11y-native)

Two paths added that sidestep coordinate regression entirely:

- **Path X (SoM)**: render numbered overlays on each interactable element from the
  a11y tree; model picks an element index instead of coordinates.
- **Path W (a11y-native)**: feed the a11y tree directly as text alongside the
  screenshot; model picks an `element_id`. No SoM rendering needed.

Path W matches how production a11y agents actually work and how AndroidControl-paper
baselines report numbers.

### Run I: Path W SFT — first run to clear baseline

1-epoch QLoRA SFT on the a11y-native formulation. **Cleared the zero-shot baseline
by +0.307 absolute on the original tap-radius metric**, and the gain held under the
stricter element-accuracy metric (see below).

## The metric pivot

After Run I we re-scored every saved prediction file with a stricter metric and the
training story flipped:

| metric | best ckpt | shape |
|---|---|---|
| Tap-radius (`pred bbox-center within 0.14 of GT`) | early (ckpt-1500) | apparent peak then "cliff" |
| **Element-accuracy** (`pred element_id == GT element_id`) | late (ckpt-7800) | **monotonic improvement** |

**Tap-radius was forgiving wrong-element picks in clustered UIs** — a row of icons
gives the model a free pass for any icon-in-the-row. As training proceeds the model
commits to specific element IDs, sometimes the wrong one, and tap-radius starts
catching mispicks it previously forgave. The "cliff" was a metric artifact, not
catastrophic forgetting.

**Going forward**: element-accuracy is the validation metric. Tap-radius is reported
alongside as `(legacy)` for backward comparison only. Best-checkpoint selection is
done on a held-out val split (`val.jsonl`, 686 rows), never on the test set.

## Run J — in flight

Same Path W recipe as Run I, with adjustments aimed at the residual failure modes
(`wait` and `open_app` action types stuck near zero accuracy):

- `--lr 2e-5` (was 5e-5) — gentler peak shift
- `--epochs 0.5` — Run I plateaus by step 4500
- `--save-steps 200` (was 300) — finer peak detection
- `--action-weight-scheme cui` — class-balanced loss for rare action types
- `--save-total-limit 30`

Success criterion: best-by-val ckpt's full-test element-accuracy ≥ **0.493** → cui
ablation helped. Else no.

A chained pipeline (`scripts/lifts_chain.sh`) handles: full-test element-accuracy
rescore → Run I val sweep → best-val-ckpt full-test eval → Run J training → Run J
val sweep → final summary doc. Each step appends results to `TRAINING_LOG.md`
automatically.

## Repo layout

```
scripts/
├── prepare_androidcontrol.py        # Step 1 data prep (parallelized)
├── train_sft.py                     # SFT trainer (Unsloth QLoRA)
├── eval_androidcontrol.py           # Eval harness (tap-radius + element metrics)
├── rescore_native_element.py        # Re-score saved predictions with element-accuracy
├── lifts_chain.sh                   # Run I sweep + Run J training pipeline
├── lifts_chain_summary.py           # Aggregator for chain output
└── run_j_clean.sh                   # Run J launcher
data/androidcontrol/                 # train/val/test JSONL + PNGs
outputs/                             # checkpoints + eval JSONs + summaries
TRAINING_LOG.md                      # detailed run-by-run log
ACTION_SCHEMA.md                     # action JSON contract
```

## Key design decisions

- **Exact HF repo**: `smolagents/android-control` (no fallback — reproducibility)
- **Step-level rows**: one image + one action per row (avoids context blowup)
- **Two granularity modes**: `goal` + `step_instruction` (standard eval protocol)
- **Per-image coord normalization**: each PNG's actual W×H (heterogeneous resolutions)
- **Official split preserved**: train/test = HF card (~12,232 / ~3,051 episodes)
- **Val split carved from train**: 686 rows, used for ckpt selection (no test peeking)
- **a11y-native input**: Path W feeds the a11y tree as text; model emits `element_id`
- **Element-accuracy is the headline metric**; tap-radius retained only as legacy
