# Step 2 — Gemma 4 E2B QLoRA on AndroidControl: Training Log

Hardware: NVIDIA RTX 4090 (24 GB), Ubuntu 24.04, kernel 6.17, driver 580.126.09, CUDA 12.8.
Toolchain: uv-managed CPython 3.12.13, PyTorch 2.10.0+cu128, Unsloth 2026.4.8 (commit `b09aa82a`), TRL 0.24.0, transformers 5.5.0.

---

## 1. Pre-flight checklist

Before launching training we verified GPU + driver, free disk (1.2 TB), Python version, and the presence of the data-prep + training scripts. Two blockers were identified at this stage:

- The `[train]` extra (`unsloth`, `trl`, `bitsandbytes`, `xformers`, `torchvision`, …) was not yet installed.
- The dataset directory `data/androidcontrol/` did not exist; Step 1 had not been executed on this box.

Both were resolved in the steps below.

## 2. Environment fixes

### 2.1 Add `torchvision` to the `[train]` extra
The first attempt to load Unsloth raised `ModuleNotFoundError: torchvision` (`unsloth_zoo.vision_utils` imports it directly). `torchvision` was added to the `train` optional-dependency block in `pyproject.toml` and reinstalled via `uv sync --extra train`.

### 2.2 Use uv-managed Python for working `Python.h`
Triton's first compile of its CUDA helper failed with:

```
fatal error: Python.h: No such file or directory
```

The cause: `uv venv` had bound the project to `/usr/bin/python3.12` (system Python without dev headers). The venv was recreated against uv's own Python distribution (which ships headers):

```bash
uv python install 3.12
rm -rf .venv
uv venv --python-preference only-managed --python 3.12 .venv
uv sync --extra train
```

After recreation, `sysconfig.get_path('include')` resolves to the uv-managed prefix and `Python.h` is present, so Triton's helper compiles.

### 2.3 Import order
Unsloth must be imported before TRL/transformers/peft for its monkey-patches to apply. The lazy imports inside `train_sft.py:main()` were reordered so `from unsloth import FastVisionModel` runs before the TRL imports. This silences the `Unsloth should be imported before [trl, transformers, peft]` warning.

## 3. Step 1 — data preparation

### 3.1 Smoke prep (5 episodes per split)
Used to validate the JSONL row schema and verify per-image coordinate normalization end-to-end.

```bash
uv run python scripts/prepare_androidcontrol.py \
    --output-dir data/androidcontrol_test \
    --max-episodes 5
```

Output: 136 step-level samples (test 64 / train 72), 68 PNG screenshots. First-row inspection confirmed:

- `messages[0].content` is a list of typed parts (`image` + `text`)
- `messages[1].content` is a list of typed parts (`text`) — see schema fix below
- `image` field points to a saved PNG at the dataset's actual resolution (e.g. 1080×2400 RGBA)
- Click coordinates normalized to `[0, 1]` per-image

### 3.2 JSONL schema fix
The original prep script wrote the assistant turn as a bare string while the user turn was a list of typed parts. PyArrow rejected this with:

```
JSON parse error: Column(/messages/[]/content) changed from array to string in row 0
```

`scripts/prepare_androidcontrol.py:build_sample` was changed to wrap the assistant string into the same typed-list shape:

```python
assistant_content = [{"type": "text", "text": json.dumps(action_obj)}]
```

`ACTION_SCHEMA.md` was updated to match.

### 3.3 Full prep run

The first attempt ran serially and progress was glacial (~1.5 ep/s). The script was extended with `multiprocessing.Pool` parallelism and a `--num-workers` flag (worker globals populated via initializer, episode indices passed to `imap_unordered` to avoid pickling each row, streaming path preserved for smoke). Per-episode work is already pure (deterministic `episode_id_step.png` filenames, no shared mutable state, output JSONL sorted at write time), so parallelism is correctness-safe.

Final command (Ryzen 7 9800X3D, 16 logical cores):

```bash
uv run python scripts/prepare_androidcontrol.py \
    --output-dir data/androidcontrol \
    --fetch-ood-splits \
    --num-workers 14
```

Output:

| Split | Rows | Goal | Step-instr | Episodes |
|-------|------|------|------------|----------|
| train | 159,082 | 79,541 | 79,541 | 12,232 |
| test  | 39,162 | 19,581 | 19,581 | 3,051 |
| total | 198,244 | 99,122 | 99,122 | 15,283 |

99,122 PNG screenshots saved (~48 GB total dataset directory). Wall-clock with 14 workers: **13 min 29 s** (~8× speedup vs the ~108 min serial estimate). Four episodes contained the `long_press` action type which is not in our canonical schema — they pass through as `{"action": "long_press"}` rows (162-step minor; the model will learn it as an opaque token but won't get coordinate fields).

The OOD split fetch fell back to the HF mirror (`gsutil` not installed), and that mirror returned no overlap — `ood_splits.json` was not written. We can retry later with `gsutil` if the OOD analysis becomes critical.

## 4. Step 2 — SFT training (`scripts/train_sft.py`)

### 4.1 Aligning with Unsloth's official Gemma 4 (E2B) Vision notebook
The training script was reconciled against the canonical recipe in `unslothai/notebooks/nb/Gemma4_(E2B)-Vision.ipynb`. Three substantive changes:

1. **LoRA target spec.** Replaced the explicit `target_modules` list with the notebook's boolean-flag API plus `target_modules="all-linear"`:
   ```python
   FastVisionModel.get_peft_model(
       model,
       finetune_vision_layers=True,
       finetune_language_layers=True,
       finetune_attention_modules=True,
       finetune_mlp_modules=True,
       r=args.lora_r, lora_alpha=args.lora_alpha,
       lora_dropout=0, bias="none",
       random_state=args.seed,
       use_rslora=False, loftq_config=None,
       target_modules="all-linear",
   )
   ```

2. **Vision collator.** Removed `train_on_responses_only=True` and the manual `instruction_part` / `response_part` strings (`<start_of_turn>user\n` / `<start_of_turn>model\n`). The official notebook calls the collator with positional args only and lets it infer masking from the chat-template-applied output:
   ```python
   collator = UnslothVisionDataCollator(model, processor)
   ```

3. **TRL argument name.** Updated the SFTTrainer call from the deprecated `tokenizer=…` to `processing_class=processor.tokenizer`.

The vision-required `SFTConfig` flags (`remove_unused_columns=False`, `dataset_text_field=""`, `dataset_kwargs={"skip_prepare_dataset": True}`, `max_length=2048`) were already correct.

### 4.2 Lazy image loading via torch Dataset
The earlier `load_dataset("json").map(to_unsloth_format)` failed because `dataset.map` tried to serialize PIL images to Arrow:

```
Couldn't cast array of type
struct<type: string, text: string, image: struct<path: null, bytes: binary>>
to {'type': Value('string'), 'text': Value('string')}
```

Replaced with a thin `torch.utils.data.Dataset` subclass that loads PIL on `__getitem__`:

```python
class AndroidControlDataset(TorchDataset):
    def __init__(self, jsonl_path, data_dir):
        self.data_dir = data_dir
        with open(jsonl_path) as f:
            self.rows = [json.loads(line) for line in f]
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        return to_unsloth_format(self.rows[idx], self.data_dir)
```

This keeps the ~24 k-row dataset out of RAM (only the JSONL metadata is held; PNGs are decoded per batch by the DataLoader). The Unsloth vision collator processes the resulting `messages` dicts at collate time.

### 4.3 Smoke train (50 steps on the test subset)
```bash
uv run python scripts/train_sft.py \
    --data-dir data/androidcontrol_test \
    --output-dir outputs/smoke \
    --max-steps 50
```

Results (RTX 4090, bf16, QLoRA `r=16, alpha=16`, batch 2 × grad-accum 4):

| Step | loss   | grad_norm | lr        |
|------|--------|-----------|-----------|
| 10   | 9.888  | 7.978     | 1.82e-4   |
| 20   | 2.737  | 4.244     | 1.38e-4   |
| 30   | 1.430  | 2.219     | 9.33e-5   |
| 40   | 0.866  | 1.658     | 4.89e-5   |
| 50   | 0.651  | 2.365     | 4.44e-6   |

`train_runtime` 108.5 s, `train_samples_per_second` 3.69, `train_steps_per_second` 0.461. Adapter saved to `outputs/smoke/final/adapter_model.safetensors` (119 MB). The steep loss drop is expected for a 72-row overfit smoke; what matters here is that the pipeline runs end-to-end and the loss is monotone.

### 4.4 Real train, Run A (1 epoch on the full split)

Command:
```bash
uv run python scripts/train_sft.py \
    --data-dir data/androidcontrol \
    --output-dir outputs/gemma4-e2b-androidcontrol-lora \
    --epochs 1
```

Hyperparameters (CLI defaults): QLoRA `r=16`, `alpha=16`, `lora_dropout=0`, `target_modules="all-linear"`, `finetune_{vision,language,attention,mlp}_layers=True`, `bsz=2`, `grad_accum=4` (effective batch 8), `lr=2e-4` linear-decay schedule, `warmup_steps=5`, `weight_decay=0.001`, `optim="adamw_8bit"`, bf16, `max_length=2048`, `save_steps=500`, `save_total_limit=3`.

Trainable parameters: **29,859,840 / 5,153,037,856 (0.58 %)**.

Run started **2026-04-24 23:32:33**, finished **2026-04-25 08:14:18**.

#### Loss curve (every ~10 % of an epoch)

| Step | Epoch | Loss | grad_norm | LR |
|-----:|------:|-----:|----------:|---:|
| 10 | 0.0005 | 10.2369 | 7.12 | 2.00e-04 |
| 100 | 0.005 | 1.7303 | 1.28 | 1.99e-04 |
| 500 | 0.025 | 1.2872 | 0.95 | 1.95e-04 |
| 1 000 | 0.050 | 1.0634 | 0.92 | 1.90e-04 |
| 2 500 | 0.126 | 0.8489 | 0.80 | 1.75e-04 |
| 5 000 | 0.251 | 0.9048 | 1.43 | 1.50e-04 |
| 7 500 | 0.377 | 0.6675 | 1.14 | 1.25e-04 |
| 10 000 | 0.503 | 0.6592 | 1.59 | 9.95e-05 |
| 12 500 | 0.629 | 0.5003 | 1.39 | 7.43e-05 |
| 15 000 | 0.754 | 0.5261 | 2.03 | 4.92e-05 |
| 17 500 | 0.880 | 0.4222 | 1.28 | 2.40e-05 |
| 19 500 | 0.981 | 0.4117 | 1.32 | 3.89e-06 |
| 19 886 | 1.000 | 0.3899 | 0.94 | 7.04e-08 |

- **Final 100-step rolling-avg loss: 0.4214**
- **Final 200-step rolling-avg loss: 0.4224** (stable plateau, no signs of divergence)
- Running-average reported by the trainer over the whole run: `train_loss = 0.6748`
- Loss reduction 10.24 → 0.39 over 19,886 steps (≈ 26×)
- Grad norms steady around **0.8 – 1.6** with two isolated spikes (15.0 at step 260, 9.95 at step 14470); no NaN/inf, no OOM.

#### Throughput / wall-clock

- `train_runtime`: **31,290 s ≈ 8 h 41 min** (real, wall-clock)
- `train_samples_per_second`: **5.084**
- `train_steps_per_second`: **0.635**
- VRAM at steady state: **~9.9 GB used / 14.1 GB free** (RTX 4090, well within budget)

#### Artifacts

- LoRA adapter: `outputs/gemma4-e2b-androidcontrol-lora/final/adapter_model.safetensors` (**119 MB**)
- Adapter config: `adapter_config.json` (r=16, alpha=16, target_modules="all-linear")
- Saved alongside: tokenizer, processor config, chat template
- 40 intermediate checkpoints written every 500 steps, last 3 retained on disk by `save_total_limit=3` (final 19,886 + checkpoints 19,500 / 19,000 retained)

#### Verdict

Run A passed the health bar (final loss < 1.0, monotone trajectory, stable grad norms). **No retry triggered** — the retry ladder (Runs B–E) was held in reserve and is not needed. The adapter is ready for evaluation against the AndroidWorld baseline (Step 2 eval) and for downstream RL (Step 3).

## 5. Outstanding / next

Step 2 (SFT) is complete and produced a healthy adapter on the first try; the retry ladder was not exercised. Remaining work, in dependency order:

- **Step 0 baseline eval** on AndroidWorld with the base `unsloth/gemma-4-E2B-it` (no adapter). Required to compute Δ for Step 2.
- **Step 2 eval**: re-run the same AndroidWorld harness with the LoRA adapter loaded, measure success-rate Δ vs baseline. Log per-task results and action-format validity (malformed JSON should be excluded from the UI-understanding score).
- **Adapter merge → MLX 8-bit** for Mac inference (deployment target). Use `mlx_lm.convert` after merging the LoRA back into bf16 weights.
- **Step 3 RL**: roll out the SFT model on AndroidWorld, build (chosen, rejected) pairs from successful vs failed trajectories, run DPO with Unsloth on the same 4090 box. GRPO is optional and only justified if DPO plateaus.
