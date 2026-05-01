#!/usr/bin/env python3
"""Smoke SFT trainer for the pathZ M3A-format recipe.

Loads `data/pathZ/smoke/train.jsonl` (built by prepare_smoke_data.py),
QLoRAs Gemma 4 E2B for `--max-steps` steps, writes a single checkpoint
to `--output-dir/checkpoint-final`.

Designed for fast iteration: defaults sized for ~3-5 min wall time on
one RTX 4090 at batch=1, grad_accum=4, max_steps=200.

The hyperparams here are the *recipe* — autoresearch edits this file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# These get set on the CLI by autoresearch.sh; the file's *defaults* are
# the recipe under test for the current iteration.
DEFAULTS = {
    "max_steps": 300,
    "lr": 2e-4,
    "batch_size": 1,
    "grad_accum": 4,
    "warmup_steps": 15,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_length": 4096,
    "train_projector": True,
    "seed": 3407,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", type=Path,
                    default=Path("data/pathZ/smoke/train.jsonl"))
    ap.add_argument("--data-dir", type=Path,
                    default=Path("data/androidcontrol_a11y_native_v3"))
    ap.add_argument("--model", default="unsloth/gemma-4-E2B-it")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("outputs/pathZ_smoke"))
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_', '-')}",
                            action="store_true", default=v)
            ap.add_argument(f"--no-{k.replace('_', '-')}",
                            dest=k, action="store_false")
        elif isinstance(v, int):
            ap.add_argument(f"--{k.replace('_', '-')}", type=int, default=v)
        elif isinstance(v, float):
            ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    args = ap.parse_args()

    print(f"[smoke-train] recipe: max_steps={args.max_steps} lr={args.lr} "
          f"batch={args.batch_size}x{args.grad_accum} "
          f"lora_r={args.lora_r} alpha={args.lora_alpha} "
          f"projector={'on' if args.train_projector else 'off'}")

    from PIL import Image
    from torch.utils.data import Dataset
    import torch
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer

    model, processor = FastVisionModel.from_pretrained(
        args.model, load_in_4bit=True, use_gradient_checkpointing="unsloth",
    )

    modules_to_save = ["embedding_projection"] if args.train_projector else []
    peft_kwargs = dict(
        finetune_vision_layers=True, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=0, bias="none", random_state=args.seed,
        target_modules="all-linear",
    )
    if modules_to_save:
        peft_kwargs["modules_to_save"] = modules_to_save
    model = FastVisionModel.get_peft_model(model, **peft_kwargs)
    FastVisionModel.for_training(model)

    # Lazy-image dataset, same shape as Run L's trainer.
    class SmokeDataset(Dataset):
        def __init__(self, p: Path, root: Path):
            self.root = root
            with open(p) as f:
                self.rows = [json.loads(line) for line in f]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            row = self.rows[idx]
            img = Image.open(self.root / row["image"]).convert("RGB")
            ut = next(c["text"] for c in row["messages"][0]["content"]
                      if c["type"] == "text")
            at = next(c["text"] for c in row["messages"][1]["content"]
                      if c["type"] == "text")
            return {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": ut},
                        {"type": "image", "image": img},
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": at},
                    ]},
                ]
            }

    dataset = SmokeDataset(args.train_jsonl, args.data_dir)
    print(f"[smoke-train] {len(dataset)} train rows")

    # Mask everything before the assistant turn so loss only fires on
    # Reason+Action tokens. Markers same as Run L (Gemma 4 chat template).
    collator = UnslothVisionDataCollator(
        model, processor,
        train_on_responses_only=True,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    sft_kwargs = dict(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="no",  # we manually save at end
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=args.seed,
        output_dir=str(args.output_dir),
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=args.max_length,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )
    trainer = SFTTrainer(
        model=model, processing_class=processor.tokenizer,
        data_collator=collator,
        train_dataset=dataset,
        args=SFTConfig(**sft_kwargs),
    )
    train_result = trainer.train()

    out = args.output_dir / "checkpoint-final"
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    print(f"[smoke-train] saved {out}")
    print(f"METRIC train_loss_final={train_result.training_loss:.4f}")


if __name__ == "__main__":
    main()
