#!/usr/bin/env python3
"""
Step 2 — SFT: QLoRA fine-tune Gemma 4 E2B on AndroidControl using Unsloth.

NVIDIA GPU only (uses bitsandbytes 4-bit + Unsloth kernels). Run on the 4090
training box, not on Mac.

Setup (NVIDIA box):
    uv sync --extra train

Usage:
    uv run python scripts/train_sft.py \\
        --data-dir data/androidcontrol \\
        --output-dir outputs/gemma4-e2b-androidcontrol-lora \\
        --epochs 1

The data dir is expected to contain train.jsonl + test.jsonl + images/ as
produced by scripts/prepare_androidcontrol.py.
"""

import argparse
import json
from pathlib import Path

from PIL import Image


GEMMA4_MODEL = "unsloth/gemma-4-E2B-it"


def to_unsloth_format(row: dict, data_dir: Path) -> dict:
    """Convert one prep-script JSONL row into the shape Unsloth's vision SFT expects."""
    img_path = data_dir / row["image"]
    img = Image.open(img_path).convert("RGB")

    user_text = next(
        c["text"] for c in row["messages"][0]["content"] if c["type"] == "text"
    )
    assistant_text = next(
        c["text"] for c in row["messages"][1]["content"] if c["type"] == "text"
    )

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "image": img},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/androidcontrol"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gemma4-e2b-androidcontrol-lora"))
    parser.add_argument("--model", default=GEMMA4_MODEL)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override epochs with a fixed step count (smoke testing).")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in output dir.")
    args = parser.parse_args()

    train_jsonl = args.data_dir / "train.jsonl"
    if not train_jsonl.exists():
        raise SystemExit(
            f"Missing {train_jsonl}. Run scripts/prepare_androidcontrol.py first."
        )

    # Lazy imports — these require the [train] extra (CUDA-only).
    # Unsloth must be imported before trl/transformers/peft so its patches apply.
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    import torch
    from trl import SFTConfig, SFTTrainer

    print(f"Loading {args.model} (4-bit QLoRA)...")
    model, processor = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    print(f"Attaching LoRA adapters (r={args.lora_r}, alpha={args.lora_alpha})...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
    )

    print(f"Loading dataset from {train_jsonl}...")
    # PIL images can't round-trip through Arrow, so we keep the dataset as a
    # plain torch Dataset that loads images lazily on __getitem__. The Unsloth
    # vision collator handles tokenization + image encoding at collate time.
    from torch.utils.data import Dataset as TorchDataset

    class AndroidControlDataset(TorchDataset):
        def __init__(self, jsonl_path: Path, data_dir: Path):
            self.data_dir = data_dir
            with open(jsonl_path) as f:
                self.rows = [json.loads(line) for line in f]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return to_unsloth_format(self.rows[idx], self.data_dir)

    dataset = AndroidControlDataset(train_jsonl, args.data_dir)
    print(f"  Train rows: {len(dataset)}")

    FastVisionModel.for_training(model)

    collator = UnslothVisionDataCollator(model, processor)

    sft_kwargs = dict(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=5,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=args.seed,
        output_dir=str(args.output_dir),
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),

        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=args.max_length,
    )
    if args.max_steps is not None:
        sft_kwargs["max_steps"] = args.max_steps
    else:
        sft_kwargs["num_train_epochs"] = args.epochs

    trainer = SFTTrainer(
        model=model,
        processing_class=processor.tokenizer,
        data_collator=collator,
        train_dataset=dataset,
        args=SFTConfig(**sft_kwargs),
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume or None)

    final_dir = args.output_dir / "final"
    print(f"Saving LoRA adapter to {final_dir}")
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))


if __name__ == "__main__":
    main()
