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
from pathlib import Path

from datasets import load_dataset
from PIL import Image


GEMMA4_MODEL = "unsloth/gemma-4-E2B-it"


def to_unsloth_format(row: dict, data_dir: Path) -> dict:
    """Convert one prep-script JSONL row into the shape Unsloth's vision SFT expects."""
    img_path = data_dir / row["image"]
    img = Image.open(img_path).convert("RGB")

    user_text = next(
        c["text"] for c in row["messages"][0]["content"] if c["type"] == "text"
    )
    assistant_text = row["messages"][1]["content"]

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

    # Lazy imports — these require the [train] extra (CUDA-only)
    import torch
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    print(f"Loading {args.model} (4-bit QLoRA)...")
    model, processor = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    print(f"Attaching LoRA adapters (r={args.lora_r}, alpha={args.lora_alpha})...")
    model = FastVisionModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    print(f"Loading dataset from {train_jsonl}...")
    raw = load_dataset("json", data_files=str(train_jsonl), split="train")
    dataset = raw.map(
        lambda row: to_unsloth_format(row, args.data_dir),
        remove_columns=raw.column_names,
    )
    print(f"  Train rows: {len(dataset)}")

    FastVisionModel.for_training(model)

    # Gemma 4 chat-template markers — verify against the Gemma 4 vision notebook
    # before a long run. If they're wrong, the loss masking is wrong.
    collator = UnslothVisionDataCollator(
        model=model,
        processor=processor,
        train_on_responses_only=True,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

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
        tokenizer=processor.tokenizer,
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
