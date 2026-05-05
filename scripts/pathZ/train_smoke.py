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
import math
from pathlib import Path

# These get set on the CLI by autoresearch.sh; the file's *defaults* are
# the recipe under test for the current iteration.
DEFAULTS = {
    "max_steps": 400,
    "lr": 2e-4,
    "batch_size": 1,
    "grad_accum": 4,
    "warmup_steps": 12,
    "lora_r": 32,
    "lora_alpha": 64,
    "max_length": 32768,
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
    ap.add_argument("--text-only", action="store_true",
                    help="Drop the image from each row at train time. "
                         "The user prompt already contains the rendered "
                         "UI elements text (a11y), so this exercises "
                         "AndroidLab paper's XML/a11y-only mode.")
    ap.add_argument("--epochs", type=float, default=None,
                    help="Optional epoch target. When set, max_steps is "
                         "computed after loading the dataset as "
                         "ceil(rows * epochs / effective_batch).")
    ap.add_argument("--blank-image-placeholder", action="store_true",
                    help="Legacy opt-in: for non-text-only rows without an "
                         "image path, insert an image content block anyway.")
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

    model_path = Path(args.model)
    adapter_config_path = model_path / "adapter_config.json"
    effective_model_name = args.model
    if adapter_config_path.is_file():
        with adapter_config_path.open() as f:
            adapter_config = json.load(f)
        effective_model_name = adapter_config.get("base_model_name_or_path") or args.model

    print(f"[smoke-train] recipe: max_steps={args.max_steps} "
          f"epochs={args.epochs if args.epochs is not None else 'off'} "
          f"lr={args.lr} "
          f"batch={args.batch_size}x{args.grad_accum} "
          f"lora_r={args.lora_r} alpha={args.lora_alpha} "
          f"projector={'on' if args.train_projector else 'off'}")
    if effective_model_name != args.model:
        print(f"[smoke-train] adapter base_model_name_or_path={effective_model_name}")

    from PIL import Image
    from torch.utils.data import Dataset
    import torch
    from unsloth import FastVisionModel, FastLanguageModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer

    # r53: when training with images, force eager attention. sdpa + bf16
    # + image tokens triggers NaN on first forward pass (numerical overflow
    # in attention scores). Eager is slower but numerically stable.
    fp_kwargs = dict(load_in_4bit=True, use_gradient_checkpointing="unsloth")
    if not args.text_only:
        fp_kwargs["attn_implementation"] = "eager"
    # r60: support text-only models that have no vision tower (e.g. Qwen3-4B).
    # Try FastVisionModel first (handles vision-capable models like Gemma 4
    # and Qwen2.5-VL); fall back to FastLanguageModel for pure text models.
    is_text_only_model = False
    try:
        model, processor = FastVisionModel.from_pretrained(args.model, **fp_kwargs)
    except Exception as e:
        if "image" in str(e).lower() or "vision" in str(e).lower():
            print(f"[smoke-train] FastVisionModel failed ({e}); using FastLanguageModel")
        is_text_only_model = True
        model, tokenizer = FastLanguageModel.from_pretrained(args.model, **fp_kwargs)
        processor = tokenizer
        # FastLanguageModel returns just (model, tokenizer); fake a processor
        # interface that the existing code uses.
        if not hasattr(processor, "tokenizer"):
            processor.tokenizer = tokenizer

    # r53 fix: when training with images (text_only=False), keep vision
    # tower + projector FROZEN. With them unfrozen + LoRA at lr=2e-4,
    # forward pass on first image step explodes to NaN. Standard vision
    # SFT recipe: vision tower frozen, projector frozen, text-side LoRA
    # learns to attend to image embeddings.
    vision_mode = not args.text_only
    if vision_mode:
        modules_to_save = []  # no embedding_projection unfreeze
    else:
        modules_to_save = ["embedding_projection"] if args.train_projector else []
    peft_kwargs = dict(
        finetune_vision_layers=not vision_mode,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=0, bias="none", random_state=args.seed,
        target_modules="all-linear",
    )
    if modules_to_save:
        peft_kwargs["modules_to_save"] = modules_to_save
    is_adapter_path = adapter_config_path.is_file()
    model_is_peft = hasattr(model, "peft_config") or "peft" in type(model).__module__.lower()
    if is_adapter_path or model_is_peft:
        print("[smoke-train] loaded PEFT adapter; continuing without new LoRA wrap "
              f"(adapter_config={is_adapter_path}, model_peft={model_is_peft})")
    else:
        model = FastVisionModel.get_peft_model(model, **peft_kwargs)
    FastVisionModel.for_training(model)
    if vision_mode:
        print(f"[smoke-train] vision mode: vision tower + projector FROZEN")

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
            ut = next(c["text"] for c in row["messages"][0]["content"]
                      if c["type"] == "text")
            at = next(c["text"] for c in row["messages"][1]["content"]
                      if c["type"] == "text")
            user_content = [{"type": "text", "text": ut}]
            images = []
            if not args.text_only:
                # Per-row _image_root takes precedence (lets us mix data
                # sources rooted in different directories — AC + AndroidLab).
                img_root = Path(row.get("_image_root") or self.root)
                img_path = row.get("image")
                if img_path:
                    img = Image.open(img_root / img_path).convert("RGB")
                    # r53: prepend image placeholder in user content so the chat
                    # template emits image markers, but keep the actual PIL image
                    # at row level so the collator passes it to processor.images=.
                    user_content.insert(0, {"type": "image", "image": img})
                    images.append(img)
                elif args.blank_image_placeholder:
                    # Legacy behavior for explicit experiments only. Default
                    # keeps missing-image rows text-only so the processor does
                    # not see image markers without a corresponding image.
                    user_content.insert(0, {"type": "image"})
            sample = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": at},
                    ]},
                ]
            }
            if images:
                sample["images"] = images
            return sample

    dataset = SmokeDataset(args.train_jsonl, args.data_dir)
    train_rows = len(dataset)
    effective_batch = args.batch_size * args.grad_accum
    if effective_batch <= 0:
        raise ValueError("effective batch must be positive")
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        args.max_steps = math.ceil(train_rows * args.epochs / effective_batch)
    estimated_epochs = args.max_steps * effective_batch / train_rows if train_rows else 0.0
    print(f"[smoke-train] train_rows={train_rows} effective_batch={effective_batch} "
          f"requested_epochs={args.epochs if args.epochs is not None else 'off'} "
          f"max_steps={args.max_steps} estimated_actual_epochs={estimated_epochs:.3f}")

    # Mask everything before the assistant turn so loss only fires on
    # Reason+Action tokens. r57: pick markers per model family
    # (Gemma uses <|turn>user/model, Qwen uses <|im_start|>user/assistant).
    if "qwen" in effective_model_name.lower():
        instruction_part = "<|im_start|>user\n"
        response_part = "<|im_start|>assistant\n"
        print(f"[smoke-train] using Qwen ChatML markers for response masking")
    else:
        instruction_part = "<|turn>user\n"
        response_part = "<|turn>model\n"
    # r60: try vision collator (works for Gemma 4 / Qwen2.5-VL / Qwen3.5-4B
    # which carry vision modules). Falls back to trl's response-only
    # collator for pure text models like Qwen3-4B.
    try:
        collator = UnslothVisionDataCollator(
            model, processor,
            train_on_responses_only=True,
            instruction_part=instruction_part,
            response_part=response_part,
        )
    except TypeError as e:
        if "image models" in str(e):
            from trl import DataCollatorForCompletionOnlyLM
            print(f"[smoke-train] text-only model detected; using trl response-only collator")
            collator = DataCollatorForCompletionOnlyLM(
                response_template=response_part,
                tokenizer=processor.tokenizer if hasattr(processor, "tokenizer") else processor,
            )
        else:
            raise

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
    # r53: pass full processor (not just tokenizer) when training with
    # images so the vision pipeline is wired up. With tokenizer-only,
    # the chat template emitted 264 image-marker tokens per row but the
    # input_ids contained 0 image_token_ids → mismatch error.
    trainer_processing_class = (
        processor if not args.text_only else processor.tokenizer
    )
    trainer = SFTTrainer(
        model=model, processing_class=trainer_processing_class,
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
