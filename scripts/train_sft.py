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
from collections import Counter
from pathlib import Path

from PIL import Image


GEMMA4_MODEL = "unsloth/gemma-4-E2B-it"


def compute_action_weights(
    jsonl_path: Path,
    scheme: str,
    cui_beta: float = 0.999,
    clamp_max: float = 5.0,
) -> dict[str, float]:
    """Per-action-type weight dict, normalized so sum_c f_c * w_c == 1.

    scheme ∈ {none, inverse, sqrt-inverse, cui}. 'none' returns all-1.0.
    Clamps weights to [1/clamp_max, clamp_max].
    """
    if scheme == "none":
        return {}
    counts: Counter = Counter()
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            try:
                txt = row["messages"][1]["content"][0]["text"]
                obj = json.loads(txt)
                a = obj.get("action")
                if isinstance(a, str):
                    counts[a] += 1
            except (KeyError, IndexError, json.JSONDecodeError, TypeError):
                continue
    total = sum(counts.values())
    if total == 0:
        return {}
    raw: dict[str, float] = {}
    for a, n in counts.items():
        f = n / total
        if scheme == "inverse":
            w = 1.0 / f
        elif scheme == "sqrt-inverse":
            w = 1.0 / (f ** 0.5)
        elif scheme == "cui":
            eff = 1.0 - (cui_beta ** n)
            w = (1.0 - cui_beta) / max(eff, 1e-12)
        else:
            raise ValueError(f"unknown scheme: {scheme}")
        raw[a] = w
    # Normalize so count-weighted mean is 1, THEN clamp final values to
    # [1/clamp_max, clamp_max]. (Clamping pre-normalization breaks schemes like
    # 'cui' whose raw values are all sub-1.)
    weighted_mean = sum((counts[a] / total) * w for a, w in raw.items())
    if weighted_mean <= 0:
        return {}
    norm = 1.0 / weighted_mean
    out = {a: w * norm for a, w in raw.items()}
    out = {a: max(1.0 / clamp_max, min(clamp_max, w)) for a, w in out.items()}
    # Re-normalize after clamping so the count-weighted mean stays ≈ 1
    # (preserves overall loss magnitude → existing LR/scheduler tuning stays valid).
    weighted_mean2 = sum((counts[a] / total) * w for a, w in out.items())
    if weighted_mean2 > 0:
        renorm = 1.0 / weighted_mean2
        out = {a: w * renorm for a, w in out.items()}
    return out


def _find_per_layer_embedding(model):
    """Locate Gemma 3/4's embed_tokens_per_layer module if present, else None."""
    for name, mod in model.named_modules():
        if name.endswith("embed_tokens_per_layer") and hasattr(mod, "weight"):
            return mod
    return None


def _add_discrete_loc_tokens(model, tokenizer, grid_size: int, init_strategy: str) -> None:
    """Add <loc_x_K>/<loc_y_K> tokens, resize embeddings, init, verify single-token.

    Must be called BEFORE FastVisionModel.get_peft_model (Unsloth ordering).
    """
    import torch

    new_tokens = [f"<loc_x_{k}>" for k in range(grid_size)] + [
        f"<loc_y_{k}>" for k in range(grid_size)
    ]
    added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    print(f"  Added {added}/{len(new_tokens)} discrete loc tokens to tokenizer")
    if added != len(new_tokens):
        # add_special_tokens returns 0 if tokens already present (re-adding).
        # That's fine when resuming, but on a fresh model warn loudly.
        print(f"  WARN: only {added} new tokens — some loc tokens may already exist.")

    # Verify single-token round-trip on a few sentinels.
    for sentinel in ("<loc_x_0>", f"<loc_x_{grid_size - 1}>", f"<loc_y_{grid_size // 2}>"):
        ids = tokenizer.encode(sentinel, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"[discrete] {sentinel!r} encodes to {len(ids)} tokens; "
                f"expected 1. Tokenizer is BPE-splitting the special token."
            )

    # Pre-resize state for embedding init.
    emb_layer = model.get_input_embeddings()
    pre_size = emb_layer.weight.size(0)
    pretrained_rms = emb_layer.weight[:pre_size].norm(dim=-1).mean().item()
    new_size = len(tokenizer)
    print(f"  Resizing embeddings: {pre_size} → {new_size}  pretrained_rms={pretrained_rms:.4f}")
    model.resize_token_embeddings(new_size)
    emb = model.get_input_embeddings().weight.data  # [V, D]

    # Gemma3/Gemma4-specific: there's a SECOND embedding table 'embed_tokens_per_layer'
    # at model.language_model.embed_tokens_per_layer with shape (vocab, hidden_per_layer).
    # resize_token_embeddings() does NOT touch it. Grow it manually or training crashes
    # with index-out-of-bounds when looking up the new token IDs.
    pl_emb = _find_per_layer_embedding(model)
    if pl_emb is not None:
        pl_pre = pl_emb.weight.size(0)
        pl_dim = pl_emb.weight.size(1)
        if pl_pre < new_size:
            print(f"  Resizing embed_tokens_per_layer: {pl_pre} → {new_size}  (dim={pl_dim})")
            old_w = pl_emb.weight.data
            # Initialize new rows from the existing rows' RMS-normalized mean.
            mean_row = old_w.mean(dim=0)
            mean_rms = mean_row.norm()
            new_rows = mean_row.unsqueeze(0).expand(new_size - pl_pre, pl_dim).clone()
            # Add small jitter so identical-init rows don't get stuck in a degenerate
            # state during early training.
            new_rows = new_rows + torch.randn_like(new_rows) * (mean_rms / pl_dim ** 0.5) * 0.05
            grown = torch.cat([old_w, new_rows.to(old_w.dtype)], dim=0)
            pl_emb.weight = torch.nn.Parameter(grown.contiguous())
            # Some torch.nn.Embedding subclasses cache num_embeddings; sync it.
            if hasattr(pl_emb, "num_embeddings"):
                pl_emb.num_embeddings = new_size
            print(f"  embed_tokens_per_layer now {tuple(pl_emb.weight.shape)}")

    # Initialize new rows.
    new_token_ids = []
    for axis in ("x", "y"):
        for k in range(grid_size):
            tid = tokenizer.convert_tokens_to_ids(f"<loc_{axis}_{k}>")
            new_token_ids.append(tid)
            if init_strategy == "zero":
                vec = torch.zeros_like(emb[0])
            elif init_strategy == "random":
                vec = torch.randn_like(emb[0]) * 0.02
            else:  # subtoken_mean
                digit_ids = tokenizer.encode(str(k), add_special_tokens=False)
                if len(digit_ids) == 0:
                    vec = torch.randn_like(emb[0]) * 0.02
                else:
                    vec = emb[digit_ids].mean(dim=0)
            n = vec.norm()
            if n > 0:
                vec = vec * (pretrained_rms / n)
            emb[tid] = vec.to(emb.dtype)

    new_rms = emb[new_token_ids].norm(dim=-1).mean().item()
    print(f"  New-token RMS after init = {new_rms:.4f}  (target ≈ {pretrained_rms:.4f})")

    # If lm_head is a separate parameter (untied), copy embeddings over so the
    # output projection knows about the new tokens. With tied embeddings this is
    # a no-op (resize already grew lm_head as a view).
    tied = getattr(model.config, "tie_word_embeddings", False)
    print(f"  tie_word_embeddings = {tied}")
    if not tied:
        try:
            lm_head = model.get_output_embeddings()
            if lm_head is not None and lm_head.weight.size(0) >= new_size:
                lm_head.weight.data[new_token_ids] = emb[new_token_ids]
                print(f"  Copied new embeddings into lm_head (untied).")
        except Exception as e:
            print(f"  WARN: could not sync lm_head: {e}")


def _audit_lora_coverage(model, require_projector: bool = True) -> None:
    """Print region-wise trainable-param breakdown after get_peft_model().

    For 5 prior runs we silently shipped with embed_vision.embedding_projection
    completely frozen — Unsloth's all-linear heuristic skips top-level Linears.
    This audit walks every trainable parameter, buckets by region, and aborts
    if the projector bucket is empty when we asked for it to be trained.
    """
    buckets: dict[str, int] = {}
    projector_params = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "vision_tower" in name:
            r = "vision_tower"
        elif "embed_vision" in name:
            r = "embed_vision (projector)"
            projector_params += p.numel()
        elif "embed_audio" in name:
            r = "embed_audio"
        elif "audio_tower" in name:
            r = "audio_tower"
        elif "language_model" in name or "model.layers" in name:
            r = "language_model"
        elif "embed_tokens" in name:
            r = "embed_tokens (modules_to_save)"
        elif "lm_head" in name:
            r = "lm_head (modules_to_save)"
        else:
            r = "other"
        buckets[r] = buckets.get(r, 0) + p.numel()
    print("[lora-audit] trainable param breakdown:")
    for r, n in sorted(buckets.items(), key=lambda x: -x[1]):
        print(f"  {r:38s} {n:>14,}")
    if require_projector and projector_params == 0:
        raise RuntimeError(
            "[lora-audit] FATAL: embed_vision projector has 0 trainable params. "
            "Add 'embedding_projection' to modules_to_save or pass --no-train-projector "
            "if you intentionally want to reproduce the legacy frozen-projector behavior."
        )


def _print_weight_table(weights: dict[str, float], counts_path: Path) -> None:
    counts: Counter = Counter()
    with open(counts_path) as f:
        for line in f:
            row = json.loads(line)
            try:
                obj = json.loads(row["messages"][1]["content"][0]["text"])
                a = obj.get("action")
                if isinstance(a, str):
                    counts[a] += 1
            except Exception:
                continue
    total = sum(counts.values()) or 1
    print(f"  {'action':18s} {'count':>8s} {'freq':>7s} {'weight':>7s}")
    for a in sorted(weights.keys(), key=lambda k: -counts[k]):
        print(f"  {a:18s} {counts[a]:8d} {counts[a]/total:7.3f} {weights[a]:7.3f}")
    wmean = sum(weights[a] * counts[a] / total for a in weights)
    print(f"  count-weighted mean weight = {wmean:.4f} (should be ~1.0)")


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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in output dir.")
    parser.add_argument("--no-response-only", action="store_true",
                        help="Disable train_on_responses_only masking (Run A behavior).")
    parser.add_argument("--coord-loss-weight", type=float, default=0.0,
                        help="Weight for coord-aware Huber aux loss on tap rows. "
                             "0 = off (default; behavior identical to vanilla SFT).")
    parser.add_argument("--coord-huber-delta", type=float, default=0.05,
                        help="Huber transition point in normalized screen distance. "
                             "Only used when --coord-loss-weight > 0.")
    parser.add_argument("--action-weight-scheme",
                        choices=["none", "inverse", "sqrt-inverse", "cui"],
                        default="none",
                        help="Per-row CE rebalancing by action-type frequency. "
                             "'none' = identical to vanilla SFT.")
    parser.add_argument("--action-weight-cui-beta", type=float, default=0.999,
                        help="β for the cui scheme (Cui et al. CVPR 2019).")
    parser.add_argument("--action-weight-clamp-max", type=float, default=5.0,
                        help="Per-class weight clamp range [1/c, c].")
    parser.add_argument("--coord-encoding", choices=["float", "discrete"], default="float",
                        help="discrete: add <loc_x_K>/<loc_y_K> special tokens to "
                             "tokenizer, resize embeddings, train via modules_to_save.")
    parser.add_argument("--grid-size", type=int, default=1024,
                        help="Discrete grid resolution per axis. Must match data prep.")
    parser.add_argument("--init-strategy",
                        choices=["subtoken_mean", "random", "zero"],
                        default="subtoken_mean",
                        help="Embedding init for new <loc_*> tokens (discrete only).")
    parser.add_argument("--train-projector", action="store_true", default=True,
                        help="Train the multimodal projector (embed_vision.embedding_projection). "
                             "Default ON: Unsloth's all-linear filter silently skips this top-level "
                             "Linear, which froze the vision→LM bridge in runs B/C/D2/E/G. "
                             "Pass --no-train-projector to reproduce the broken legacy behavior.")
    parser.add_argument("--no-train-projector", dest="train_projector", action="store_false",
                        help="Disable projector training (legacy/ablation only).")
    args = parser.parse_args()

    # Unsloth disables raw logit return by default (2024.11+) for memory savings.
    # Coord-aware loss AND per-row weighted CE both need raw logits.
    if args.coord_loss_weight > 0.0 or args.action_weight_scheme != "none":
        import os
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

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

    modules_to_save: list[str] = []
    if args.coord_encoding == "discrete":
        _add_discrete_loc_tokens(
            model, processor.tokenizer, args.grid_size, args.init_strategy
        )
        modules_to_save.extend(["embed_tokens", "lm_head"])

    # The Gemma 4 multimodal projector (embed_vision.embedding_projection) is a
    # top-level Linear that Unsloth's FastVisionModel target-module filter silently
    # skips even with target_modules="all-linear" + finetune_vision_layers=True.
    # Force it trainable via modules_to_save — the projector is the bridge between
    # vision features and LM token space, and grounding lives in this layer. Runs
    # B/C/D2/E/G all trained with this layer FROZEN, which is part of why none
    # cleared baseline. Single Linear (~3M params), full fine-tune cost negligible.
    if args.train_projector:
        modules_to_save.append("embedding_projection")

    print(f"Attaching LoRA adapters (r={args.lora_r}, alpha={args.lora_alpha})...")
    peft_kwargs = dict(
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
    if modules_to_save:
        peft_kwargs["modules_to_save"] = modules_to_save
        print(f"  modules_to_save = {modules_to_save}")
    model = FastVisionModel.get_peft_model(model, **peft_kwargs)

    _audit_lora_coverage(model, require_projector=args.train_projector)

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

    collator_kwargs = {}
    if not args.no_response_only:
        # Mask everything before the assistant turn so the loss only fires on
        # JSON-action tokens. Without this, image + user-prompt tokens dominate
        # the average loss and the LoRA under-trains on the response (Run A).
        collator_kwargs["train_on_responses_only"] = True
        # Gemma 4 chat template uses these literal markers (not the Gemma 1/2/3
        # `<start_of_turn>` form). Verified by rendering a sample conversation.
        collator_kwargs["instruction_part"] = "<|turn>user\n"
        collator_kwargs["response_part"] = "<|turn>model\n"
        print("Enabled train_on_responses_only in collator (loss masked to assistant tokens).")

    collator = UnslothVisionDataCollator(model, processor, **collator_kwargs)

    # Compute per-action weights once (used by collator + trainer).
    action_weights: dict[str, float] = {}
    if args.action_weight_scheme != "none":
        action_weights = compute_action_weights(
            train_jsonl,
            scheme=args.action_weight_scheme,
            cui_beta=args.action_weight_cui_beta,
            clamp_max=args.action_weight_clamp_max,
        )
        print(f"Action-type weights ({args.action_weight_scheme}, "
              f"clamp={args.action_weight_clamp_max}):")
        _print_weight_table(action_weights, train_jsonl)

    use_meta_collator = args.coord_loss_weight > 0.0 or bool(action_weights)
    if use_meta_collator:
        # Wrap to attach coord-loss metadata + (optional) action weights.
        from coord_aware_collator import CoordAwareCollator
        collator = CoordAwareCollator(
            collator,
            processor.tokenizer,
            action_weights=action_weights or None,
        )
        if args.coord_loss_weight > 0.0:
            print(
                f"Enabled coord-aware loss: weight={args.coord_loss_weight}, "
                f"huber_delta={args.coord_huber_delta}"
            )
        if action_weights:
            print(f"Enabled per-row action-type weights ({len(action_weights)} classes).")

    sft_kwargs = dict(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=5,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
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

    if args.coord_loss_weight > 0.0 or bool(action_weights):
        from coord_aware_trainer import CoordAwareSFTTrainer
        trainer_cls = CoordAwareSFTTrainer
        trainer_extra = dict(
            coord_loss_weight=args.coord_loss_weight,
            huber_delta=args.coord_huber_delta,
            use_sample_weights=bool(action_weights),
            # Discrete coord encoding skips digit-token validation (digits are still
            # one token each, but coord-aware Huber aux is unsupported in that mode).
            digit_validation=(args.coord_encoding == "float"),
        )
    else:
        trainer_cls = SFTTrainer
        trainer_extra = {}

    trainer = trainer_cls(
        model=model,
        processing_class=processor.tokenizer,
        data_collator=collator,
        train_dataset=dataset,
        args=SFTConfig(**sft_kwargs),
        **trainer_extra,
    )

    # FastVisionModel.for_training(model) (called above) resets UNSLOTH_RETURN_LOGITS=0
    # to skip lm_head computation for memory. Coord-aware loss and weighted CE both
    # need the logits, so re-enable here, just before training starts.
    if args.coord_loss_weight > 0.0 or args.action_weight_scheme != "none":
        import os
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume or None)

    final_dir = args.output_dir / "final"
    print(f"Saving LoRA adapter to {final_dir}")
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    # Belt-and-suspenders: explicitly save tokenizer so the new <loc_*> special
    # tokens round-trip on the eval side.
    if args.coord_encoding == "discrete":
        processor.tokenizer.save_pretrained(str(final_dir))


if __name__ == "__main__":
    main()
