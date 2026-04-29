#!/usr/bin/env python3
"""Batched eval for a11y-native action format (Path W) — drop-in replacement for
eval_a11y_native.py with N-row batched generation.

Per-row inference does ~64 forward passes (autoregressive). At batch=1 the GPU
is idle most of the time. Batched generation processes B rows in parallel: same
forward-pass count, same model, but the kernel-launch and KV-cache overhead is
amortized across the batch. Empirically ~5-8x speedup on RTX 4090 with batch=8.

The scoring logic (action_match, _normalize_pred, resolve_element,
_coerce_action_json) is imported VERBATIM from eval_a11y_native.py so output
metrics are bit-identical given the same (model, predictions). Only the
generation loop differs.

Usage (matches eval_a11y_native.py exactly + --batch-size):
    uv run python scripts/eval_a11y_native_batched.py \\
        --data-dir data/androidcontrol_a11y_native \\
        --adapter outputs/.../checkpoint-7800 \\
        --num-samples 9999 --seed 3407 \\
        --batch-size 8 \\
        --output outputs/eval/runI_ckpt7800_batched.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Re-use scoring helpers + GEMMA4_MODEL constant from the proven eval.
sys.path.insert(0, str(Path(__file__).parent))
from eval_a11y_native import (  # noqa: E402
    GEMMA4_MODEL,
    TAP_RADIUS,
    _coerce_action_json,
    action_match,
    resolve_element,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/androidcontrol_a11y_native"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model", default=GEMMA4_MODEL)
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max-new-tokens", type=int, default=64,
                    help="Match sequential default. Long `type` actions can use most of 64.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save-all-predictions", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    print(f"[native-eval-batched] data={args.data_dir}/{args.split}.jsonl model={args.model}"
          + (f" adapter={args.adapter}" if args.adapter else "")
          + f" batch={args.batch_size}")

    # Load + sort + shuffle (matches eval_a11y_native.py exactly).
    rows: list[dict] = []
    with open(args.data_dir / f"{args.split}.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    rows.sort(key=lambda r: (r.get("episode_id"), r.get("step_index")))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.num_samples and args.num_samples < len(rows):
        rows = rows[: args.num_samples]
    print(f"[native-eval-batched] evaluating {len(rows)} rows")

    from PIL import Image
    from unsloth import FastVisionModel
    import torch

    if args.adapter:
        model, processor = FastVisionModel.from_pretrained(
            str(args.adapter), load_in_4bit=True, use_gradient_checkpointing=False
        )
    else:
        model, processor = FastVisionModel.from_pretrained(
            args.model, load_in_4bit=True, use_gradient_checkpointing=False
        )
    FastVisionModel.for_inference(model)

    # Make sure padding side is left so generation continues from the right.
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    correct = 0
    parse_fail = 0
    per_type: dict[str, dict] = {}
    confusion: dict[str, dict] = {}
    all_preds = []
    n_tap_rows = 0
    n_oracle_reachable = 0
    t0 = time.time()

    # Iterate in batches.
    for batch_start in range(0, len(rows), args.batch_size):
        batch_rows = rows[batch_start: batch_start + args.batch_size]
        batch_imgs = []
        batch_text_prompts = []
        for row in batch_rows:
            instr = row["messages"][0]["content"][1]["text"]
            img = Image.open(args.data_dir / row["image"]).convert("RGB")
            batch_imgs.append(img)
            msgs = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": instr},
            ]}]
            text_prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
            batch_text_prompts.append(text_prompt)

        # Use nested-list form: one image per text message. The Gemma 4
        # processor calls make_nested_list_of_images on the input — passing
        # a flat `[img1, ..., imgN]` is interpreted as a SINGLE sample with
        # N images, which then mismatches len(text)==N and raises. Wrapping
        # each image in its own list makes the per-row binding explicit.
        inputs = processor(
            text=batch_text_prompts,
            images=[[img] for img in batch_imgs],
            return_tensors="pt", padding=True,
        ).to(model.device)
        if batch_start == 0:
            if "pixel_values" not in inputs:
                sys.exit("[native-eval-batched] FATAL: pixel_values missing — image not bound.")
            pv = inputs["pixel_values"]
            B = len(batch_text_prompts)
            # Defensive shape check: expect pv.shape[0] == B (one image per row).
            print(f"[native-eval-batched] probe: input_ids.shape={tuple(inputs['input_ids'].shape)}, "
                  f"pixel_values.shape={tuple(pv.shape)}, expected leading dim = {B}")
            if pv.shape[0] != B:
                sys.exit(f"[native-eval-batched] FATAL: cannot bind 1 image per text "
                         f"(pixel_values batch dim {pv.shape[0]} != {B})")

        # RoPE-padding fix (§35.6): with left-padding, content tokens land at
        # different absolute sequence indices in batch=1 vs batch=8. RoPE rotates
        # Q/K vectors by absolute position, so attention scores diverge and
        # greedy decisions flip on borderline tokens. The smoke in §35.6 saw 7/200
        # mismatches due to this. Fix: pass explicit `position_ids` derived from
        # `attention_mask` so content tokens get the same absolute positions
        # whether they were left-padded or not. Pad positions get a placeholder
        # value (1) — they're masked out of attention so the value doesn't
        # affect outputs.
        attn_mask = inputs["attention_mask"]
        position_ids = attn_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attn_mask == 0, 1)
        inputs["position_ids"] = position_ids

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        # Slice out new tokens per row. With left-padding, the prompt length is
        # the same across the batch (it's padded to max), so the slice is the
        # same offset for every row. CAVEAT: if the processor expands image
        # tokens AFTER tokenizer padding, `out` and `inputs.input_ids` can
        # diverge in sequence length. Sanity-check on first batch by decoding
        # the boundary.
        prompt_len = inputs["input_ids"].shape[-1]
        if batch_start == 0:
            # If slicing is wrong, the "first generated tokens" decode as
            # tail-of-prompt (e.g. "model\n" or whitespace) instead of "{".
            tok = processor.tokenizer
            head_first = tok.decode(out[0, prompt_len:prompt_len + 6], skip_special_tokens=True)
            head_last  = tok.decode(out[-1, prompt_len:prompt_len + 6], skip_special_tokens=True)
            print(f"[native-eval-batched] gen-head[0]={head_first!r}  gen-head[-1]={head_last!r}")
            # Expect both to start with `{` or whitespace then `{` (JSON action).
            # If either looks like end-of-prompt tokens, slice is wrong.
        gen_only = out[:, prompt_len:]
        gen_texts = processor.tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        for i_row, (row, gen) in enumerate(zip(batch_rows, gen_texts)):
            gt = json.loads(row["messages"][1]["content"][0]["text"])
            gt_action_type = (gt.get("action") or "").lower()
            gt_xy = row.get("gt_xy")

            pred = _coerce_action_json(gen) or {}
            if not pred:
                parse_fail += 1

            if (pred.get("action") or "").lower() in ("tap", "long_press"):
                xy = resolve_element(pred, row.get("elements") or [])
                if xy is not None:
                    pred["_resolved_x"], pred["_resolved_y"] = xy

            # Oracle reachability ceiling
            if gt_action_type == "tap" and gt_xy is not None:
                n_tap_rows += 1
                oracle_xy = resolve_element({"element_id": gt.get("element_id")},
                                            row.get("elements") or [])
                if oracle_xy is not None:
                    dx, dy = oracle_xy[0] - gt_xy[0], oracle_xy[1] - gt_xy[1]
                    if (dx * dx + dy * dy) <= (TAP_RADIUS * TAP_RADIUS):
                        n_oracle_reachable += 1

            gt_for_match = dict(gt)
            gt_for_match["action"] = gt_action_type
            ok = action_match(pred, gt_for_match, gt_xy=gt_xy)
            correct += int(ok)
            per_type.setdefault(gt_action_type, {"n": 0, "ok": 0})
            per_type[gt_action_type]["n"] += 1
            per_type[gt_action_type]["ok"] += int(ok)
            p_t = (pred.get("action") or "").lower() or "<parse_fail>"
            confusion.setdefault(gt_action_type, {}).setdefault(p_t, 0)
            confusion[gt_action_type][p_t] += 1

            if args.save_all_predictions:
                all_preds.append({
                    "episode_id": row["episode_id"],
                    "step_index": row["step_index"],
                    "gt": gt, "gt_xy": gt_xy,
                    "pred_raw": gen.strip(), "pred": pred,
                    "n_elements": len(row.get("elements") or []),
                    "ok": ok,
                })

        # Periodic progress
        n_seen = batch_start + len(batch_rows)
        if n_seen % 50 == 0 or n_seen == len(rows):
            elapsed = time.time() - t0
            print(f"  [{n_seen}/{len(rows)}] full_match={correct/n_seen:.3f} "
                  f"parse_fail={parse_fail} {elapsed/60:.1f} min")

    metrics = {
        "label": "native-baseline-batched" if not args.adapter else "native-lora-batched",
        "model": args.model,
        "adapter": args.adapter,
        "num_samples": len(rows),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "tap_radius": TAP_RADIUS,
        "parse_rate": (len(rows) - parse_fail) / max(len(rows), 1),
        "full_match": correct / max(len(rows), 1),
        "tap_oracle_reachability": n_oracle_reachable / max(n_tap_rows, 1),
        "n_tap_rows": n_tap_rows,
        "per_type": {k: {"n": v["n"], "correct": v["ok"], "accuracy": v["ok"] / max(v["n"], 1)}
                     for k, v in per_type.items()},
        "confusion": confusion,
        "wall_seconds": time.time() - t0,
    }
    out_payload = {"metrics": metrics}
    if args.save_all_predictions:
        out_payload["all_predictions"] = all_preds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_payload, indent=2))
    print(f"[native-eval-batched] full_match={metrics['full_match']:.3f} "
          f"parse={metrics['parse_rate']:.3f} "
          f"tap_oracle={metrics['tap_oracle_reachability']:.3f} "
          f"({n_oracle_reachable}/{n_tap_rows}) "
          f"wall={metrics['wall_seconds']/60:.1f}min → {args.output}")


if __name__ == "__main__":
    main()
