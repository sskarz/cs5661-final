#!/usr/bin/env python3
"""Smoke evaluator: reads `data/pathZ/smoke/eval.jsonl`, generates with
Gemma 4 E2B (+ optional adapter), parses the M3A `Reason: ... Action: {...}`
emission, scores against the gt_m3a action.

Outputs `METRIC <name>=<value>` lines for autoresearch:
  full_match     — % rows where action_type AND grounding arg match gt
  type_match     — % where action_type matches gt (looser)
  parse_pct      — % rows that produced a parseable Action: {...} JSON
  reason_pct     — % rows that emitted a Reason: ... line
  per_type_*     — type_match split by gt action_type
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Local helpers
sys.path.insert(0, str(Path(__file__).parent))
from m3a_format import parse_m3a_emission, m3a_action_match


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", type=Path,
                    default=Path("data/pathZ/smoke/eval.jsonl"))
    ap.add_argument("--data-dir", type=Path,
                    default=Path("data/androidcontrol_a11y_native_v3"))
    ap.add_argument("--model", default="unsloth/gemma-4-E2B-it")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-preds", type=Path, default=None)
    args = ap.parse_args()

    rows: list[dict] = []
    with open(args.eval_jsonl) as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]
    print(f"[smoke-eval] {len(rows)} rows; adapter={args.adapter}")

    from PIL import Image
    import torch
    from unsloth import FastVisionModel

    if args.adapter:
        model, processor = FastVisionModel.from_pretrained(
            str(args.adapter), load_in_4bit=True,
            use_gradient_checkpointing=False,
        )
    else:
        model, processor = FastVisionModel.from_pretrained(
            args.model, load_in_4bit=True,
            use_gradient_checkpointing=False,
        )
    FastVisionModel.for_inference(model)

    n_full = 0
    n_type = 0
    n_parse = 0
    n_reason = 0
    per_type_total: Counter[str] = Counter()
    per_type_full: Counter[str] = Counter()
    per_type_type: Counter[str] = Counter()
    pred_action_types: Counter[str] = Counter()
    saved = []

    t0 = time.time()
    for i, row in enumerate(rows):
        gt_m3a = row["gt_m3a"]
        user_text = next(
            c["text"] for c in row["messages"][0]["content"] if c["type"] == "text"
        )
        img = Image.open(args.data_dir / row["image"]).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": user_text},
        ]}]
        text_prompt = processor.apply_chat_template(
            msgs, add_generation_prompt=True
        )
        inputs = processor(text=text_prompt, images=[img], return_tensors="pt"
                           ).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        gen = processor.decode(
            out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        reason, pred = parse_m3a_emission(gen)
        n_reason += int(reason is not None)
        n_parse += int(pred is not None)
        match = m3a_action_match(pred, gt_m3a)
        n_full += int(match["full_match"])
        n_type += int(match["type_match"])
        gt_at = gt_m3a.get("action_type", "?")
        per_type_total[gt_at] += 1
        per_type_full[gt_at] += int(match["full_match"])
        per_type_type[gt_at] += int(match["type_match"])
        if pred is not None:
            pred_action_types[pred.get("action_type", "?")] += 1
        else:
            pred_action_types["<parse_fail>"] += 1
        if args.save_preds is not None:
            saved.append({
                "image": row["image"],
                "gt": gt_m3a,
                "pred_raw": gen,
                "pred": pred,
                "reason": reason,
                "match": match,
            })
        if (i + 1) % 25 == 0 or i == len(rows) - 1:
            el = time.time() - t0
            print(f"  [{i+1}/{len(rows)}] full={n_full/(i+1):.3f} "
                  f"type={n_type/(i+1):.3f} parse={n_parse/(i+1):.3f} "
                  f"{el:.0f}s", flush=True)

    n = max(1, len(rows))
    print(f"\n[smoke-eval] action_type dist (pred):")
    for at, c in pred_action_types.most_common():
        print(f"  {at:18s} {c:4d}  ({100.0*c/n:.1f}%)")
    print(f"[smoke-eval] per-type type-match:")
    for at, total in per_type_total.most_common():
        ok = per_type_type[at]
        full = per_type_full[at]
        print(f"  {at:18s} type={ok:3d}/{total:3d} ({100.0*ok/total:.1f}%) "
              f"full={full:3d}/{total:3d} ({100.0*full/total:.1f}%)")

    if args.save_preds is not None:
        args.save_preds.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_preds, "w") as f:
            for r in saved:
                f.write(json.dumps(r) + "\n")
        print(f"[smoke-eval] preds saved to {args.save_preds}")

    full_pct = 100.0 * n_full / n
    type_pct = 100.0 * n_type / n
    parse_pct = 100.0 * n_parse / n
    reason_pct = 100.0 * n_reason / n
    print(f"\nMETRIC full_match={full_pct:.2f}")
    print(f"METRIC type_match={type_pct:.2f}")
    print(f"METRIC parse_pct={parse_pct:.2f}")
    print(f"METRIC reason_pct={reason_pct:.2f}")
    for at, total in per_type_total.most_common():
        if total >= 5:  # only stable subsets
            type_acc = 100.0 * per_type_type[at] / total
            print(f"METRIC type_match_{at}={type_acc:.2f}")


if __name__ == "__main__":
    main()
