#!/usr/bin/env python3
"""Set-of-Mark eval harness for AndroidControl.

Wraps the existing eval pipeline with SoM rendering. Reads
data/androidcontrol_a11y/test.jsonl, overlays numbered marks on each
screenshot via render_som.render_marks, prompts the model with the marked
image + legend, parses {"action":"tap","mark": K}, looks up mark K's
centroid as the predicted (x,y), then scores via the existing
actions_match() with tap_radius=0.14.

Runs against the BASELINE Gemma 4 E2B (no LoRA) by default — the whole
point is that SoM is inference-time only.

Usage:
    uv run python scripts/eval_som.py \\
        --data-dir data/androidcontrol_a11y \\
        --num-samples 200 --seed 3407 \\
        --output outputs/eval/som_baseline.json

For a stratified sweep matching prior eval splits, --num-samples 200 with
the same seed reproduces the same row indices.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

# Reuse the existing eval primitives (parse, action match, granularity stratification).
sys.path.insert(0, str(Path(__file__).parent))
from render_som import render_marks, build_mark_prompt  # noqa: E402

GEMMA4_MODEL = "unsloth/gemma-4-E2B-it"
TAP_RADIUS = 0.14


def _coerce_action_json(text: str) -> dict | None:
    """Brace-balanced first-object scan. Handles nested JSON; matches eval_androidcontrol."""
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE)
    depth = 0; start = -1
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(s[start:i + 1])
                except Exception:
                    start = -1
    return None


def som_pred_to_xy(pred: dict, marks: list[dict]) -> tuple[float, float] | None:
    """Translate the model's mark id into (x,y). None if invalid."""
    mid = pred.get("mark")
    if mid is None:
        return None
    try:
        mid = int(mid)
    except Exception:
        return None
    for m in marks:
        if m["id"] == mid:
            return float(m["cx"]), float(m["cy"])
    return None


def action_match(pred: dict, gt: dict, tap_radius: float = TAP_RADIUS) -> bool:
    """Mirrors eval_androidcontrol.actions_match — minus discrete-coord paths.

    Tap: must have 'mark' that resolves to (cx,cy) within `tap_radius` of GT.
    Non-tap: action_type match plus the relevant subfield.
    """
    p_t = (pred.get("action") or pred.get("action_type") or "").lower()
    g_t = (gt.get("action") or gt.get("action_type") or "").lower()
    if not p_t or p_t != g_t:
        return False
    if g_t in ("tap", "long_press"):
        # In SoM, the predicted (x,y) comes from the resolved mark centroid.
        px = pred.get("_resolved_x"); py = pred.get("_resolved_y")
        if px is None or py is None:
            return False
        gx = gt.get("x"); gy = gt.get("y")
        if gx is None or gy is None:
            return False
        # Match eval_androidcontrol.py: <=, comparison in squared form.
        dx, dy = px - gx, py - gy
        return (dx * dx + dy * dy) <= (tap_radius * tap_radius)
    if g_t == "scroll":
        return (pred.get("direction") or "").lower() == (gt.get("direction") or "").lower()
    if g_t == "type":
        return (pred.get("text") or "").strip() == (gt.get("text") or "").strip()
    if g_t == "open_app":
        return (pred.get("app_name") or "").strip().lower() == (gt.get("app_name") or "").strip().lower()
    # navigate_back, navigate_home, wait, done — type match is enough
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/androidcontrol_a11y"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--adapter", default=None,
                    help="Optional LoRA adapter to load on top of base model.")
    ap.add_argument("--model", default=GEMMA4_MODEL)
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max-marks", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--save-all-predictions", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    print(f"[som-eval] data={args.data_dir}/{args.split}.jsonl model={args.model}"
          + (f" adapter={args.adapter}" if args.adapter else ""))

    rows: list[dict] = []
    with open(args.data_dir / f"{args.split}.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if not row.get("a11y"):
                continue
            rows.append(row)
    print(f"[som-eval] loaded {len(rows)} rows with a11y")

    # Deterministic ordering before shuffle so the same seed picks the same
    # rows even if upstream parser reorders partials.
    rows.sort(key=lambda r: (r.get("episode_id"), r.get("step_index")))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.num_samples and args.num_samples < len(rows):
        rows = rows[: args.num_samples]
    print(f"[som-eval] evaluating {len(rows)} rows")

    # Lazy import — keep --help fast.
    from PIL import Image
    from unsloth import FastVisionModel
    import torch

    # Pass adapter dir directly to FastVisionModel — stock peft.PeftModel
    # rejects Unsloth's Gemma4ClippableLinear wrapper. Mirrors eval_androidcontrol.
    if args.adapter:
        model, processor = FastVisionModel.from_pretrained(
            str(args.adapter), load_in_4bit=True, use_gradient_checkpointing=False
        )
    else:
        model, processor = FastVisionModel.from_pretrained(
            args.model, load_in_4bit=True, use_gradient_checkpointing=False
        )
    FastVisionModel.for_inference(model)

    correct = 0
    parse_fail = 0
    per_type: dict[str, dict] = {}
    confusion: dict[str, dict] = {}
    all_preds: list[dict] = []
    # Diagnostic: how often is the GT tap target reachable from the rendered marks?
    # Caps the achievable score on tap rows.
    n_tap_rows = 0
    n_tap_gt_in_marks = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        gt = json.loads(row["messages"][1]["content"][0]["text"])
        instr = row["messages"][0]["content"][1]["text"]
        img = Image.open(args.data_dir / row["image"]).convert("RGB")
        marked, marks = render_marks(img, row["a11y"], max_marks=args.max_marks)
        prompt = build_mark_prompt(instr, marks)

        # Two-step pattern matching eval_androidcontrol.py: text-only
        # apply_chat_template, then bind the image via processor(text=, images=).
        # Inline {"image": pil} keys do NOT propagate through Unsloth's wrapped
        # Gemma 4 chat template — would silently produce a text-only run.
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text_prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = processor(text=text_prompt, images=[marked], return_tensors="pt").to(model.device)
        if i == 0 and "pixel_values" not in inputs:
            sys.exit("[som-eval] FATAL: pixel_values missing from inputs — image not bound. "
                     "Check processor invocation.")
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        gen = processor.decode(
            out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        pred = _coerce_action_json(gen) or {}
        if not pred:
            parse_fail += 1
        # Resolve mark -> (cx,cy) for tap/long_press
        if (pred.get("action") or "").lower() in ("tap", "long_press"):
            xy = som_pred_to_xy(pred, marks)
            if xy is not None:
                pred["_resolved_x"], pred["_resolved_y"] = xy

        # Reachability diagnostic: for each tap row, is the GT target in any mark's bbox?
        if (gt.get("action") or "").lower() in ("tap", "long_press"):
            n_tap_rows += 1
            gx, gy = gt.get("x"), gt.get("y")
            if gx is not None and gy is not None:
                for m in marks:
                    b = m["bbox"]
                    if b[0] <= gx <= b[2] and b[1] <= gy <= b[3]:
                        n_tap_gt_in_marks += 1
                        break

        ok = action_match(pred, gt)
        correct += int(ok)
        gt_t = (gt.get("action") or "").lower()
        per_type.setdefault(gt_t, {"n": 0, "ok": 0})
        per_type[gt_t]["n"] += 1
        per_type[gt_t]["ok"] += int(ok)

        # Confusion: GT type vs predicted type (or <parse_fail>)
        p_t = (pred.get("action") or "").lower() or "<parse_fail>"
        confusion.setdefault(gt_t, {}).setdefault(p_t, 0)
        confusion[gt_t][p_t] += 1

        if args.save_all_predictions:
            all_preds.append({
                "episode_id": row["episode_id"],
                "step_index": row["step_index"],
                "instruction": instr,
                "gt": gt,
                "pred_raw": gen.strip(),
                "pred": pred,
                "n_marks": len(marks),
                "ok": ok,
            })
        if (i + 1) % 25 == 0 or i == len(rows) - 1:
            elapsed = time.time() - t0
            print(f"  [{i + 1}/{len(rows)}] full_match={correct/(i+1):.3f} parse_fail={parse_fail} {elapsed/60:.1f} min")

    metrics = {
        "label": "som-baseline" if not args.adapter else "som-lora",
        "model": args.model,
        "adapter": args.adapter,
        "num_samples": len(rows),
        "seed": args.seed,
        "tap_radius": TAP_RADIUS,
        "parse_rate": (len(rows) - parse_fail) / max(len(rows), 1),
        "full_match": correct / max(len(rows), 1),
        "per_type": {k: {"n": v["n"], "correct": v["ok"], "accuracy": v["ok"] / max(v["n"], 1)}
                     for k, v in per_type.items()},
        "confusion": confusion,
        # Tap reachability ceiling: % of tap rows where GT is inside any mark's bbox.
        "tap_rows": n_tap_rows,
        "tap_gt_in_marks": n_tap_gt_in_marks,
        "tap_reachability": n_tap_gt_in_marks / max(n_tap_rows, 1),
        "wall_seconds": time.time() - t0,
    }
    out = {"metrics": metrics}
    if args.save_all_predictions:
        out["all_predictions"] = all_preds

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[som-eval] full_match={metrics['full_match']:.3f} parse={metrics['parse_rate']:.3f} → {args.output}")


if __name__ == "__main__":
    main()
