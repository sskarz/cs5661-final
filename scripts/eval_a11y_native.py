#!/usr/bin/env python3
"""Eval harness for a11y-native action format (Path W).

Reads data/androidcontrol_a11y_native/test.jsonl produced by
prepare_a11y_native.py. Each row carries:
  - the legend already baked into the user prompt
  - `elements` list (id -> bbox)
  - `gt_xy` original tap target (for radius scoring)

The model emits {"action":"tap","element_id":K}; we translate to bbox center
and score with the existing tap-radius distance threshold (0.14 normalized)
so numbers compare apples-to-apples with prior eval_androidcontrol.py runs.

Usage (baseline, no training):
    uv run python scripts/eval_a11y_native.py \\
        --data-dir data/androidcontrol_a11y_native \\
        --num-samples 200 --seed 3407 \\
        --output outputs/eval/native_baseline.json

Usage (with adapter):
    uv run python scripts/eval_a11y_native.py \\
        --data-dir data/androidcontrol_a11y_native \\
        --adapter outputs/runI/checkpoint-N \\
        --num-samples 200 --seed 3407 \\
        --output outputs/eval/runI_native.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

GEMMA4_MODEL = "unsloth/gemma-4-E2B-it"
TAP_RADIUS = 0.14


def _coerce_action_json(text: str) -> dict | None:
    """Brace-balanced first-object scan. Handles nested JSON; matches eval_androidcontrol."""
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
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


def _flatten_v2(action: dict) -> dict:
    """v2 -> v1 schema. Idempotent on v1 input.

    v2: {"action_type": "tap", "action_args": {"element_id": 7}}
    v1: {"action": "tap", "element_id": 7}

    Run K (CoCo-Agent CAP) writes v2; Run I/J use v1. The rest of this
    eval (action_match, resolve_element, scoring) speaks v1, so we flatten
    at every ingest point. Auto-detected — no --target-format flag needed.
    """
    if not isinstance(action, dict) or not action:
        return action
    if "action_type" in action and "action_args" in action:
        out = {"action": action["action_type"]}
        args = action.get("action_args") or {}
        if isinstance(args, dict):
            out.update(args)
        return out
    return action


def resolve_element(pred: dict, elements: list[dict]) -> tuple[float, float] | None:
    eid = pred.get("element_id")
    if eid is None:
        return None
    try:
        eid = int(eid)
    except Exception:
        return None
    for e in elements:
        if e["id"] == eid:
            b = e["bbox"]
            return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return None


_SCROLL_DIR_RE = re.compile(r"^scroll[\s_\-]*(up|down|left|right)$", re.IGNORECASE)


def _normalize_pred(pred: dict) -> dict:
    """Canonicalize model output before scoring.

    Run I confusion matrix showed model occasionally emits action strings
    like "scroll down" / "scroll_up" / "scroll-left" instead of the
    canonical {"action":"scroll", "direction":"down"} schema. Split those
    into the canonical form at score time. Defensive — does NOT modify the
    model output (we keep the raw output for diagnostics).
    """
    out = dict(pred)
    raw = (out.get("action") or out.get("action_type") or "").strip()
    m = _SCROLL_DIR_RE.match(raw)
    if m:
        out["action"] = "scroll"
        # Don't overwrite an explicit direction the model already set.
        if not out.get("direction"):
            out["direction"] = m.group(1).lower()
    return out


def action_match(pred: dict, gt: dict, gt_xy: list[float] | None) -> bool:
    pred = _normalize_pred(pred)
    p_t = (pred.get("action") or pred.get("action_type") or "").lower()
    g_t = (gt.get("action") or "").lower()
    if not p_t or p_t != g_t:
        return False
    if g_t in ("tap", "long_press"):
        if gt_xy is None:
            return False
        rx = pred.get("_resolved_x"); ry = pred.get("_resolved_y")
        if rx is None or ry is None:
            return False
        dx, dy = rx - gt_xy[0], ry - gt_xy[1]
        return (dx * dx + dy * dy) <= (TAP_RADIUS * TAP_RADIUS)
    if g_t == "scroll":
        return (pred.get("direction") or "").lower() == (gt.get("direction") or "").lower()
    if g_t == "type":
        return (pred.get("text") or "").strip() == (gt.get("text") or "").strip()
    if g_t == "open_app":
        return (pred.get("app_name") or "").strip().lower() == (gt.get("app_name") or "").strip().lower()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/androidcontrol_a11y_native"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model", default=GEMMA4_MODEL)
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--save-all-predictions", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    print(f"[native-eval] data={args.data_dir}/{args.split}.jsonl model={args.model}"
          + (f" adapter={args.adapter}" if args.adapter else ""))

    rows: list[dict] = []
    with open(args.data_dir / f"{args.split}.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))

    rows.sort(key=lambda r: (r.get("episode_id"), r.get("step_index")))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.num_samples and args.num_samples < len(rows):
        rows = rows[: args.num_samples]
    print(f"[native-eval] evaluating {len(rows)} rows")

    from PIL import Image
    from unsloth import FastVisionModel
    import torch

    # Adapter path: pass directly to FastVisionModel.from_pretrained, which
    # detects adapter_config.json and loads base + adapter together. Stock
    # peft.PeftModel.from_pretrained rejects Gemma4ClippableLinear (Unsloth's
    # wrapper around the LM Linears) — see eval_androidcontrol.py for the
    # same pattern.
    if args.adapter:
        model, processor = FastVisionModel.from_pretrained(
            str(args.adapter), load_in_4bit=True, use_gradient_checkpointing=False
        )
    else:
        model, processor = FastVisionModel.from_pretrained(
            args.model, load_in_4bit=True, use_gradient_checkpointing=False
        )
    FastVisionModel.for_inference(model)

    correct = 0; parse_fail = 0
    per_type: dict[str, dict] = {}
    confusion: dict[str, dict] = {}
    all_preds = []
    # Reachability ceiling: oracle-picks gt's element_id, projects to bbox
    # center, scores under tap_radius. Tells us how much accuracy is leaving
    # on the table due to the projection (vs. model error).
    n_tap_rows = 0
    n_oracle_reachable = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        gt = _flatten_v2(json.loads(row["messages"][1]["content"][0]["text"]))
        # gt may already be in element_id form (from prepare_a11y_native);
        # we still need the original action_type and (for scoring) gt_xy.
        gt_action_type = (gt.get("action") or "").lower()
        gt_xy = row.get("gt_xy")
        instr = row["messages"][0]["content"][1]["text"]  # legend already baked in

        img = Image.open(args.data_dir / row["image"]).convert("RGB")
        # Two-step pattern (text apply_chat_template + processor binds image).
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": instr},
        ]}]
        text_prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = processor(text=text_prompt, images=[img], return_tensors="pt").to(model.device)
        if i == 0 and "pixel_values" not in inputs:
            sys.exit("[native-eval] FATAL: pixel_values missing — image not bound.")
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        gen = processor.decode(
            out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        pred = _flatten_v2(_coerce_action_json(gen) or {})
        if not pred:
            parse_fail += 1

        if (pred.get("action") or "").lower() in ("tap", "long_press"):
            xy = resolve_element(pred, row.get("elements") or [])
            if xy is not None:
                pred["_resolved_x"], pred["_resolved_y"] = xy

        # Reachability ceiling: if model had picked the GT element_id, would
        # the projected bbox center fall within tap_radius of gt_xy?
        if gt_action_type == "tap" and gt_xy is not None:
            n_tap_rows += 1
            oracle_xy = resolve_element({"element_id": gt.get("element_id")},
                                        row.get("elements") or [])
            if oracle_xy is not None:
                dx, dy = oracle_xy[0] - gt_xy[0], oracle_xy[1] - gt_xy[1]
                if (dx * dx + dy * dy) <= (TAP_RADIUS * TAP_RADIUS):
                    n_oracle_reachable += 1

        # For scoring we need GT in original-action-type form. The stored
        # `gt` here may have `element_id` instead of `(x,y)`; rebuild it.
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

        if (i + 1) % 25 == 0 or i == len(rows) - 1:
            elapsed = time.time() - t0
            print(f"  [{i + 1}/{len(rows)}] full_match={correct/(i+1):.3f} parse_fail={parse_fail} {elapsed/60:.1f} min")

    metrics = {
        "label": "native-baseline" if not args.adapter else "native-lora",
        "model": args.model,
        "adapter": args.adapter,
        "num_samples": len(rows),
        "seed": args.seed,
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
    out = {"metrics": metrics}
    if args.save_all_predictions:
        out["all_predictions"] = all_preds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[native-eval] full_match={metrics['full_match']:.3f} "
          f"parse={metrics['parse_rate']:.3f} "
          f"tap_oracle={metrics['tap_oracle_reachability']:.3f} "
          f"({n_oracle_reachable}/{n_tap_rows}) → {args.output}")


if __name__ == "__main__":
    main()
