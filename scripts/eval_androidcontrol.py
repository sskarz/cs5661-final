#!/usr/bin/env python3
"""
Step 0 / Step 2 eval — Action-match accuracy on the AndroidControl test set.

Loads Gemma 4 E2B-it (optionally with a trained LoRA adapter), runs greedy
generation on N sampled test rows, parses the JSON action, and reports:

  - parse_rate              : % of completions that are valid JSON with an "action" field
  - action_type_accuracy    : % where predicted action type == ground truth
  - full_match              : full action match (type + args; tap uses radius tolerance)
  - per-action-type breakdown

Tap predictions are scored as correct if (norm_dx**2 + norm_dy**2) <= radius**2,
where radius defaults to 0.14 (the AndroidControl-paper convention).

Usage:
    # Baseline (no adapter)
    uv run python scripts/eval_androidcontrol.py \\
        --data-dir data/androidcontrol \\
        --num-samples 500 \\
        --output outputs/eval/baseline.json

    # With LoRA
    uv run python scripts/eval_androidcontrol.py \\
        --data-dir data/androidcontrol \\
        --adapter outputs/gemma4-e2b-androidcontrol-lora/final \\
        --num-samples 500 \\
        --output outputs/eval/lora.json
"""

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


GEMMA4_MODEL = "unsloth/gemma-4-E2B-it"
JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
LOC_X_RE = re.compile(r"<loc_x_(\d+)>")
LOC_Y_RE = re.compile(r"<loc_y_(\d+)>")

INSTRUCTION_PREFIX = (
    "You are an Android UI agent. Look at the screenshot and the user's goal, "
    "then output the next action as a single JSON object on one line and nothing else. "
    "Use exactly one of these schemas:\n"
    '  {"action": "tap", "x": <0..1>, "y": <0..1>}\n'
    '  {"action": "type", "text": "<string>"}\n'
    '  {"action": "open_app", "app_name": "<string>"}\n'
    '  {"action": "scroll", "direction": "up|down|left|right"}\n'
    '  {"action": "navigate_back"}\n'
    '  {"action": "navigate_home"}\n'
    '  {"action": "wait"}\n'
    '  {"action": "done"}\n'
    "Coordinates are normalized to [0, 1]. Output JSON only.\n\n"
    "Goal: "
)

INSTRUCTION_PREFIX_DISCRETE = (
    "You are an Android UI agent. Look at the screenshot and the user's goal, "
    "then output the next action as a single JSON object on one line and nothing else. "
    "Use exactly one of these schemas:\n"
    '  {"action": "tap", "x": "<loc_x_K>", "y": "<loc_y_K>"} where K is 0..{grid_max}\n'
    '  {"action": "type", "text": "<string>"}\n'
    '  {"action": "open_app", "app_name": "<string>"}\n'
    '  {"action": "scroll", "direction": "up|down|left|right"}\n'
    '  {"action": "navigate_back"}\n'
    '  {"action": "navigate_home"}\n'
    '  {"action": "wait"}\n'
    '  {"action": "done"}\n'
    "Output JSON only.\n\n"
    "Goal: "
)


def _parse_coord(val, grid_size: int, axis: str) -> float | None:
    """Parse a coord value that may be a float, a numeric string, or '<loc_x_K>'."""
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return None
    s = val.strip()
    pat = LOC_X_RE if axis == "x" else LOC_Y_RE
    m = pat.fullmatch(s) or pat.search(s)
    if m:
        k = int(m.group(1))
        if 0 <= k < grid_size:
            return (k + 0.5) / grid_size
        return None
    try:
        return float(s)
    except ValueError:
        return None


def extract_user_text(row: dict) -> str:
    for c in row["messages"][0]["content"]:
        if c["type"] == "text":
            return c["text"]
    raise ValueError("no user text")


def extract_gt_action(row: dict) -> dict:
    txt = row["messages"][1]["content"][0]["text"]
    return json.loads(txt)


def parse_prediction(text: str) -> dict | None:
    """Pull the first JSON object out of model output. Returns None on failure."""
    m = JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "action" not in obj:
        return None
    return obj


def actions_match(pred: dict, gt: dict, tap_radius: float, grid_size: int = 1024) -> bool:
    if pred.get("action") != gt.get("action"):
        return False
    a = gt["action"]
    if a == "tap":
        px = _parse_coord(pred.get("x"), grid_size, "x")
        py = _parse_coord(pred.get("y"), grid_size, "y")
        gx = _parse_coord(gt.get("x"), grid_size, "x")
        gy = _parse_coord(gt.get("y"), grid_size, "y")
        if None in (px, py, gx, gy):
            return False
        dx = px - gx
        dy = py - gy
        return (dx * dx + dy * dy) <= tap_radius * tap_radius
    if a == "type":
        return str(pred.get("text", "")).strip() == str(gt.get("text", "")).strip()
    if a == "open_app":
        return str(pred.get("app_name", "")).strip().lower() == str(gt.get("app_name", "")).strip().lower()
    if a == "scroll":
        return pred.get("direction") == gt.get("direction")
    # navigate_back / navigate_home / wait / done — type match is enough
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/androidcontrol"))
    parser.add_argument("--model", default=GEMMA4_MODEL)
    parser.add_argument("--adapter", type=Path, default=None,
                        help="Path to a saved LoRA adapter dir. Omit for baseline.")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--no-prefix", action="store_true",
                        help="Skip the schema instruction prefix (matches training prompt exactly).")
    parser.add_argument("--tap-radius", type=float, default=0.14)
    parser.add_argument("--save-all-predictions", action="store_true",
                        help="Save every (gt, pred) tuple, not just first 25 failures.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to dump the metrics JSON.")
    parser.add_argument("--coord-encoding", choices=["float", "discrete"], default="float",
                        help="discrete: parse <loc_x_K>/<loc_y_K> in pred and GT.")
    parser.add_argument("--grid-size", type=int, default=1024,
                        help="Discrete grid resolution (must match training).")
    args = parser.parse_args()

    test_jsonl = args.data_dir / "test.jsonl"
    if not test_jsonl.exists():
        raise SystemExit(f"Missing {test_jsonl}")

    # Lazy imports — Unsloth before transformers/peft.
    from unsloth import FastVisionModel
    import torch

    label = "lora" if args.adapter else "baseline"
    print(f"=== AndroidControl eval ({label}) ===")
    if args.adapter and args.coord_encoding == "discrete":
        # Discrete adapters can't be loaded via Unsloth's from_pretrained(adapter_dir)
        # because (a) PEFT load_state_dict crashes on size mismatch (base 262144 vs
        # saved 264192) and (b) plain peft.PeftModel.from_pretrained rejects Gemma4's
        # ClippableLinear modules. So: load base, resize, re-attach via Unsloth's
        # get_peft_model with the SAME LoRA config, then copy in saved weights.
        import json as _json
        from safetensors.torch import load_file as _load_safetensors

        cfg_path = args.adapter / "adapter_config.json"
        with open(cfg_path) as fh:
            cfg = _json.load(fh)
        base_name = cfg["base_model_name_or_path"]
        print(f"Loading base {base_name} (4-bit) for discrete adapter...")
        model, processor = FastVisionModel.from_pretrained(
            base_name,
            load_in_4bit=True,
            use_gradient_checkpointing=False,
        )

        # Load tokenizer (with new <loc_*> tokens) from adapter dir.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(args.adapter))
        processor.tokenizer = tok
        new_size = len(tok)

        # Resize main embeddings (input + tied lm_head).
        if model.get_input_embeddings().weight.size(0) != new_size:
            print(f"  Resizing embeddings: ... → {new_size}")
            model.resize_token_embeddings(new_size)

        # Resize Gemma 3/4 per-layer embedding (resize_token_embeddings misses it).
        for nm, mod in model.named_modules():
            if nm.endswith("embed_tokens_per_layer") and hasattr(mod, "weight"):
                if mod.weight.size(0) < new_size:
                    print(f"  Resizing {nm}: {mod.weight.size(0)} → {new_size}")
                    old_w = mod.weight.data
                    pad = old_w.mean(dim=0, keepdim=True).expand(
                        new_size - old_w.size(0), old_w.size(1)
                    ).contiguous().to(old_w.dtype)
                    grown = torch.cat([old_w, pad], dim=0)
                    mod.weight = torch.nn.Parameter(grown)
                    if hasattr(mod, "num_embeddings"):
                        mod.num_embeddings = new_size
                break

        # Re-attach LoRA via Unsloth's path (handles Gemma4ClippableLinear).
        # Use the same config the training run wrote into adapter_config.json.
        print(f"  Re-attaching LoRA (r={cfg['r']}, alpha={cfg['lora_alpha']})...")
        modules_to_save = cfg.get("modules_to_save") or None
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=cfg["r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg.get("lora_dropout", 0),
            bias=cfg.get("bias", "none"),
            random_state=3407,
            use_rslora=cfg.get("use_rslora", False),
            loftq_config=None,
            target_modules=cfg.get("target_modules", "all-linear"),
            modules_to_save=modules_to_save,
        )

        # Load saved weights into the freshly wrapped model.
        # PEFT save_pretrained writes *.lora_A.weight / *.embed_tokens.weight, but
        # the multi-adapter wrapped model expects *.lora_A.default.weight /
        # *.embed_tokens.modules_to_save.default.weight. Translate keys.
        sd_path = args.adapter / "adapter_model.safetensors"
        saved = _load_safetensors(str(sd_path))
        translated = {}
        for k, v in saved.items():
            new_k = k
            if k.endswith(".lora_A.weight"):
                new_k = k[:-len(".lora_A.weight")] + ".lora_A.default.weight"
            elif k.endswith(".lora_B.weight"):
                new_k = k[:-len(".lora_B.weight")] + ".lora_B.default.weight"
            elif k.endswith("embed_tokens.weight"):
                new_k = k[:-len(".weight")] + ".modules_to_save.default.weight"
            elif k.endswith("lm_head.weight"):
                new_k = k[:-len(".weight")] + ".modules_to_save.default.weight"
            translated[new_k] = v
        result = model.load_state_dict(translated, strict=False)
        adapter_keys = lambda ks: [k for k in ks if "lora" in k.lower() or "modules_to_save" in k]
        miss = adapter_keys(result.missing_keys)
        unexp = adapter_keys(result.unexpected_keys)
        print(f"  Loaded adapter: {len(translated)} tensors. "
              f"Adapter-relevant missing={len(miss)} unexpected={len(unexp)}")
        if unexp:
            print(f"    first unexpected: {unexp[:2]}")
        if miss:
            print(f"    first missing: {miss[:2]}")

        sentinel_id = tok.convert_tokens_to_ids("<loc_x_0>")
        if sentinel_id is None or sentinel_id == tok.unk_token_id:
            print("WARN: <loc_x_0> not a known token; adapter may be wrong.")
    elif args.adapter:
        print(f"Loading base + LoRA from {args.adapter} (4-bit)...")
        model, processor = FastVisionModel.from_pretrained(
            str(args.adapter),
            load_in_4bit=True,
            use_gradient_checkpointing=False,
        )
    else:
        print(f"Loading {args.model} (4-bit)...")
        model, processor = FastVisionModel.from_pretrained(
            args.model,
            load_in_4bit=True,
            use_gradient_checkpointing=False,
        )

    FastVisionModel.for_inference(model)

    print(f"Loading test rows from {test_jsonl}...")
    with open(test_jsonl) as f:
        all_rows = [json.loads(line) for line in f]
    print(f"  Total test rows: {len(all_rows)}")

    rng = random.Random(args.seed)
    rng.shuffle(all_rows)
    rows = all_rows[: args.num_samples]
    print(f"  Evaluating {len(rows)} rows (seed={args.seed})")

    parse_ok = 0
    type_ok = 0
    full_ok = 0
    per_type_total: Counter = Counter()
    per_type_correct: Counter = Counter()
    confusion: dict = defaultdict(Counter)  # gt_type -> pred_type counter
    failures: list[dict] = []
    all_predictions: list[dict] = []
    # Per-granularity (HL=goal, LL=step_instruction) tallies
    gran_total: Counter = Counter()
    gran_parse: Counter = Counter()
    gran_type: Counter = Counter()
    gran_full: Counter = Counter()

    t0 = time.time()
    for i, row in enumerate(rows):
        gt = extract_gt_action(row)
        gt_type = gt["action"]
        per_type_total[gt_type] += 1
        gran = row.get("granularity", "<missing>")
        gran_total[gran] += 1

        user_text = extract_user_text(row)
        if args.no_prefix:
            prompt_text = user_text
        elif args.coord_encoding == "discrete":
            prompt_text = (
                INSTRUCTION_PREFIX_DISCRETE.replace("{grid_max}", str(args.grid_size - 1))
                + user_text
            )
        else:
            prompt_text = INSTRUCTION_PREFIX + user_text
        img = Image.open(args.data_dir / row["image"]).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            text=prompt,
            images=[img],
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        # Discrete <loc_*> tokens are registered as additional_special_tokens —
        # skip_special_tokens=True would erase them. Keep them in discrete mode.
        completion = processor.decode(
            gen_ids, skip_special_tokens=(args.coord_encoding == "float")
        )

        pred = parse_prediction(completion)
        pred_type = pred["action"] if pred and "action" in pred else "<parse_fail>"
        confusion[gt_type][pred_type] += 1
        if args.save_all_predictions:
            all_predictions.append({
                "episode_id": row.get("episode_id"),
                "step_index": row.get("step_index"),
                "image": row.get("image"),
                "granularity": gran,
                "user_text": user_text,
                "gt": gt,
                "pred_raw": completion,
                "pred_parsed": pred,
            })

        if pred is not None:
            parse_ok += 1
            gran_parse[gran] += 1
            if pred.get("action") == gt_type:
                type_ok += 1
                gran_type[gran] += 1
            if actions_match(pred, gt, args.tap_radius, args.grid_size):
                full_ok += 1
                per_type_correct[gt_type] += 1
                gran_full[gran] += 1
            else:
                if len(failures) < 25:
                    failures.append({
                        "episode_id": row.get("episode_id"),
                        "step_index": row.get("step_index"),
                        "image": row.get("image"),
                        "user_text": user_text,
                        "gt": gt,
                        "pred_raw": completion,
                        "pred_parsed": pred,
                    })
        else:
            if len(failures) < 25:
                failures.append({
                    "episode_id": row.get("episode_id"),
                    "step_index": row.get("step_index"),
                    "image": row.get("image"),
                    "user_text": user_text,
                    "gt": gt,
                    "pred_raw": completion,
                    "pred_parsed": None,
                })

        if (i + 1) % 25 == 0 or i == len(rows) - 1:
            dt = time.time() - t0
            rate = (i + 1) / max(dt, 1e-6)
            print(
                f"  [{i+1:4d}/{len(rows)}] "
                f"parse={parse_ok/(i+1):.3f} "
                f"type={type_ok/(i+1):.3f} "
                f"full={full_ok/(i+1):.3f} "
                f"({rate:.2f} ex/s)"
            )

    n = len(rows)
    metrics = {
        "label": label,
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "num_samples": n,
        "seed": args.seed,
        "tap_radius": args.tap_radius,
        "parse_rate": parse_ok / n,
        "action_type_accuracy": type_ok / n,
        "full_match": full_ok / n,
        "per_type": {
            t: {
                "n": per_type_total[t],
                "correct": per_type_correct[t],
                "accuracy": per_type_correct[t] / per_type_total[t] if per_type_total[t] else None,
            }
            for t in sorted(per_type_total)
        },
        "confusion": {gt: dict(c) for gt, c in confusion.items()},
        "per_granularity": {
            g: {
                "n": gran_total[g],
                "parse_rate": gran_parse[g] / gran_total[g] if gran_total[g] else None,
                "action_type_accuracy": gran_type[g] / gran_total[g] if gran_total[g] else None,
                "full_match": gran_full[g] / gran_total[g] if gran_total[g] else None,
            }
            for g in sorted(gran_total)
        },
        "wall_seconds": time.time() - t0,
    }

    print()
    print(f"=== Results ({label}, n={n}) ===")
    print(f"parse_rate            = {metrics['parse_rate']:.4f}")
    print(f"action_type_accuracy  = {metrics['action_type_accuracy']:.4f}")
    print(f"full_match            = {metrics['full_match']:.4f}")
    print("per-granularity (HL=goal, LL=step_instruction):")
    for g, v in metrics["per_granularity"].items():
        print(
            f"  {g:18s}  n={v['n']:4d}  "
            f"parse={v['parse_rate']:.3f}  "
            f"type={v['action_type_accuracy']:.3f}  "
            f"full={v['full_match']:.3f}"
        )
    print("per-action-type:")
    for t, v in metrics["per_type"].items():
        acc = v["accuracy"]
        acc_s = f"{acc:.3f}" if acc is not None else " n/a "
        print(f"  {t:16s}  n={v['n']:4d}  correct={v['correct']:4d}  acc={acc_s}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {"metrics": metrics, "failures": failures}
        if args.save_all_predictions:
            payload["all_predictions"] = all_predictions
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote metrics → {args.output}")


if __name__ == "__main__":
    main()
