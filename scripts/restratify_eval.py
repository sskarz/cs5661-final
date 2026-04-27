"""Post-hoc stratify an existing `*_full.json` eval by AndroidControl granularity.

Loads predictions from a `*_full.json` (must have been produced with
`--save-all-predictions`) and rejoins them with `data/androidcontrol/test.jsonl`
on `(episode_id, step_index)` to recover the `granularity` field, then prints
HL (`goal`) vs LL (`step_instruction`) splits of parse_rate / action_type_acc /
full_match.

Usage:
    uv run python scripts/restratify_eval.py outputs/eval/baseline_full.json
    uv run python scripts/restratify_eval.py outputs/eval/lora_runB_full.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Match scripts/eval_androidcontrol.py
TAP_RADIUS = 0.14


def actions_match(pred: dict, gt: dict, tap_radius: float = TAP_RADIUS) -> bool:
    if pred is None or gt is None:
        return False
    if pred.get("action") != gt.get("action"):
        return False
    a = gt["action"]
    if a in ("done", "navigate_back", "navigate_home", "wait"):
        return True
    if a == "tap" or a == "long_press":
        for k in ("x", "y"):
            if k not in pred or k not in gt:
                return False
        try:
            d = math.hypot(float(pred["x"]) - float(gt["x"]), float(pred["y"]) - float(gt["y"]))
        except (TypeError, ValueError):
            return False
        return d <= tap_radius
    if a == "type":
        return str(pred.get("text", "")).strip() == str(gt.get("text", "")).strip()
    if a == "open_app":
        return str(pred.get("app_name", "")).strip().lower() == str(gt.get("app_name", "")).strip().lower()
    if a == "scroll":
        return str(pred.get("direction", "")).strip().lower() == str(gt.get("direction", "")).strip().lower()
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--test-jsonl", type=Path, default=Path("data/androidcontrol/test.jsonl"))
    args = ap.parse_args()

    with open(args.path) as f:
        data = json.load(f)
    preds = data.get("all_predictions") or []
    if not preds:
        sys.exit(f"!! no all_predictions in {args.path}")

    # Build (episode_id, step_index) -> granularity index from test.jsonl
    gran_idx: dict[tuple[str, int], str] = {}
    with open(args.test_jsonl) as f:
        for line in f:
            r = json.loads(line)
            key = (str(r.get("episode_id")), int(r.get("step_index", -1)))
            gran_idx[key] = r.get("granularity", "<missing>")

    gran_total: Counter = Counter()
    gran_parse: Counter = Counter()
    gran_type: Counter = Counter()
    gran_full: Counter = Counter()
    gran_per_action_total: dict = defaultdict(Counter)
    gran_per_action_correct: dict = defaultdict(Counter)
    missing = 0

    for p in preds:
        key = (str(p.get("episode_id")), int(p.get("step_index", -1)))
        gran = gran_idx.get(key)
        if gran is None:
            missing += 1
            continue
        gran_total[gran] += 1
        gt = p["gt"]
        gt_action = gt.get("action")
        gran_per_action_total[gran][gt_action] += 1
        pred = p["pred_parsed"]
        if pred is not None:
            gran_parse[gran] += 1
            if pred.get("action") == gt_action:
                gran_type[gran] += 1
            if actions_match(pred, gt):
                gran_full[gran] += 1
                gran_per_action_correct[gran][gt_action] += 1

    print(f"=== Stratified results: {args.path.name} (n={len(preds)}, missing={missing}) ===")
    for g in sorted(gran_total):
        n = gran_total[g]
        print(
            f"  {g:18s}  n={n:4d}  "
            f"parse={gran_parse[g]/n:.3f}  "
            f"type={gran_type[g]/n:.3f}  "
            f"full={gran_full[g]/n:.3f}"
        )

    print()
    print("Per-action-type x granularity (correct / n):")
    all_actions = sorted(set().union(*(set(c) for c in gran_per_action_total.values())))
    grans = sorted(gran_total)
    header = f"  {'action':16s}  " + "  ".join(f"{g:>20s}" for g in grans)
    print(header)
    for a in all_actions:
        row = f"  {a:16s}  "
        for g in grans:
            tot = gran_per_action_total[g][a]
            corr = gran_per_action_correct[g][a]
            if tot == 0:
                cell = "       -      "
            else:
                cell = f"{corr:3d}/{tot:3d} ({corr/tot:.3f})"
            row += f"  {cell:>20s}"
        print(row)


if __name__ == "__main__":
    main()
