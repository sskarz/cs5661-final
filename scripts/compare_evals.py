"""Head-to-head: same eval row, two adapters. Categorize where they agree/disagree.

Joins two `*_full.json` files (both produced with `--save-all-predictions`) on
`(episode_id, step_index, granularity)` — the unique row key in
`data/androidcontrol/test.jsonl` — and buckets each row into:

  - regression   : A correct, B wrong   (what B broke)
  - gain         : A wrong, B correct   (what B fixed)
  - both_correct : both right           (preserved capability)
  - both_wrong   : both wrong           (independent of B)

Then prints summary counts (with HL/LL split) and dumps regressions for
inspection.

Usage:
    uv run python scripts/compare_evals.py \\
        --a outputs/eval/baseline_v2.json --a-name baseline \\
        --b outputs/eval/lora_runC.json --b-name runC \\
        --out outputs/eval/baseline_vs_runC.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

TAP_RADIUS = 0.14


def actions_match(pred: dict | None, gt: dict, tap_radius: float = TAP_RADIUS) -> bool:
    if pred is None or gt is None:
        return False
    if pred.get("action") != gt.get("action"):
        return False
    a = gt["action"]
    if a in ("done", "navigate_back", "navigate_home", "wait"):
        return True
    if a in ("tap", "long_press"):
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


def index(preds: list[dict]) -> dict[tuple, dict]:
    idx: dict[tuple, dict] = {}
    for p in preds:
        key = (str(p.get("episode_id")), int(p.get("step_index", -1)), p.get("granularity"))
        idx[key] = p
    return idx


def categorize(a_correct: bool, b_correct: bool) -> str:
    if a_correct and b_correct:
        return "both_correct"
    if a_correct and not b_correct:
        return "regression"
    if not a_correct and b_correct:
        return "gain"
    return "both_wrong"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--show-regressions", type=int, default=20)
    ap.add_argument("--show-gains", type=int, default=20)
    args = ap.parse_args()

    a_data = json.load(open(args.a))
    b_data = json.load(open(args.b))
    a_idx = index(a_data.get("all_predictions") or [])
    b_idx = index(b_data.get("all_predictions") or [])

    common = sorted(set(a_idx) & set(b_idx))
    print(f"=== {args.a_name} vs {args.b_name} (n_common={len(common)}) ===")
    print(f"  {args.a_name} rows: {len(a_idx)}")
    print(f"  {args.b_name} rows: {len(b_idx)}")
    print()

    bucket_total: Counter = Counter()
    bucket_by_gran: dict = defaultdict(Counter)
    bucket_by_action: dict = defaultdict(Counter)
    pred_a_action_in_regressions: Counter = Counter()
    pred_b_action_in_regressions: Counter = Counter()
    pred_a_action_in_gains: Counter = Counter()
    pred_b_action_in_gains: Counter = Counter()

    rows_out: list[dict] = []

    for key in common:
        a = a_idx[key]
        b = b_idx[key]
        gt = a["gt"]  # same row, same gt
        a_correct = actions_match(a["pred_parsed"], gt)
        b_correct = actions_match(b["pred_parsed"], gt)
        cat = categorize(a_correct, b_correct)
        bucket_total[cat] += 1
        gran = a.get("granularity") or "<missing>"
        bucket_by_gran[gran][cat] += 1
        bucket_by_action[gt.get("action", "<?>")][cat] += 1

        if cat == "regression":
            a_act = (a["pred_parsed"] or {}).get("action") or "<parse_fail>"
            b_act = (b["pred_parsed"] or {}).get("action") or "<parse_fail>"
            pred_a_action_in_regressions[a_act] += 1
            pred_b_action_in_regressions[b_act] += 1
        elif cat == "gain":
            a_act = (a["pred_parsed"] or {}).get("action") or "<parse_fail>"
            b_act = (b["pred_parsed"] or {}).get("action") or "<parse_fail>"
            pred_a_action_in_gains[a_act] += 1
            pred_b_action_in_gains[b_act] += 1

        rows_out.append({
            "key": list(key),
            "user_text": a.get("user_text"),
            "image": a.get("image"),
            "granularity": gran,
            "gt": gt,
            f"{args.a_name}_pred": a["pred_parsed"],
            f"{args.b_name}_pred": b["pred_parsed"],
            f"{args.a_name}_correct": a_correct,
            f"{args.b_name}_correct": b_correct,
            "category": cat,
        })

    n = len(common)
    print("Categories (full_match basis):")
    for cat in ("both_correct", "regression", "gain", "both_wrong"):
        c = bucket_total[cat]
        print(f"  {cat:14s}  {c:4d}  ({c/n*100:5.1f}%)")
    a_full = bucket_total["both_correct"] + bucket_total["regression"]
    b_full = bucket_total["both_correct"] + bucket_total["gain"]
    print(f"\nfull_match  {args.a_name}: {a_full}/{n} ({a_full/n:.4f})")
    print(f"full_match  {args.b_name}: {b_full}/{n} ({b_full/n:.4f})")
    print(f"net change ({args.b_name} - {args.a_name}): {b_full - a_full}")

    print("\nBy granularity:")
    for g in sorted(bucket_by_gran):
        print(f"  {g}")
        gn = sum(bucket_by_gran[g].values())
        for cat in ("both_correct", "regression", "gain", "both_wrong"):
            c = bucket_by_gran[g][cat]
            print(f"    {cat:14s}  {c:4d}  ({c/gn*100:5.1f}%)")

    print("\nBy ground-truth action type:")
    for a in sorted(bucket_by_action):
        an = sum(bucket_by_action[a].values())
        c_both = bucket_by_action[a]["both_correct"]
        c_reg = bucket_by_action[a]["regression"]
        c_gain = bucket_by_action[a]["gain"]
        c_bw = bucket_by_action[a]["both_wrong"]
        net = c_gain - c_reg
        print(f"  {a:16s}  n={an:4d}  both_correct={c_both:3d}  reg={c_reg:3d}  gain={c_gain:3d}  bw={c_bw:3d}  net={net:+d}")

    print(f"\nIn regressions: action {args.a_name} predicted (correct):")
    for act, c in pred_a_action_in_regressions.most_common():
        print(f"  {act:16s}  {c}")
    print(f"In regressions: action {args.b_name} predicted (wrong):")
    for act, c in pred_b_action_in_regressions.most_common():
        print(f"  {act:16s}  {c}")
    print(f"\nIn gains: action {args.a_name} predicted (wrong):")
    for act, c in pred_a_action_in_gains.most_common():
        print(f"  {act:16s}  {c}")
    print(f"In gains: action {args.b_name} predicted (correct):")
    for act, c in pred_b_action_in_gains.most_common():
        print(f"  {act:16s}  {c}")

    if args.show_regressions > 0:
        print(f"\n=== Sample regressions (first {args.show_regressions}, where {args.a_name} got it but {args.b_name} broke it): ===")
        regressions = [r for r in rows_out if r["category"] == "regression"]
        for r in regressions[: args.show_regressions]:
            print(f"\n  [{r['granularity']:18s}] {r['user_text'][:100]}")
            print(f"    GT:        {r['gt']}")
            print(f"    {args.a_name}: {r[f'{args.a_name}_pred']}")
            print(f"    {args.b_name}: {r[f'{args.b_name}_pred']}")

    if args.show_gains > 0:
        gains = [r for r in rows_out if r["category"] == "gain"]
        if gains:
            print(f"\n=== Sample gains (first {args.show_gains}, where {args.b_name} fixed something {args.a_name} got wrong): ===")
            for r in gains[: args.show_gains]:
                print(f"\n  [{r['granularity']:18s}] {r['user_text'][:100]}")
                print(f"    GT:        {r['gt']}")
                print(f"    {args.a_name}: {r[f'{args.a_name}_pred']}")
                print(f"    {args.b_name}: {r[f'{args.b_name}_pred']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({
            "a": args.a_name, "b": args.b_name,
            "n_common": n,
            "buckets": dict(bucket_total),
            "by_granularity": {g: dict(c) for g, c in bucket_by_gran.items()},
            "by_action": {a: dict(c) for a, c in bucket_by_action.items()},
            "rows": rows_out,
        }, open(args.out, "w"), indent=2)
        print(f"\nWrote → {args.out}")


if __name__ == "__main__":
    main()
