"""Spatial diagnostic for AndroidControl tap predictions.

Loads a `*_full.json` produced by `eval_androidcontrol.py --save-all-predictions`
and reports how the model's predicted tap coordinates are distributed.

Usage:
    uv run python scripts/analyze_tap_coords.py outputs/eval/lora_runB_full.json
    uv run python scripts/analyze_tap_coords.py outputs/eval/baseline_full.json
    uv run python scripts/analyze_tap_coords.py --compare lora_runB_full.json baseline_full.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def load_predictions(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data.get("all_predictions") or []


def is_tap(action: dict | None) -> bool:
    if not action or action.get("action") != "tap":
        return False
    if "x" not in action or "y" not in action:
        return False
    try:
        action["x"] = float(action["x"])
        action["y"] = float(action["y"])
        return True
    except (TypeError, ValueError):
        return False


def coord_buckets(values: list[float], n_bins: int = 10) -> list[int]:
    bins = [0] * n_bins
    for v in values:
        idx = min(int(v * n_bins), n_bins - 1)
        if idx < 0:
            idx = 0
        bins[idx] += 1
    return bins


def fmt_hist(bins: list[int], width: int = 40) -> str:
    total = sum(bins)
    if total == 0:
        return "(empty)"
    peak = max(bins)
    n = len(bins)
    out = []
    for i, c in enumerate(bins):
        lo = i / n
        hi = (i + 1) / n
        bar_len = int(c / peak * width) if peak else 0
        out.append(f"  [{lo:.2f}-{hi:.2f}]  {c:4d}  {'#' * bar_len}")
    return "\n".join(out)


def analyze(path: Path, label: str | None = None) -> dict:
    label = label or path.stem
    preds = load_predictions(path)
    if not preds:
        print(f"!! No predictions in {path}")
        return {}

    pred_taps = [p for p in preds if is_tap(p["pred_parsed"])]
    gt_taps = [p for p in preds if is_tap(p["gt"])]
    pred_tap_on_gt_tap = [p for p in gt_taps if is_tap(p["pred_parsed"])]

    pred_xs = [p["pred_parsed"]["x"] for p in pred_taps]
    pred_ys = [p["pred_parsed"]["y"] for p in pred_taps]

    print(f"\n========== {label} ==========")
    print(f"Total predictions:     {len(preds)}")
    print(f"GT == tap:             {len(gt_taps)}")
    print(f"Pred == tap (any GT):  {len(pred_taps)}")
    print(f"Tap predicted on tap:  {len(pred_tap_on_gt_tap)}")

    # Coordinate range / sanity
    if pred_xs:
        print(f"Pred x range: [{min(pred_xs):.4f}, {max(pred_xs):.4f}]  mean={sum(pred_xs)/len(pred_xs):.4f}")
        print(f"Pred y range: [{min(pred_ys):.4f}, {max(pred_ys):.4f}]  mean={sum(pred_ys)/len(pred_ys):.4f}")

    # Histograms
    print("\nx histogram (over all tap predictions):")
    print(fmt_hist(coord_buckets(pred_xs)))
    print("\ny histogram (over all tap predictions):")
    print(fmt_hist(coord_buckets(pred_ys)))

    # Top-N most common (x,y) values (rounded to 3 decimals to detect mode-collapse)
    rounded = Counter((round(p["pred_parsed"]["x"], 3), round(p["pred_parsed"]["y"], 3)) for p in pred_taps)
    print(f"\nTop-15 most-common predicted (x,y) [rounded to 3 dp]:")
    for (x, y), c in rounded.most_common(15):
        pct = c / len(pred_taps) * 100
        print(f"  ({x:.3f}, {y:.3f})   n={c:4d}   {pct:5.1f}%")

    # Distinct coordinate count
    print(f"\nDistinct (x,y) coords (3dp): {len(rounded)} / {len(pred_taps)} predictions")

    # Distance from GT, conditioned on pred=tap and gt=tap
    if pred_tap_on_gt_tap:
        dists = []
        within_radius = 0
        radius = 0.14
        for p in pred_tap_on_gt_tap:
            gt = p["gt"]
            pr = p["pred_parsed"]
            d = math.hypot(pr["x"] - gt["x"], pr["y"] - gt["y"])
            dists.append(d)
            if d <= radius:
                within_radius += 1
        dists.sort()
        n = len(dists)
        med = dists[n // 2]
        p75 = dists[int(n * 0.75)]
        p95 = dists[int(n * 0.95)]
        print(f"\nL2 distance from GT on tap-on-tap (n={n}, radius={radius}):")
        print(f"  within radius: {within_radius}/{n} = {within_radius/n*100:.1f}%")
        print(f"  median distance: {med:.4f}")
        print(f"  p75:            {p75:.4f}")
        print(f"  p95:            {p95:.4f}")
        print(f"  max:            {max(dists):.4f}")

        # How often is pred near a "canonical" position?
        canonicals = {
            "screen_center (0.5, 0.5)": (0.5, 0.5),
            "top_center (0.5, 0.1)": (0.5, 0.1),
            "bottom_center (0.5, 0.9)": (0.5, 0.9),
            "left_edge (0.0, 0.5)": (0.0, 0.5),
            "image_origin (0.0, 0.0)": (0.0, 0.0),
        }
        print(f"\nFraction of tap preds within 0.05 of canonical screen positions:")
        for name, (cx, cy) in canonicals.items():
            n_near = sum(1 for p in pred_taps if math.hypot(p["pred_parsed"]["x"] - cx, p["pred_parsed"]["y"] - cy) <= 0.05)
            print(f"  {name:30s}  {n_near:4d}/{len(pred_taps)} = {n_near/len(pred_taps)*100:.1f}%")

    return {
        "label": label,
        "n_preds": len(preds),
        "n_pred_tap": len(pred_taps),
        "n_gt_tap": len(gt_taps),
        "x_hist": coord_buckets(pred_xs),
        "y_hist": coord_buckets(pred_ys),
        "distinct_coords_3dp": len(rounded),
        "top_coords": rounded.most_common(15),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args()
    for p in args.paths:
        analyze(p)


if __name__ == "__main__":
    main()
