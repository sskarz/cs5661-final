#!/usr/bin/env python3
"""Re-score a saved eval_a11y_native.py JSON using the current action_match.

Loads `all_predictions` from a saved eval JSON, re-runs `action_match` (which
now applies scroll-direction normalization), and prints before/after metrics.
Lets us quantify the scroll-fix lift without re-running model inference.

Usage:
    uv run python scripts/rescore_native.py outputs/eval/runI_ckpt1500.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_a11y_native import action_match, TAP_RADIUS  # noqa: E402


def rescore(eval_json_path: Path) -> dict:
    raw = json.loads(eval_json_path.read_text())
    preds = raw.get("all_predictions") or []
    if not preds:
        raise SystemExit(f"No all_predictions in {eval_json_path}")

    ok_old = sum(1 for p in preds if p.get("ok"))
    per_type_new: dict[str, dict] = {}
    ok_new = 0
    flips = []  # (idx, gt_action, raw_pred_action, was_ok, now_ok)

    for i, p in enumerate(preds):
        gt = p.get("gt") or {}
        gt_xy = p.get("gt_xy")
        # The stored gt may have element_id form; rebuild with original action.
        gt_for_match = dict(gt)
        # gt_action stored as the canonical action string (already canonicalized)
        gt_for_match["action"] = (gt.get("action") or "").lower()

        pred = p.get("pred") or {}
        # _resolved_x/y were computed at eval time; preserve them for tap scoring.
        was_ok = bool(p.get("ok"))
        now_ok = bool(action_match(pred, gt_for_match, gt_xy=gt_xy))
        ok_new += int(now_ok)
        if was_ok != now_ok:
            flips.append((i, gt_for_match["action"], pred.get("action"), was_ok, now_ok))

        gt_t = gt_for_match["action"]
        per_type_new.setdefault(gt_t, {"n": 0, "ok": 0})
        per_type_new[gt_t]["n"] += 1
        per_type_new[gt_t]["ok"] += int(now_ok)

    n = len(preds)
    print(f"=== Re-score {eval_json_path.name} ===")
    print(f"  rows:                {n}")
    print(f"  full_match (old):    {ok_old/n:.4f} ({ok_old}/{n})")
    print(f"  full_match (new):    {ok_new/n:.4f} ({ok_new}/{n})")
    print(f"  delta:               {(ok_new - ok_old)/n:+.4f} ({ok_new - ok_old:+d} rows)")
    print()
    if flips:
        print(f"  Flipped rows (old → new) — {len(flips)} total:")
        for idx, gt_t, raw, old, new in flips[:10]:
            arrow = "→ HIT" if new else "→ MISS"
            print(f"    row {idx:>3}  gt={gt_t:<10} pred.action={raw!r:<24} {arrow}")
        if len(flips) > 10:
            print(f"    ... +{len(flips) - 10} more")
    print()
    print("  Per-type (new):")
    for k, v in sorted(per_type_new.items(), key=lambda x: -x[1]['n']):
        print(f"    {k:>16}: n={v['n']:>3}  acc={v['ok']/max(v['n'],1):.3f}")

    return {
        "path": str(eval_json_path),
        "rows": n,
        "full_match_old": ok_old / n,
        "full_match_new": ok_new / n,
        "delta": (ok_new - ok_old) / n,
        "per_type_new": per_type_new,
        "n_flipped": len(flips),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_json", type=Path)
    args = ap.parse_args()
    rescore(args.eval_json)


if __name__ == "__main__":
    main()
