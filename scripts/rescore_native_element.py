#!/usr/bin/env python3
"""Re-score eval_a11y_native JSONs using element-accuracy instead of tap-radius.

Tap-radius scoring projects predicted element_id → bbox center, then checks
whether the projected (x,y) is within 0.14 normalized of GT (x,y). This caps
at the projection ceiling (0.858 for typical AndroidControl bboxes — wide
list-items have centers far from edge taps).

Element-accuracy compares predicted element_id directly to the GT element_id
embedded in the canonicalized assistant target. For non-tap actions the
metric is unchanged (direction match for scroll, text match for type, etc.).
This is the natural metric for a11y-aware UI agents and matches how
AndroidControl-paper baselines that consume a11y trees report numbers.

Usage:
    uv run python scripts/rescore_native_element.py \\
        outputs/eval/runI_ckpt*.json \\
        --out outputs/eval/element_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Normalize "scroll down" / "scroll_up" / etc. into canonical scroll+direction.
_SCROLL_DIR_RE = re.compile(r"^scroll[\s_\-]*(up|down|left|right)$", re.IGNORECASE)


def _flatten_v2(action: dict) -> dict:
    """v2 -> v1 schema; idempotent on v1. See eval_a11y_native._flatten_v2."""
    if not isinstance(action, dict) or not action:
        return action
    if "action_type" in action and "action_args" in action:
        out = {"action": action["action_type"]}
        args = action.get("action_args") or {}
        if isinstance(args, dict):
            out.update(args)
        return out
    return action


def _normalize(pred: dict) -> dict:
    out = dict(pred)
    raw = (out.get("action") or out.get("action_type") or "").strip()
    m = _SCROLL_DIR_RE.match(raw)
    if m:
        out["action"] = "scroll"
        if not out.get("direction"):
            out["direction"] = m.group(1).lower()
    return out


def element_match(pred: dict, gt: dict) -> bool:
    """Pure element-accuracy: matches action_type, then matches the
    type-specific identifier (element_id for taps, direction for scrolls, etc.).
    No coordinate projection."""
    pred = _normalize(pred)
    p_t = (pred.get("action") or pred.get("action_type") or "").lower()
    g_t = (gt.get("action") or "").lower()
    if not p_t or p_t != g_t:
        return False
    if g_t in ("tap", "long_press"):
        try:
            return int(pred.get("element_id")) == int(gt.get("element_id"))
        except (TypeError, ValueError):
            return False
    if g_t == "scroll":
        return (pred.get("direction") or "").lower() == (gt.get("direction") or "").lower()
    if g_t == "type":
        return (pred.get("text") or "").strip() == (gt.get("text") or "").strip()
    if g_t == "open_app":
        return (pred.get("app_name") or "").strip().lower() == (gt.get("app_name") or "").strip().lower()
    return True  # navigate_back / navigate_home / wait / done — action_type alone


def rescore_file(path: Path) -> dict:
    raw = json.loads(path.read_text())
    preds = raw.get("all_predictions") or []
    if not preds:
        return {"path": str(path), "skipped": "no all_predictions"}

    ok_radius = sum(1 for p in preds if p.get("ok"))
    ok_element = 0
    per_type: dict[str, dict] = {}
    confusion_element: dict[str, dict] = {}

    for p in preds:
        gt = _flatten_v2(p.get("gt") or {})
        pred = _flatten_v2(p.get("pred") or {})
        gt_t = (gt.get("action") or "").lower()
        ok_e = element_match(pred, gt)
        ok_element += int(ok_e)
        per_type.setdefault(gt_t, {"n": 0, "ok": 0})
        per_type[gt_t]["n"] += 1
        per_type[gt_t]["ok"] += int(ok_e)
        p_norm = _normalize(pred)
        p_t = (p_norm.get("action") or "").lower() or "<parse_fail>"
        confusion_element.setdefault(gt_t, {}).setdefault(p_t, 0)
        confusion_element[gt_t][p_t] += 1

    n = len(preds)
    return {
        "path": str(path),
        "n": n,
        "full_match_radius": ok_radius / n,
        "full_match_element": ok_element / n,
        "delta": (ok_element - ok_radius) / n,
        "per_type_element": {k: {"n": v["n"], "correct": v["ok"],
                                  "accuracy": v["ok"] / max(v["n"], 1)}
                              for k, v in per_type.items()},
        "confusion_element": confusion_element,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_jsons", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional: write aggregated summary JSON here.")
    ap.add_argument("--label", type=str, default=None,
                    help="Optional label for printed table (e.g. 'sweep', 'fulltest').")
    args = ap.parse_args()

    results = []
    for p in args.eval_jsons:
        r = rescore_file(p)
        if "skipped" in r:
            print(f"  skip: {p} ({r['skipped']})")
            continue
        results.append(r)

    # Sort by ckpt number when filename contains it, else alphabetic.
    def _key(r):
        m = re.search(r"ckpt(\d+)", r["path"])
        return int(m.group(1)) if m else 1 << 30

    results.sort(key=_key)

    print(f"\n=== Element-accuracy rescore ({args.label or 'all'}) ===")
    print(f"{'file':<55} {'n':>5} {'radius':>8} {'element':>8}  Δ")
    for r in results:
        name = Path(r["path"]).name
        print(f"{name:<55} {r['n']:>5} "
              f"{r['full_match_radius']:>8.4f} {r['full_match_element']:>8.4f}  "
              f"{r['delta']:+.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nWrote summary -> {args.out}")


if __name__ == "__main__":
    main()
