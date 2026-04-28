#!/usr/bin/env python3
"""Re-encode AndroidControl as a11y-native action format (Path W).

Reads data/androidcontrol_a11y/{train,val,test}.jsonl produced by
parse_a11y_data.py (which embeds the `a11y` node list per row), and rewrites
them so that:

1. The user prompt includes a numbered list of clickable elements (the same
   sort order as the SoM renderer for consistency), so the model can refer
   to them by index.
2. For tap/long_press actions, the assistant target becomes:
       {"action": "tap", "element_id": <int>}
   where `element_id` is the index into the listed-elements legend.
   The original (x,y) is matched to the smallest containing bbox at prep time.
3. Other action types pass through unchanged.

This is SeeClick's framing: the model predicts text/integer identifiers
instead of coordinates, which is what VLMs do natively. At eval time we
translate `element_id` → bbox centroid → (x,y) and score with the existing
tap-radius distance metric.

Outputs to data/androidcontrol_a11y_native/{train,val,test}.jsonl alongside
images/ symlinked back to the source dir.

Usage:
    uv run python scripts/prepare_a11y_native.py \\
        --src-dir data/androidcontrol_a11y \\
        --output-dir data/androidcontrol_a11y_native \\
        --max-elements 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Reuse the SoM priority + label ordering so train and SoM-eval see the same legend.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from render_som import _short_label, _bbox_area, filter_and_order_nodes  # noqa: E402


NEAREST_FALLBACK_RADIUS = 0.04  # ~24 px on a 1080-tall screen — tight enough to
# avoid teaching the model wrong-element fires when GT lands between elements.


def find_containing_node(x: float, y: float, ordered: list[dict]) -> int | None:
    """Return the 1-indexed element_id whose bbox contains (x,y).

    Tie-break: clickable elements first, then SMALLEST containing bbox. A
    label TextView nested inside a clickable Button shares (x,y) but is not
    the actual tap target — without the clickable-first preference we'd
    teach the model the wrong index. Falls back to nearest centroid only
    within NEAREST_FALLBACK_RADIUS.
    """
    contains: list[tuple[int, float, int]] = []  # (clickable_priority, area, idx)
    nearest: list[tuple[float, int]] = []
    for i, n in enumerate(ordered, start=1):
        b = n.get("bbox") or [0, 0, 0, 0]
        clickable = bool(n.get("is_clickable") or n.get("is_long_clickable"))
        if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
            contains.append((0 if clickable else 1, _bbox_area(b), i))
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        nearest.append(((cx - x) ** 2 + (cy - y) ** 2, i))
    if contains:
        contains.sort()
        return contains[0][2]
    if nearest:
        nearest.sort()
        dist = nearest[0][0] ** 0.5
        return nearest[0][1] if dist < NEAREST_FALLBACK_RADIUS else None
    return None


def build_user_prompt(instruction: str, ordered: list[dict]) -> str:
    legend = "\n".join(
        f"  {i}: {_short_label(n)}"
        for i, n in enumerate(ordered, start=1)
    )
    return (
        f"Task: {instruction}\n\n"
        f"Below is a list of UI elements visible on screen, numbered.\n"
        f"Output a single JSON action.\n"
        f"For tap/long_press: {{\"action\":\"tap\",\"element_id\":<int>}}\n"
        f"For type: also include text. For scroll: include direction. "
        f"For navigate_back/navigate_home/wait/done: no element_id.\n\n"
        f"Elements:\n{legend}"
    )


def transform_row(
    row: dict, max_elements: int, stats: dict, min_side_px: int = 16
) -> dict | None:
    nodes = row.get("a11y") or []
    if not nodes:
        stats["dropped_no_a11y"] += 1
        return None
    img_w = int(row.get("image_w") or 0)
    img_h = int(row.get("image_h") or 0)
    if img_w <= 0 or img_h <= 0:
        # Old parsed rows (pre-image_w/h schema) — fall back to legacy
        # behavior (no px filter). Re-parse to pick up the new contract.
        stats["fallback_no_dims"] += 1
        ordered = sorted(
            [n for n in nodes if n.get("bbox") and len(n["bbox"]) == 4],
            key=lambda n: (n.get("is_clickable") and 0 or 1, _bbox_area(n["bbox"])),
        )[:max_elements]
    else:
        ordered = filter_and_order_nodes(nodes, img_w, img_h,
                                         max_marks=max_elements, min_side_px=min_side_px)
    if not ordered:
        stats["dropped_empty"] += 1
        return None

    gt = json.loads(row["messages"][1]["content"][0]["text"])
    gt_t = (gt.get("action") or "").lower()
    new_action: dict = dict(gt)

    gt_xy = None
    # parse_a11y_data.py canonicalizes click/tap/long_press → "tap". This
    # script trusts that contract; if upstream ever changes, the assertion
    # below will catch it (no silent long_press leakage).
    if gt_t == "tap":
        x = gt.get("x"); y = gt.get("y")
        if x is None or y is None:
            stats["dropped_tap_no_xy"] += 1
            return None
        eid = find_containing_node(float(x), float(y), ordered)
        if eid is None:
            stats["dropped_tap_no_match"] += 1
            return None
        new_action = {"action": "tap", "element_id": eid}
        gt_xy = [round(float(x), 4), round(float(y), 4)]
        stats["tap_matched"] += 1
    else:
        # Non-tap rows still get the legend (model needs to see UI for context),
        # but we strip x/y from the action and pass through other fields.
        new_action.pop("x", None); new_action.pop("y", None)
        stats["non_tap"] += 1

    instr = row["messages"][0]["content"][1]["text"]
    user_text = build_user_prompt(instr, ordered)

    out_row = {
        "messages": [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": json.dumps(new_action)},
            ]},
        ],
        "episode_id": row["episode_id"],
        "step_index": row["step_index"],
        "total_steps": row.get("total_steps"),
        "granularity": row.get("granularity", "step"),
        "image": row["image"],
        "goal": row.get("goal"),
        # Carry the ordered element list in the row so eval can translate
        # element_id → bbox centroid without recomputing.
        "elements": [
            {"id": i, "bbox": n["bbox"], "label": _short_label(n)}
            for i, n in enumerate(ordered, start=1)
        ],
    }
    # Preserve original (x,y) on tap rows for tap-radius distance scoring
    # during eval; model still trains on element_id.
    if gt_xy is not None:
        out_row["gt_xy"] = gt_xy
    return out_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=Path, default=Path("data/androidcontrol_a11y"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/androidcontrol_a11y_native"))
    ap.add_argument("--max-elements", type=int, default=40)
    ap.add_argument("--min-side-px", type=int, default=16,
                    help="Min element side in pixels (must match render_som default).")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    img_link = args.output_dir / "images"
    src_images = (args.src_dir / "images").resolve()
    if not src_images.is_dir():
        raise SystemExit(
            f"Source images dir not found: {src_images}. "
            f"Run parse_a11y_data.py before prepare_a11y_native.py."
        )
    # Refresh symlink if it points at a stale location (different src-dir).
    if img_link.is_symlink() or img_link.exists():
        if not img_link.is_symlink() or img_link.resolve() != src_images:
            if img_link.is_symlink():
                img_link.unlink()
            else:
                raise SystemExit(f"{img_link} exists and is not a symlink; refusing to clobber.")
            img_link.symlink_to(src_images)
    else:
        img_link.symlink_to(src_images)

    overall: dict[str, int] = {}
    for split in ("train", "val", "test"):
        src = args.src_dir / f"{split}.jsonl"
        if not src.exists():
            print(f"[skip] {src} missing")
            continue
        dst = args.output_dir / f"{split}.jsonl"
        stats: dict[str, int] = {
            "in": 0, "out": 0, "dropped_no_a11y": 0, "dropped_empty": 0,
            "dropped_tap_no_xy": 0, "dropped_tap_no_match": 0,
            "tap_matched": 0, "non_tap": 0, "fallback_no_dims": 0,
        }
        with open(src) as fin, open(dst, "w") as fout:
            for line in fin:
                stats["in"] += 1
                row = json.loads(line)
                new = transform_row(row, args.max_elements, stats,
                                    min_side_px=args.min_side_px)
                if new is None:
                    continue
                fout.write(json.dumps(new) + "\n")
                stats["out"] += 1
        for k, v in stats.items():
            overall[k] = overall.get(k, 0) + v
        print(f"[{split}] {stats}")

    print(f"\n[total] {overall}")
    print(f"  output -> {args.output_dir}")


if __name__ == "__main__":
    main()
