"""Set-of-Mark (SoM) renderer for AndroidControl screenshots.

Given a screenshot + a list of a11y nodes (with normalized [0,1] bboxes),
overlay numbered marks at each clickable element's centroid. The renderer
returns the marked image and a `marks` table that maps mark_id -> (cx, cy)
in normalized coords. The eval harness uses this to translate model output
"tap mark 7" into a click coord, which then goes through the existing
distance-radius matching.

Yang et al. 2023 showed SoM gives big lifts on GPT-4V; the underlying
mechanism (reduce dense regression to multiple-choice over a small set of
proposals) is model-scale-agnostic, so it should help Gemma 4 E2B too.
For us, the mark candidates come from GT a11y bboxes (perfect by
construction), which sidesteps the mark-detector quality risk.

Usage as a library:
    from render_som import render_marks, build_mark_prompt
    img, marks = render_marks(pil_img, nodes, max_marks=40)
    # marks: list[dict] with id, bbox, cx, cy, label_short

Standalone:
    uv run python scripts/render_som.py \\
        --jsonl data/androidcontrol_a11y/test.jsonl \\
        --out-dir outputs/som_smoke --num 5
    Renders 5 sample marked screenshots for visual inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Bright contrasting colors for mark numerals (cycled).
MARK_COLORS = [
    (255, 50, 50),    # red
    (50, 200, 50),    # green
    (50, 100, 255),   # blue
    (255, 150, 0),    # orange
    (200, 50, 200),   # magenta
    (50, 200, 200),   # cyan
    (255, 220, 50),   # yellow
    (150, 50, 255),   # violet
]


def _node_priority(n: dict) -> tuple:
    """Sort nodes so most-likely-actionable elements get the lowest mark IDs.

    Returns a sort key — lower sorts FIRST (smaller mark id, more visible).
    Priority: clickable > editable > text-bearing > other; smaller bbox first
    (small icons matter more than huge containers); shallower depth first.
    """
    has_text = bool(n.get("text") or n.get("content_description"))
    return (
        0 if n.get("is_clickable") else (1 if n.get("is_editable") else (2 if has_text else 3)),
        n.get("depth", 0),
        _bbox_area(n.get("bbox") or [0, 0, 0, 0]),  # smaller first
    )


def _bbox_area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _short_label(n: dict, max_len: int = 24) -> str:
    for k in ("text", "content_description", "view_id_resource_name"):
        v = (n.get(k) or "").strip()
        if v:
            return v[:max_len]
    cls = (n.get("class_name") or "").split(".")[-1]
    return cls or "?"


def filter_and_order_nodes(
    nodes: list[dict],
    img_w: int,
    img_h: int,
    max_marks: int = 40,
    min_side_px: int = 16,
) -> list[dict]:
    """Canonical filter+sort used by BOTH SoM render and a11y-native prep.

    Path X (render_marks) and Path W (prepare_a11y_native) MUST produce the
    same ordering for the same input nodes — otherwise mark `N` at train time
    refers to a different element than mark `N` at SoM eval. This is the
    single source of truth.
    """
    keep = []
    for n in nodes:
        b = n.get("bbox")
        if not b or len(b) != 4:
            continue
        x0, y0, x1, y1 = b
        if (x1 - x0) * img_w < min_side_px or (y1 - y0) * img_h < min_side_px:
            continue
        keep.append(n)
    keep.sort(key=_node_priority)
    return keep[:max_marks]


def render_marks(
    img: Image.Image,
    nodes: list[dict],
    max_marks: int = 40,
    min_side_px: int = 16,
) -> tuple[Image.Image, list[dict]]:
    """Overlay numbered marks on a copy of `img`.

    Returns (marked_image_RGB, marks). `marks` is a list of:
        {"id": int, "bbox": [x0,y0,x1,y1] norm, "cx": float, "cy": float,
         "label": str}

    `id` is the mark number printed on the image (1-indexed).
    Filters elements smaller than `min_side_px` on either axis (illegible).
    """
    W, H = img.size
    base = img.convert("RGB").copy()
    draw = ImageDraw.Draw(base, "RGBA")

    keep = filter_and_order_nodes(nodes, W, H, max_marks=max_marks, min_side_px=min_side_px)

    # Use default font; PIL ships a small bitmap, but we scale via radius.
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    marks: list[dict] = []
    for i, n in enumerate(keep, start=1):
        b = n["bbox"]
        x0, y0, x1, y1 = b[0] * W, b[1] * H, b[2] * W, b[3] * H
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        color = MARK_COLORS[(i - 1) % len(MARK_COLORS)]

        # Box outline (semi-transparent fill).
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        # Number badge in top-left of element.
        r = 14
        bx, by = x0 + 2, y0 + 2
        draw.ellipse([bx, by, bx + 2 * r, by + 2 * r], fill=(255, 255, 255, 230), outline=color, width=2)
        txt = str(i)
        tl, tt, tr_, tb = draw.textbbox((0, 0), txt, font=font)
        tw, th = tr_ - tl, tb - tt
        draw.text((bx + r - tw / 2 - tl, by + r - th / 2 - tt), txt, font=font, fill=color)

        marks.append({
            "id": i,
            "bbox": [round(b[0], 4), round(b[1], 4), round(b[2], 4), round(b[3], 4)],
            "cx": round((b[0] + b[2]) / 2, 4),
            "cy": round((b[1] + b[3]) / 2, 4),
            "label": _short_label(n),
        })

    return base, marks


def build_mark_prompt(instruction: str, marks: list[dict]) -> str:
    """Format the user prompt for the SoM eval.

    The model sees the marked image plus a numbered list of element labels.
    It must respond in our action JSON, but with `mark` instead of `x`/`y`
    for tap actions:
        {"action": "tap", "mark": 7}
        {"action": "scroll", "direction": "up"}
        {"action": "type", "text": "hello", "mark": 3}
    """
    legend = "\n".join(f"  {m['id']}: {m['label']}" for m in marks)
    return (
        f"Task: {instruction}\n\n"
        f"The screenshot shows numbered marks on every clickable element. "
        f"Output a JSON action.\n"
        f"For tap/long_press: include {{\"mark\": <int>}} from the list.\n"
        f"For type: also include text. For scroll: include direction. "
        f"For navigate_back/navigate_home/wait/done: no mark.\n"
        f"Output the action and nothing else.\n\n"
        f"Available marks:\n{legend}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True,
                    help="A11y-augmented JSONL (data/androidcontrol_a11y/test.jsonl).")
    ap.add_argument("--data-dir", type=Path,
                    help="Where images/ lives. Defaults to parent of --jsonl.")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/som_smoke"))
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--max-marks", type=int, default=40)
    args = ap.parse_args()

    data_dir = args.data_dir or args.jsonl.parent
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_done = 0
    with open(args.jsonl) as f:
        for line in f:
            row = json.loads(line)
            if not row.get("a11y"):
                continue
            img_path = data_dir / row["image"]
            if not img_path.exists():
                continue
            img = Image.open(img_path)
            marked, marks = render_marks(img, row["a11y"], max_marks=args.max_marks)
            base = f"{row['episode_id']}_{row['step_index']}"
            marked.save(args.out_dir / f"{base}_marked.png")
            (args.out_dir / f"{base}_marks.json").write_text(json.dumps(marks, indent=2))
            (args.out_dir / f"{base}_prompt.txt").write_text(
                build_mark_prompt(row["messages"][0]["content"][1]["text"], marks)
            )
            print(f"[{n_done + 1}] {base}: {len(marks)} marks rendered")
            n_done += 1
            if n_done >= args.num:
                break


if __name__ == "__main__":
    main()
