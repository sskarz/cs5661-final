#!/usr/bin/env python3
"""Build vision training data from genesis trajectories with screenshots.

Reads genesis trajectories collected with --save-screenshots and converts
them into M3A-format SFT rows with aligned (screenshot, action) pairs.

Each training row contains:
  - The raw screenshot PNG captured BEFORE the teacher's action was taken
  - The teacher's action prompt (with UI elements + history)
  - The teacher's reasoning + action emission

Usage:
  # Convert genesis trajectories (must have image field in steps):
  python scripts/pathZ/genesis_vision_train.py \\
      --src data/pathZ/genesis/train.jsonl \\
      --out data/pathZ/genesis/vision_train.jsonl \\
      --seed 3407

  # Mix with existing smoke data (50/50 per action class):
  python scripts/pathZ/genesis_vision_train.py \\
      --src data/pathZ/genesis/train.jsonl \\
      --mix data/pathZ/smoke/train.jsonl \\
      --out data/pathZ/genesis/vision_mixed.jsonl \\
      --per-class-target 250

  # Train directly on vision data:
  python scripts/pathZ/train_smoke.py \\
      --train-jsonl data/pathZ/genesis/vision_train.jsonl \\
      --data-dir data/pathZ/genesis/screenshots \\
      --max-steps 400
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def _validate_screenshot(img_root: Path, rel_path: str) -> bool:
    """Check if a screenshot file exists and is valid."""
    p = img_root / rel_path
    if not p.exists():
        return False
    # Quick check: file must be non-zero size
    if p.stat().st_size < 100:
        return False
    return True


def _extract_action(emission: str) -> dict | None:
    """Extract the action JSON from a teacher emission string."""
    # Find "Action: {...}" pattern
    idx = emission.find("Action:")
    if idx < 0:
        return None
    action_text = emission[idx + len("Action:"):].strip()
    # Find balanced braces
    depth = 0
    start = -1
    for i, c in enumerate(action_text):
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(action_text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _build_vision_row(genesis_row: dict, image_root: Path | None) -> dict | None:
    """Convert one genesis training row to a vision-compatible SFT row.

    The output format matches what prepare_smoke_data.py produces:
    - messages with [image, text] in user content
    - image field with relative path
    - _image_root pointing to screenshots directory
    - gt_m3a with the action JSON
    """
    # Extract prompt and emission from messages
    user_content = genesis_row.get("messages", [{}])[0].get("content", [])
    assistant_content = genesis_row.get("messages", [{}, {}])[1].get("content", [])

    prompt_text = None
    for c in user_content:
        if c.get("type") == "text":
            prompt_text = c.get("text", "")
            break

    emission_text = None
    for c in assistant_content:
        if c.get("type") == "text":
            emission_text = c.get("text", "")
            break

    if not prompt_text or not emission_text:
        return None

    # Get screenshot info
    img_rel = genesis_row.get("image")
    gt_m3a = genesis_row.get("gt_m3a") or _extract_action(emission_text)

    if not gt_m3a:
        return None

    # Build vision row matching prepare_smoke_data.py format
    row = {
        "messages": [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": emission_text},
            ]},
        ],
        "image": img_rel,
        "gt_m3a": gt_m3a,
        "_image_root": str(image_root) if image_root else None,
        "_source": "genesis",
        "_genesis_app": genesis_row.get("_genesis_app"),
        "_genesis_goal": genesis_row.get("_genesis_goal"),
        "_genesis_step": genesis_row.get("_genesis_step"),
        "_genesis_bucket": genesis_row.get("_genesis_bucket"),
    }

    # Only include rows that have valid screenshots
    if img_rel and image_root:
        if not _validate_screenshot(image_root, img_rel):
            return None
    else:
        # No screenshot available — skip for vision training
        return None

    return row


def main():
    ap = argparse.ArgumentParser(
        description="Build vision training data from genesis screenshots"
    )
    ap.add_argument("--src", type=Path,
                    default=Path("data/pathZ/genesis/train.jsonl"),
                    help="Genesis training JSONL from genesis_to_train.py")
    ap.add_argument("--out", type=Path,
                    default=Path("data/pathZ/genesis/vision_train.jsonl"),
                    help="Output vision training JSONL")
    ap.add_argument("--image-root", type=Path, default=None,
                    help="Absolute path to screenshots directory. "
                         "Auto-detected from --src if not provided.")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="Cap total output rows (0 = unlimited)")
    ap.add_argument("--per-app-max", type=int, default=0,
                    help="Cap rows per app (0 = unlimited)")

    # Mixing with existing smoke data
    ap.add_argument("--mix", type=Path, default=None,
                    help="Path to existing smoke train.jsonl to mix with. "
                         "Genesis rows are added to the same file, balanced "
                         "per action class.")
    ap.add_argument("--per-class-target", type=int, default=250,
                    help="Target rows per action type when mixing.")
    args = ap.parse_args()

    src = args.src
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    # Resolve image root
    image_root = args.image_root
    if image_root is None:
        # Auto-detect: look for screenshots/ next to the source
        for candidate in [
            src.parent / "screenshots",
            Path("data/pathZ/genesis/screenshots"),
        ]:
            if candidate.exists():
                image_root = candidate.resolve()
                break

    if image_root is None:
        print(f"[genesis_vision] ERROR: no screenshots directory found. "
              f"Run genesis_synth.py --save-screenshots first, or pass --image-root.")
        print(f"[genesis_vision] looked in: {src.parent / 'screenshots'}, "
              f"data/pathZ/genesis/screenshots")
        raise SystemExit(1)

    print(f"[genesis_vision] image_root = {image_root}")

    # Load genesis rows
    genesis_rows = []
    with open(src) as f:
        for line in f:
            row = json.loads(line)
            genesis_rows.append(row)
    print(f"[genesis_vision] loaded {len(genesis_rows)} genesis rows")

    # Convert to vision format
    vision_rows = []
    n_skipped = 0
    by_app: dict[str, list[dict]] = defaultdict(list)
    for row in genesis_rows:
        vrow = _build_vision_row(row, image_root)
        if vrow is None:
            n_skipped += 1
            continue
        app = vrow.get("_genesis_app", "unknown")
        by_app[app].append(vrow)
        vision_rows.append(vrow)

    print(f"[genesis_vision] converted {len(vision_rows)} vision rows "
          f"(skipped {n_skipped} without valid screenshots)")

    # Apply per-app cap
    if args.per_app_max > 0:
        capped = []
        for app, rows in by_app.items():
            rng = random.Random(args.seed)
            n = min(args.per_app_max, len(rows))
            capped.extend(rng.sample(rows, n))
        by_app = defaultdict(list)
        vision_rows = capped
        for r in vision_rows:
            app = r.get("_genesis_app", "unknown")
            by_app[app].append(r)
        print(f"[genesis_vision] capped to {args.per_app_max} per app: "
              f"{len(vision_rows)} total")

    # Per-app distribution
    print("[genesis_vision] per-app vision rows:")
    for app, rows in sorted(by_app.items(), key=lambda kv: -len(kv[1])):
        print(f"  {app:32s}  {len(rows):4d}")

    # Action type distribution
    action_types = Counter(r["gt_m3a"]["action_type"] for r in vision_rows)
    print(f"[genesis_vision] action_type distribution: {dict(action_types.most_common())}")

    # Mixing with existing smoke data
    if args.mix:
        print(f"[genesis_vision] mixing with {args.mix}...")
        mix_rows = []
        with open(args.mix) as f:
            for line in f:
                mix_rows.append(json.loads(line))

        # Bucket both sources by action type
        genesis_buckets: dict[str, list[dict]] = defaultdict(list)
        for r in vision_rows:
            genesis_buckets[r["gt_m3a"]["action_type"]].append(r)

        smoke_buckets: dict[str, list[dict]] = defaultdict(list)
        for r in mix_rows:
            gt = r.get("gt_m3a", {})
            if gt:
                smoke_buckets[gt["action_type"]].append(r)

        rng = random.Random(args.seed)
        mixed = []
        all_classes = set(list(genesis_buckets.keys()) + list(smoke_buckets.keys()))

        for at in sorted(all_classes):
            g_rows = genesis_buckets.get(at, [])
            s_rows = smoke_buckets.get(at, [])
            target = args.per_class_target

            # Take half from each source, capped at available
            half = target // 2
            if g_rows:
                n_g = min(half, len(g_rows))
                picked_g = rng.sample(g_rows, n_g) if len(g_rows) >= n_g \
                    else [rng.choice(g_rows) for _ in range(n_g)]
            else:
                picked_g = []

            if s_rows:
                n_s = min(target - half, len(s_rows))
                picked_s = rng.sample(s_rows, n_s) if len(s_rows) >= n_s \
                    else [rng.choice(s_rows) for _ in range(n_s)]
            else:
                picked_s = []

            batch = picked_g + picked_s
            rng.shuffle(batch)
            mixed.extend(batch)
            print(f"  {at:18s} genesis={len(picked_g):3d} smoke={len(picked_s):3d} "
                  f"total={len(batch):3d}")

        vision_rows = mixed

    # Apply max-rows cap
    if args.max_rows > 0:
        rng = random.Random(args.seed)
        vision_rows = rng.sample(vision_rows, min(args.max_rows, len(vision_rows)))

    # Write output
    with open(out, "w") as f:
        for row in vision_rows:
            f.write(json.dumps(row) + "\n")

    print(f"[genesis_vision] wrote {len(vision_rows)} rows to {out}")
    print(f"METRIC genesis_vision_rows={len(vision_rows)}")
    print(f"METRIC genesis_vision_apps={len(by_app)}")


if __name__ == "__main__":
    main()
