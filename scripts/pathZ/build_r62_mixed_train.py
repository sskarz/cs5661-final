#!/usr/bin/env python3
"""Build an optional r62 mixed Genesis train JSONL.

This is intentionally not used by the r62 launcher. It combines prior
Genesis text-only rows with screenshot-aligned Genesis vision rows while
stamping provenance on every output row.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


DEFAULT_TEXT = Path("data/pathZ/genesis/train_r51_final.jsonl")
DEFAULT_VISION = Path("data/pathZ/genesis_vision_rebuild/train.expanded.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def stamp(row: dict, source: str) -> dict:
    out = dict(row)
    if "_source" in out:
        out["_source_original"] = out["_source"]
    out["_source"] = source
    return out


def apply_limit(rows: list[dict], limit: int | None) -> list[dict]:
    if limit is None:
        return rows
    return rows[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-jsonl", type=Path, default=DEFAULT_TEXT)
    ap.add_argument("--vision-jsonl", type=Path, default=DEFAULT_VISION)
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--text-per-vision", type=float, default=1.0,
                    help="Keep up to ceil(vision_rows * ratio) text rows.")
    ap.add_argument("--text-limit", type=int, default=None,
                    help="Hard cap after applying --text-per-vision.")
    ap.add_argument("--vision-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--no-shuffle", action="store_true")
    args = ap.parse_args()

    if args.text_per_vision < 0:
        raise ValueError("--text-per-vision must be non-negative")
    if args.text_limit is not None and args.text_limit < 0:
        raise ValueError("--text-limit must be non-negative")
    if args.vision_limit is not None and args.vision_limit < 0:
        raise ValueError("--vision-limit must be non-negative")

    text_rows = read_jsonl(args.text_jsonl)
    vision_rows = read_jsonl(args.vision_jsonl)

    vision_rows = apply_limit(vision_rows, args.vision_limit)
    ratio_text_limit = int(len(vision_rows) * args.text_per_vision + 0.999999)
    text_keep = min(len(text_rows), ratio_text_limit)
    if args.text_limit is not None:
        text_keep = min(text_keep, args.text_limit)
    text_rows = text_rows[:text_keep]

    mixed = (
        [stamp(r, "genesis_r51_text") for r in text_rows]
        + [stamp(r, "genesis_vision_rebuild") for r in vision_rows]
    )
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(mixed)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as f:
        for row in mixed:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"[build_r62_mixed] text={len(text_rows)} vision={len(vision_rows)} "
          f"total={len(mixed)} output={args.output_jsonl}")


if __name__ == "__main__":
    main()
