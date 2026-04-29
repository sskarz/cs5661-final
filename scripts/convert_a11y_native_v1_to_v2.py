#!/usr/bin/env python3
"""Convert AndroidControl a11y-native dataset from v1 to v2 schema (Run K).

v1 target: {"action": "tap", "element_id": 7}
v2 target: {"action_type": "tap", "action_args": {"element_id": 7}}

The user-prompt schema instruction is rewritten in lockstep so the model
sees a consistent v2 schema in both the prompt and the assistant target.

v2 = CoCo-Agent CAP-style hierarchical decoding (Ma et al., ACL 2024,
arXiv:2402.11941). See TRAINING_LOG.md §39 for rationale.

Usage:
    # Dry-run: convert 3 train rows, print, exit. NO files written.
    uv run python scripts/convert_a11y_native_v1_to_v2.py --dry-run

    # Full conversion -> data/androidcontrol_a11y_native_v2/
    uv run python scripts/convert_a11y_native_v1_to_v2.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Exact v1 prompt-schema substring (must match scripts/prepare_a11y_native.py:81-87 verbatim).
V1_PROMPT_SCHEMA = (
    "Below is a list of UI elements visible on screen, numbered.\n"
    "Output a single JSON action.\n"
    "For tap/long_press: {\"action\":\"tap\",\"element_id\":<int>}\n"
    "For type: also include text. For scroll: include direction. "
    "For navigate_back/navigate_home/wait/done: no element_id."
)

V2_PROMPT_SCHEMA = (
    "Below is a list of UI elements visible on screen, numbered.\n"
    "Output a single JSON action with two keys: \"action_type\" (string) and \"action_args\" (object).\n"
    "For tap/long_press: {\"action_type\":\"tap\",\"action_args\":{\"element_id\":<int>}}\n"
    "For type: action_args includes \"text\". For scroll: action_args includes \"direction\". "
    "For open_app: action_args includes \"app_name\". "
    "For navigate_back/navigate_home/wait: action_args is the empty object {}."
)

VALID_ACTION_TYPES = {
    "tap", "long_press", "scroll", "type", "open_app",
    "navigate_back", "navigate_home", "wait",
}


def convert_action_v1_to_v2(action_v1: dict) -> dict:
    """{action: X, ...args...} -> {action_type: X, action_args: {...}}.

    Pulls 'action' out as the new top-level type; everything else becomes args.
    Empty action_args object {} for navigate_back/navigate_home/wait.
    """
    a = dict(action_v1)
    action_type = a.pop("action", None)
    if action_type is None:
        raise ValueError(f"v1 action missing 'action' key: {action_v1}")
    if action_type not in VALID_ACTION_TYPES:
        raise ValueError(f"unknown action_type {action_type!r}: {action_v1}")
    return {"action_type": action_type, "action_args": a}


def convert_row(row: dict) -> dict:
    user_text = row["messages"][0]["content"][1]["text"]
    if V1_PROMPT_SCHEMA not in user_text:
        raise ValueError(
            f"row {row.get('episode_id')}/{row.get('step_index')}: "
            f"v1 schema substring not found in user prompt — prep script may have drifted"
        )
    row["messages"][0]["content"][1]["text"] = user_text.replace(
        V1_PROMPT_SCHEMA, V2_PROMPT_SCHEMA
    )
    target_v1 = json.loads(row["messages"][1]["content"][0]["text"])
    target_v2 = convert_action_v1_to_v2(target_v1)
    row["messages"][1]["content"][0]["text"] = json.dumps(target_v2)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=Path, default=Path("data/androidcontrol_a11y_native"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/androidcontrol_a11y_native_v2"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--dry-run", action="store_true",
                    help="Convert and print first 3 train rows; do NOT write files.")
    args = ap.parse_args()

    if args.dry_run:
        src = args.src_dir / "train.jsonl"
        with open(src) as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                row = convert_row(json.loads(line))
                print(f"=== sample row {i} (ep={row['episode_id']} step={row['step_index']}) ===")
                print(f"USER PROMPT (first 600 chars):\n{row['messages'][0]['content'][1]['text'][:600]}...\n")
                print(f"ASSISTANT TARGET:\n  {row['messages'][1]['content'][0]['text']}\n")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Symlink images dir (large, no copy needed).
    src_imgs = args.src_dir / "images"
    out_imgs = args.out_dir / "images"
    if src_imgs.exists() and not out_imgs.exists():
        out_imgs.symlink_to(src_imgs.resolve())
        print(f"[v2-prep] symlinked images: {out_imgs} -> {src_imgs}")

    for split in args.splits:
        src = args.src_dir / f"{split}.jsonl"
        out = args.out_dir / f"{split}.jsonl"
        if not src.exists():
            print(f"[v2-prep] [skip] {src} missing")
            continue
        type_counts: Counter[str] = Counter()
        n = 0
        with open(src) as fin, open(out, "w") as fout:
            for line in fin:
                row = convert_row(json.loads(line))
                t = json.loads(row["messages"][1]["content"][0]["text"])["action_type"]
                type_counts[t] += 1
                fout.write(json.dumps(row) + "\n")
                n += 1
        print(f"[v2-prep] {split}: {n} rows, types={dict(type_counts)}")

    print(f"[v2-prep] done: {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
