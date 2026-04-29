#!/usr/bin/env python3
"""Transform v2 a11y-native dataset into v3 with prior-action history (Run L).

For each row, look up the prior step's target action within the same episode
and prepend it to the user prompt as a "Previous action:" line. Step 0 (no
prior by design) gets "Previous action: <none>".

Coverage policy (validated on v2 dataset):
  - step_index == 0                     -> <none>            (~18% of rows)
  - (episode_id, step_index - 1) exists -> that target       (~80%)
  - nearest earlier step_index in ep    -> that target       (~1.5%)
  - orphan (si > 0, no priors)          -> <none>            (~0.3%)

Mechanism rationale: TRAINING_LOG.md §43-46 (Run L plan). The current
single-screenshot input cannot disambiguate `wait` (no async-state cue) or
direction-sensitive `scroll` (no "where I came from" anchor). Prior-action
history is the standard fix in the mobile-UI-agent literature (ShowUI,
UI-TARS, Auto-UI).

Usage:
    # Dry-run: convert 5 train rows incl. step-0 + multi-step, NO files written.
    uv run python scripts/add_prior_action.py --dry-run

    # Full conversion -> data/androidcontrol_a11y_native_v3/
    uv run python scripts/add_prior_action.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Anchor substring marking where the schema block begins. The "Previous
# action:" line is inserted directly before this anchor (with surrounding
# blank lines). Hard-coded so we fail loudly if the v2 prep ever drifts.
SCHEMA_ANCHOR = "Below is a list of UI elements visible on screen, numbered."

PRIOR_NONE = "<none>"


def build_episode_index(rows: list[dict]) -> dict[int, dict[int, str]]:
    """Returns {episode_id: {step_index: assistant_target_json_text}}."""
    idx: dict[int, dict[int, str]] = defaultdict(dict)
    for r in rows:
        eid = r["episode_id"]; si = r["step_index"]
        target = r["messages"][1]["content"][0]["text"]
        idx[eid][si] = target
    return idx


def lookup_prior(idx: dict[int, dict[int, str]], episode_id: int,
                 step_index: int) -> str:
    """Return the JSON text of the prior step's action, or PRIOR_NONE."""
    if step_index == 0:
        return PRIOR_NONE
    ep = idx.get(episode_id) or {}
    if (step_index - 1) in ep:
        return ep[step_index - 1]
    # Fallback: nearest earlier step in the episode.
    earlier = [s for s in ep.keys() if s < step_index]
    if earlier:
        return ep[max(earlier)]
    return PRIOR_NONE


def transform_row(row: dict, prior_text: str) -> dict:
    user_text = row["messages"][0]["content"][1]["text"]
    if SCHEMA_ANCHOR not in user_text:
        raise ValueError(
            f"row {row.get('episode_id')}/{row.get('step_index')}: "
            f"schema anchor missing in user prompt — v2 prep may have drifted"
        )
    # Insert "Previous action: ..." with blank-line separators directly
    # before the schema block. The original v2 user prompt is
    #   "Task: ...\n\nBelow is a list of UI elements...\n..."
    # so we replace "\n\nBelow is..." with "\n\nPrevious action: X\n\nBelow is...".
    insertion = f"Previous action: {prior_text}\n\n{SCHEMA_ANCHOR}"
    new_user_text = user_text.replace(f"\n\n{SCHEMA_ANCHOR}", f"\n\n{insertion}",
                                       1)
    if new_user_text == user_text:
        # Anchor wasn't preceded by "\n\n" — fall back to single-newline insertion.
        new_user_text = user_text.replace(f"\n{SCHEMA_ANCHOR}",
                                           f"\n{insertion}", 1)
    if new_user_text == user_text:
        raise ValueError(
            f"row {row.get('episode_id')}/{row.get('step_index')}: "
            f"could not insert prior action — anchor pattern mismatch"
        )
    row["messages"][0]["content"][1]["text"] = new_user_text
    return row


def transform_split(src: Path, out: Path, *, dry_run: bool = False,
                     dry_n: int = 5) -> tuple[int, Counter]:
    rows: list[dict] = []
    with open(src) as f:
        for line in f:
            rows.append(json.loads(line))
    idx = build_episode_index(rows)

    counts: Counter[str] = Counter()
    n = 0
    out_lines: list[str] = []

    # For dry-run: show 1 step-0 row, 1 immediate-prior row, 1 fallback row,
    # and a couple of typical rows. Find candidates first.
    samples: list[dict] = []
    if dry_run:
        seen_step0 = False
        seen_immediate = False
        for r in rows:
            si = r["step_index"]
            if si == 0 and not seen_step0:
                samples.append(r); seen_step0 = True
            elif si > 0 and (si - 1) in idx[r["episode_id"]] and not seen_immediate:
                samples.append(r); seen_immediate = True
            if len(samples) >= dry_n:
                break

    for r in rows:
        prior = lookup_prior(idx, r["episode_id"], r["step_index"])
        if prior == PRIOR_NONE and r["step_index"] == 0:
            counts["step0"] += 1
        elif prior == PRIOR_NONE:
            counts["orphan_or_gap_to_none"] += 1
        elif (r["step_index"] - 1) in idx[r["episode_id"]]:
            counts["immediate"] += 1
        else:
            counts["nearest_earlier"] += 1
        new_row = transform_row(dict(r), prior)
        if dry_run and r in samples:
            print(f"=== dry-run sample (ep={r['episode_id']} step={r['step_index']}) ===")
            print(f"PRIOR: {prior}")
            print("USER PROMPT (first 600 chars):")
            print(new_row['messages'][0]['content'][1]['text'][:600])
            print("...\n")
        if not dry_run:
            out_lines.append(json.dumps(new_row))
        n += 1

    if not dry_run:
        with open(out, "w") as f:
            for ln in out_lines:
                f.write(ln + "\n")
    return n, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=Path,
                    default=Path("data/androidcontrol_a11y_native_v2"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/androidcontrol_a11y_native_v3"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--dry-run", action="store_true",
                    help="Show 5 sample rows from train (incl. step-0); no files written.")
    args = ap.parse_args()

    if args.dry_run:
        src = args.src_dir / "train.jsonl"
        n, counts = transform_split(src, Path("/dev/null"),
                                     dry_run=True)
        print(f"\n[dry-run] would have transformed {n} rows. Distribution:")
        for k, v in counts.most_common():
            print(f"  {k:>30}: {v:>6} ({100*v/n:.1f}%)")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Symlink images dir.
    src_imgs = args.src_dir / "images"
    out_imgs = args.out_dir / "images"
    if src_imgs.exists() and not out_imgs.exists():
        # src may itself be a symlink; resolve to the real path.
        out_imgs.symlink_to(src_imgs.resolve())
        print(f"[v3-prep] symlinked images: {out_imgs} -> {src_imgs.resolve()}")

    total = 0
    grand: Counter[str] = Counter()
    for split in args.splits:
        src = args.src_dir / f"{split}.jsonl"
        out = args.out_dir / f"{split}.jsonl"
        if not src.exists():
            print(f"[v3-prep] [skip] {src} missing")
            continue
        n, counts = transform_split(src, out, dry_run=False)
        print(f"[v3-prep] {split}: {n} rows, distribution={dict(counts)}")
        total += n
        grand.update(counts)
    print(f"[v3-prep] done: {args.out_dir}, {total} rows total")
    for k, v in grand.most_common():
        print(f"  {k:>30}: {v:>6} ({100*v/max(total,1):.1f}%)")


if __name__ == "__main__":
    sys.exit(main())
