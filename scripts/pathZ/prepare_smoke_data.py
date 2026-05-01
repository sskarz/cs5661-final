#!/usr/bin/env python3
"""Build the cached M3A-format smoke train/eval JSONL files.

Reads AndroidControl-v3 (Path W schema), reformats every row into M3A's
exact prompt + action vocabulary, writes:
  data/pathZ/smoke/train.jsonl  (default 2000 rows)
  data/pathZ/smoke/eval.jsonl   (default 200 rows)

The conversion is deterministic given --seed. We sort by (episode, step)
before sampling so the smoke distribution covers many episodes rather than
one.

Action conversion: tap→click(index), type→input_text(index,text),
scroll→scroll(direction), open_app→open_app(app_name), navigate_*/wait
passthrough. The reason field is a short synthetic string built from the
action so the model has SOMETHING to match for the Reason: prefix at
training time (otherwise it never learns to emit a reason line).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Local helper module
from m3a_format import pathw_to_m3a, render_m3a_prompt


def _synthesize_reason(m3a_action: dict, ui_elements: list[dict]) -> str:
    """Deterministic short reason string — never blank, always one line.

    Doesn't try to reason; just describes what the action does so the model
    learns the FORMAT. SFT teaches "always emit Reason: <something>" — the
    quality of <something> is improved later by AndroidLab CoT data.
    """
    at = m3a_action.get("action_type", "")
    label_for = lambda i: next((e.get("label", "?") for e in ui_elements
                                if e.get("id") == i), "?")
    if at == "click":
        return f'Click element {m3a_action["index"]} ("{label_for(m3a_action["index"])}").'
    if at == "long_press":
        return f'Long-press element {m3a_action["index"]}.'
    if at == "input_text":
        return f'Type text into element {m3a_action.get("index", "?")}.'
    if at == "scroll":
        return f'Scroll {m3a_action.get("direction", "down")} to reveal more content.'
    if at == "open_app":
        return f'Open the {m3a_action.get("app_name", "")} app.'
    if at == "wait":
        return "Wait for the screen to update."
    if at == "navigate_back":
        return "Navigate back to the previous screen."
    if at == "navigate_home":
        return "Navigate to the home screen."
    if at == "keyboard_enter":
        return "Press the Enter key."
    if at == "status":
        return f'Mark the task {m3a_action.get("goal_status", "complete")}.'
    if at == "answer":
        return "Answer the user's question."
    return "Perform the chosen action."


def _build_row(src_row: dict, history_text: str) -> dict | None:
    """Convert one Path-W src row → one M3A-format SFT row."""
    try:
        gt_pathw = json.loads(src_row["messages"][1]["content"][0]["text"])
    except Exception:
        return None
    gt_m3a = pathw_to_m3a(gt_pathw)
    if gt_m3a is None:
        return None
    elements = src_row.get("elements") or []
    goal = src_row.get("goal", "")
    user_text = render_m3a_prompt(goal=goal, history=history_text,
                                  ui_elements=elements)
    reason = _synthesize_reason(gt_m3a, elements)
    asst_text = f'Reason: {reason}\nAction: {json.dumps(gt_m3a)}'

    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": asst_text},
            ]},
        ],
        "image": src_row["image"],
        "episode_id": src_row.get("episode_id"),
        "step_index": src_row.get("step_index"),
        "elements": elements,
        "gt_m3a": gt_m3a,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/androidcontrol_a11y_native_v3")
    ap.add_argument("--out", default="data/pathZ/smoke")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--balance-classes", action="store_true",
                    help="Resample train per-action-type to a uniform "
                         "distribution. Eval is left untouched.")
    ap.add_argument("--per-class-target", type=int, default=250,
                    help="Target rows per action type when --balance-classes.")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def _load(p: Path) -> list[dict]:
        rows = []
        with open(p) as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    train_src = _load(src / "train.jsonl")
    val_src = _load(src / "val.jsonl")
    print(f"[prep] src train={len(train_src)} val={len(val_src)}")

    rng = random.Random(args.seed)
    rng.shuffle(train_src)
    rng.shuffle(val_src)

    # Build episode-keyed index for history lookup. AC steps within an
    # episode are sequential; for non-step-0 we synthesize one prior-action
    # line so the history block isn't always "no action performed yet".
    train_by_ep: dict[int, list[dict]] = {}
    for r in _load(src / "train.jsonl"):
        train_by_ep.setdefault(r.get("episode_id"), []).append(r)
    val_by_ep: dict[int, list[dict]] = {}
    for r in _load(src / "val.jsonl"):
        val_by_ep.setdefault(r.get("episode_id"), []).append(r)
    for d in (train_by_ep, val_by_ep):
        for ep, rs in d.items():
            rs.sort(key=lambda r: r.get("step_index", 0))

    def _history_for(row: dict, ep_idx: dict) -> str:
        ep = ep_idx.get(row.get("episode_id")) or []
        si = row.get("step_index", 0)
        if si == 0 or not ep:
            return ""
        prior_strs = []
        for prev in ep:
            if prev.get("step_index", 0) >= si:
                break
            try:
                a = pathw_to_m3a(json.loads(
                    prev["messages"][1]["content"][0]["text"]))
                if a is not None:
                    prior_strs.append(
                        f"Step {prev.get('step_index')}: {json.dumps(a)}")
            except Exception:
                pass
        return "\n".join(prior_strs[-3:])  # last 3 only

    out_train = out / "train.jsonl"
    out_eval = out / "eval.jsonl"

    if args.balance_classes:
        # Bucket all source train rows by their post-conversion M3A
        # action_type, then sample per-class up to the target count
        # (with replacement when the source pool is short).
        from collections import defaultdict
        buckets: dict[str, list[dict]] = defaultdict(list)
        for r in train_src:
            row = _build_row(r, _history_for(r, train_by_ep))
            if row is None:
                continue
            buckets[row["gt_m3a"]["action_type"]].append(row)
        # Drop classes whose source pool is much smaller than the target —
        # otherwise we replay the same handful of rows many times, which
        # overfits to that class. Keep replay factor ≤ ~2.5x.
        min_pool = max(5, args.per_class_target // 3)
        skipped = [k for k, v in buckets.items() if len(v) < min_pool]
        for k in skipped:
            print(f"[prep] dropping action_type={k} ({len(buckets[k])} rows)")
            buckets.pop(k)
        rng2 = random.Random(args.seed + 1)
        target = args.per_class_target
        n_train_written = 0
        with open(out_train, "w") as f:
            for at, rows in sorted(buckets.items()):
                # If pool is smaller than target, sample with replacement.
                if len(rows) >= target:
                    samp = rng2.sample(rows, target)
                else:
                    samp = [rng2.choice(rows) for _ in range(target)]
                rng2.shuffle(samp)
                for row in samp:
                    f.write(json.dumps(row) + "\n")
                    n_train_written += 1
                print(f"[prep]   {at:18s} pool={len(rows):4d}  "
                      f"sampled={len(samp):4d}")
    else:
        n_train_written = 0
        with open(out_train, "w") as f:
            for r in train_src:
                if n_train_written >= args.n_train:
                    break
                row = _build_row(r, _history_for(r, train_by_ep))
                if row is None:
                    continue
                f.write(json.dumps(row) + "\n")
                n_train_written += 1

    n_eval_written = 0
    with open(out_eval, "w") as f:
        for r in val_src:
            if n_eval_written >= args.n_eval:
                break
            row = _build_row(r, _history_for(r, val_by_ep))
            if row is None:
                continue
            f.write(json.dumps(row) + "\n")
            n_eval_written += 1

    # Action-type distribution sanity check
    from collections import Counter
    train_at = Counter()
    eval_at = Counter()
    with open(out_train) as f:
        for line in f:
            train_at[json.loads(line)["gt_m3a"]["action_type"]] += 1
    with open(out_eval) as f:
        for line in f:
            eval_at[json.loads(line)["gt_m3a"]["action_type"]] += 1

    print(f"[prep] wrote {n_train_written} train, {n_eval_written} eval")
    print(f"[prep] train action_type dist: {dict(train_at.most_common())}")
    print(f"[prep] eval  action_type dist: {dict(eval_at.most_common())}")


if __name__ == "__main__":
    main()
