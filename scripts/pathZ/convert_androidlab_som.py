#!/usr/bin/env python3
"""Convert THUDM/Android-Lab SoM-mode training data into our M3A-format
smoke schema, mirroring `prepare_smoke_data.py`.

Source: https://github.com/THUDM/Android-Lab + Google Drive zip — extracted
to `data/pathZ/raw/androidlab_extracted/android-lab-train/`. The XML mode
of this dataset is text-only (Llama-3.1-8B trained with no images); the
SoM mode (this script) is multimodal and matches what M3A passes at AW
eval time (one labeled screenshot + textual goal/history).

Each AndroidLab row = ONE step decision in a trajectory, with the entire
trajectory history rendered as text in `messages[0].content` and the
current step's image in `images[0]`. We map:

  AndroidLab action → M3A action
  -----------------------------
  tap(N)                    → click(index=N)
  type("X")                 → input_text(text=X, index=last_tap_index)
  swipe(N, "DIR", "LEN")    → scroll(direction=DIR, index=N)  # length dropped
  back()                    → navigate_back
  finish("...msg...")       → status(goal_status="complete")

We render the M3A user prompt with goal + history (AndroidLab's prior
rounds) and rely on the labeled screenshot for UI grounding (no XML/a11y
tree element list, since AndroidLab SoM data does NOT carry one — the
labels are baked into the image).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Local helpers (same dir)
sys.path.insert(0, str(Path(__file__).parent))
from m3a_format import M3A_PROMPT_PREFIX


# Goal is on the line immediately after `Round 0\n\n<|user|>\n`
_GOAL_RE = re.compile(
    r"Round 0\s*\n\s*<\|user\|>\n([^\n]+)", flags=re.DOTALL,
)
# A round block: from "Round N" to next "Round N+1" or end. We extract the
# action that the previous-round assistant emitted to build history.
_ROUND_BLOCK_RE = re.compile(
    r"Round (\d+).*?<\|assistant\|>\s*\n([^\n]*)", flags=re.DOTALL,
)


def _parse_androidlab_action(out: str, last_tap_index: int | None) -> dict | None:
    """Convert one AndroidLab output string to M3A action dict."""
    out = out.strip()
    m = re.match(r"tap\((\d+)\)", out)
    if m:
        return {"action_type": "click", "index": int(m.group(1))}
    m = re.match(r'type\("(.*)"\)', out)
    if m:
        text = m.group(1)
        # AndroidLab `type` doesn't carry an index; bind to the most recent tap.
        # Fall back to 0 if no prior tap exists in this trajectory.
        return {
            "action_type": "input_text",
            "text": text,
            "index": last_tap_index if last_tap_index is not None else 0,
        }
    m = re.match(r'swipe\((\d+),\s*"(\w+)"', out)
    if m:
        idx, direction = int(m.group(1)), m.group(2)
        return {"action_type": "scroll", "direction": direction, "index": idx}
    if out.startswith("back("):
        return {"action_type": "navigate_back"}
    if out.startswith("home("):
        return {"action_type": "navigate_home"}
    if out.startswith("finish("):
        # AndroidLab uses `finish` as terminal action regardless of success.
        # M3A's status w/ goal_status="complete" is the closest match.
        return {"action_type": "status", "goal_status": "complete"}
    return None


def _synthesize_reason(m3a_action: dict) -> str:
    """One-liner reason matching prepare_smoke_data.py's style."""
    at = m3a_action.get("action_type", "")
    if at == "click":
        return f'Click element {m3a_action["index"]}.'
    if at == "input_text":
        return f'Type "{m3a_action.get("text", "")}" into element ' \
               f'{m3a_action.get("index", "?")}.'
    if at == "scroll":
        return f'Scroll {m3a_action.get("direction", "down")} to reveal more.'
    if at == "navigate_back":
        return "Navigate back to the previous screen."
    if at == "navigate_home":
        return "Navigate to the home screen."
    if at == "status":
        return "Mark the task complete."
    return "Perform the chosen action."


def _convert_row(src_row: dict, src_dir: Path) -> dict | None:
    """Convert one SoM row to our smoke schema."""
    user_text = src_row["messages"][0]["content"]
    output = src_row["messages"][1]["content"]

    # Goal
    g = _GOAL_RE.search(user_text)
    if not g:
        return None
    goal = g.group(1).strip()

    # Walk all rounds — the last round's action is the LABEL we predict; the
    # earlier rounds' actions form the history. AndroidLab embeds prior
    # actions inside the user_text (after each "Round N\n...<|assistant|>\n"
    # marker). The TERMINAL <|assistant|>\n is empty (model is supposed to
    # emit the action there).
    blocks = _ROUND_BLOCK_RE.findall(user_text)
    # blocks: list of (round_number, action_string) for ALL prior rounds.
    # The terminal round's action is empty (just whitespace) — strip it.
    history_actions: list[dict] = []
    last_tap_index: int | None = None
    for r_idx, action_str in blocks:
        if not action_str.strip():
            continue  # current round, no action yet
        a = _parse_androidlab_action(action_str.strip(), last_tap_index)
        if a is None:
            continue
        history_actions.append(a)
        if a.get("action_type") == "click":
            last_tap_index = a.get("index")

    # Now the LABEL is the gold output for the current round.
    gt = _parse_androidlab_action(output, last_tap_index)
    if gt is None:
        return None

    # Build the M3A user prompt: prefix + goal + history block.
    # No UI element list — labels are in the screenshot.
    if history_actions:
        history_text = "\n".join(
            f"Step {i}: {json.dumps(a)}" for i, a in enumerate(history_actions[-3:])
        )
    else:
        history_text = ""
    history_block = (history_text or
                     "You just started, no action has been performed yet.")
    user_text_m3a = (
        M3A_PROMPT_PREFIX
        + f"\nThe current user goal/request is: {goal}\n"
        + f"\nHistory of actions so far:\n{history_block}\n"
        + "\nThe labeled screenshot shows the current screen with"
          " numeric indexes on bounding boxes around UI elements.\n"
        + "\nNow output an action from the above list.\n"
        + 'Reason: ...\nAction: {"action_type":...}\n\nYour Answer:\n'
    )

    # Image: relative path from src_dir
    img_rel = src_row["images"][0]
    if not (src_dir / img_rel).exists():
        # ~ rare; skip rows whose image is missing
        return None

    reason = _synthesize_reason(gt)
    asst_text = f"Reason: {reason}\nAction: {json.dumps(gt)}"

    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": user_text_m3a},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": asst_text},
            ]},
        ],
        "image": img_rel,                 # relative to src_dir
        "_image_root": str(src_dir),      # stash absolute root for loaders
        "elements": [],                   # AndroidLab SoM has no element list
        "gt_m3a": gt,
        "history_len": len(history_actions),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-json", type=Path,
                    default=Path("data/pathZ/raw/androidlab_extracted/"
                                 "android-lab-train/androidlab-som-train.json"))
    ap.add_argument("--src-dir", type=Path,
                    default=Path("data/pathZ/raw/androidlab_extracted/"
                                 "android-lab-train"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/pathZ/raw/androidlab_smoke.jsonl"))
    args = ap.parse_args()

    src = json.load(open(args.src_json))
    print(f"[al-conv] {len(src)} src rows")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    from collections import Counter
    at_counter: Counter[str] = Counter()
    with open(args.out, "w") as f:
        for row in src:
            conv = _convert_row(row, args.src_dir)
            if conv is None:
                n_skipped += 1
                continue
            f.write(json.dumps(conv) + "\n")
            n_written += 1
            at_counter[conv["gt_m3a"]["action_type"]] += 1

    print(f"[al-conv] wrote {n_written} rows ({n_skipped} skipped) → {args.out}")
    print(f"[al-conv] action_type dist: {dict(at_counter.most_common())}")


if __name__ == "__main__":
    main()
