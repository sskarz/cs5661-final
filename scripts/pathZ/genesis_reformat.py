"""Reformat genesis training prompts to match smoke v9 (m3a_format) style.

Genesis prompts came from M3AA11Y's runtime renderer with verbose preamble
(Path W's full instructions, not the trimmed M3A_PROMPT_PREFIX). Training
on those NaNs (verified: 50-row tiny test, swap-isolation test). Same
trajectories rebuilt with smoke-v9-style prompts using
m3a_format.M3A_PROMPT_PREFIX + render_inventory_block.

Reads `data/pathZ/genesis/train.jsonl` (or train_compact), writes new file.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "pathZ"))
from m3a_format import M3A_PROMPT_PREFIX, render_inventory_block  # noqa: E402

INV_BLOCK = render_inventory_block()


def reformat(prompt: str) -> str:
    # Goal
    m = re.search(r'The current user goal/request is:\s*(.+)', prompt)
    goal = m.group(1).strip() if m else ""
    # History
    m = re.search(
        r'Here is a history of what you have done so far:\s*\n(.+?)'
        r'(?=\n\nThe current screenshot|\nHere is a list|\Z)',
        prompt, re.DOTALL,
    )
    history = (m.group(1).strip() if m
               else "You just started, no action has been performed yet.")
    # UI elements (compact form: {"index": N, "text": "..."})
    ui_lines = [
        line.strip() for line in prompt.split('\n')
        if re.match(r'^\s*UI element \d+:\s*\{"index":', line)
    ]
    parts = [
        INV_BLOCK,
        '',
        '',
        M3A_PROMPT_PREFIX,
        f'\nThe current user goal/request is: {goal}\n',
        f'Here is a history of what you have done so far:\n{history}\n',
        'Here is the list of UI elements visible on screen '
        '(numeric indexes match the labeled screenshot):',
    ] + ['  ' + l for l in ui_lines] + [
        '',
        'Now output an action from the above list.',
        'Reason: ...',
        'Action: {"action_type":...}',
        '',
        'Your Answer:',
        '',
    ]
    return '\n'.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/pathZ/genesis/train_compact.jsonl")
    ap.add_argument("--out", default="data/pathZ/genesis/train_v3.jsonl")
    args = ap.parse_args()
    n = 0
    total_in = total_out = 0
    with open(args.src) as fin, open(args.out, "w") as fout:
        for line in fin:
            r = json.loads(line)
            for b in r["messages"][0]["content"]:
                if b.get("type") == "text":
                    orig = b["text"]
                    total_in += len(orig)
                    b["text"] = reformat(orig)
                    total_out += len(b["text"])
            fout.write(json.dumps(r) + "\n")
            n += 1
    print(f"[reformat] rows: {n}")
    print(f"[reformat] chars: {total_in} → {total_out} "
          f"({100 * total_out / max(1, total_in):.1f}%)")


if __name__ == "__main__":
    main()
