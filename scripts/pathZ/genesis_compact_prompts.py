"""Compact UI elements section in genesis training prompts.

The M3AA11Y harness emits the full a11y tree, often 50-100 elements per
screen. Many have empty `text` fields (pure layout containers). For
training we drop those empty-label rows from the UI elements list,
keeping element indices unchanged so click(N) still references the
right element. Cuts prompt length ~50%.

Reads `data/pathZ/genesis/train.jsonl`, writes new file with compact
prompts. Same row schema otherwise.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


_UI_LINE_RE = re.compile(r'^\s*UI element (\d+):\s*(\{.*\})\s*$')


def compact_prompt(prompt: str) -> str:
    """Rewrite verbose M3A UI element lines to {"index": N, "text": "..."}.

    Drops the full attribute JSON dump (is_clickable, is_long_clickable,
    etc.) and keeps only index + best label (text or content_description).
    Drops elements with no label entirely.
    """
    out_lines = []
    for line in prompt.splitlines():
        m = _UI_LINE_RE.match(line)
        if not m:
            out_lines.append(line)
            continue
        idx = int(m.group(1))
        try:
            # M3A's renderer emits Python literals (False/True/None) not JSON
            obj = ast.literal_eval(m.group(2))
        except Exception:
            out_lines.append(line)
            continue
        label = (obj.get("text") or obj.get("content_description") or "").strip()
        if not label:
            continue
        # Truncate excessively long labels.
        if len(label) > 80:
            label = label[:77] + "..."
        out_lines.append(f'  UI element {idx}: {{"index": {idx}, "text": "{label}"}}')
    return "\n".join(out_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/pathZ/genesis/train.jsonl")
    ap.add_argument("--out", default="data/pathZ/genesis/train_compact.jsonl")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    n_dropped_lines = 0
    total_in = 0
    total_out = 0
    with open(src) as fin, open(out, "w") as fout:
        for line in fin:
            r = json.loads(line)
            u_blocks = r["messages"][0]["content"]
            for b in u_blocks:
                if b.get("type") == "text":
                    orig = b["text"]
                    new = compact_prompt(orig)
                    n_dropped_lines += orig.count("\n") - new.count("\n")
                    total_in += len(orig)
                    total_out += len(new)
                    b["text"] = new
            fout.write(json.dumps(r) + "\n")
            n += 1
    print(f"[compact] rows: {n}")
    print(f"[compact] dropped UI lines: {n_dropped_lines}")
    print(f"[compact] total chars: {total_in} → {total_out} "
          f"({100 * total_out / max(1, total_in):.1f}%)")


if __name__ == "__main__":
    main()
