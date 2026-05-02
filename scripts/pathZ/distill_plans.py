#!/usr/bin/env python3
"""Teacher-generated multi-step plans for r41.

For each unique AndroidLab goal, query Gemma 4 31B 4-bit (already loaded
via Unsloth FastVisionModel) for a 3-5 step plan. The plan is grounded
on the AW 19-app inventory; the teacher does NOT see the actual UI
elements or trajectory actions.

Output: data/pathZ/al_plans.jsonl mapping goal → plan.

Why this is different from r34 distillation:
  * r34 distilled per-action `Reason:` text on AC trajectories. The
    teacher saw the same impoverished context as the student, so it
    had no edge.
  * r41 distills task-level PLANS (not per-step reasons). The teacher
    only needs to see the goal + AW inventory; it doesn't need to
    ground in specific UI elements. Plans are about goal decomposition,
    not action grounding.
  * Output is consumed at training time as a static PLAN block in the
    prompt header (set once at step 0, carried through history).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from m3a_format import HARNESS_APP_INVENTORY


DEFAULT_TEACHER_ID = "unsloth/gemma-4-31B-it-unsloth-bnb-4bit"
DEFAULT_MAX_NEW_TOKENS = 256


def _load_teacher(model_id: str):
    from unsloth import FastVisionModel
    print(f"[plans] loading teacher {model_id}", flush=True)
    t0 = time.time()
    model, processor = FastVisionModel.from_pretrained(
        model_id, load_in_4bit=True, use_gradient_checkpointing=False
    )
    FastVisionModel.for_inference(model)
    print(f"[plans] teacher loaded in {time.time() - t0:.1f}s", flush=True)
    return model, processor


def _generate(model, processor, user_text: str, max_new_tokens: int) -> str:
    msgs = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    chat = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = processor(text=chat, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    gen = processor.decode(
        out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )
    return gen


def _build_prompt(goal: str) -> str:
    inventory = "\n".join(f"  - {name}" for name, _ in HARNESS_APP_INVENTORY)
    return (
        "You are planning steps for an Android agent that operates apps via a "
        "limited action vocabulary: open_app, click(element), input_text, scroll, "
        "navigate_back, navigate_home, wait, status(complete|infeasible), answer.\n\n"
        f"Available apps on the device:\n{inventory}\n\n"
        "Given a user goal, write a SHORT plan of 3-6 numbered steps that lead "
        "to completion. Each step should describe ONE concrete action (e.g. "
        '"Open the Markor app", "Tap the file named X", "Type the text", '
        '"Confirm by clicking Save"). End with a final step that emits '
        '`status: complete`.\n\n'
        "Rules:\n"
        "  - Use only apps from the list above. If the goal mentions an app NOT "
        "in the list, write `Plan: infeasible — app not installed.` and stop.\n"
        "  - Be concrete and concise. No flowery prose.\n"
        "  - Do NOT add a Reason or Action line; just the numbered plan.\n\n"
        f"User goal: {goal}\n\n"
        "Plan:\n"
    )


def _extract_unique_goals(al_jsonl: Path) -> list[str]:
    goals: dict[str, None] = {}  # ordered uniq
    marker = "The current user goal/request is: "
    with open(al_jsonl) as f:
        for line in f:
            r = json.loads(line)
            prompt = r["messages"][0]["content"][1]["text"]
            idx = prompt.find(marker)
            if idx < 0:
                continue
            start = idx + len(marker)
            end = prompt.find("\n", start)
            g = prompt[start:end].strip()
            if g and g not in goals:
                goals[g] = None
    return list(goals.keys())


def _clean_plan(raw: str) -> str:
    """Strip teacher boilerplate, keep just the numbered list."""
    s = raw.strip()
    # Truncate at obvious end-of-plan markers
    for marker in ("\nReason:", "\nAction:", "\n\n"):
        i = s.find(marker)
        if i >= 0:
            s = s[:i]
    return s.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/pathZ/raw/androidlab_smoke.jsonl",
                    help="AndroidLab smoke jsonl with goals to distill plans for")
    ap.add_argument("--out", default="data/pathZ/al_plans.jsonl",
                    help="output JSONL: {goal, plan, raw}")
    ap.add_argument("--teacher", default=DEFAULT_TEACHER_ID)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all unique goals; otherwise stop after N (smoke)")
    ap.add_argument("--resume", action="store_true",
                    help="skip goals already in the output file")
    args = ap.parse_args()

    src_p = Path(args.src)
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    goals = _extract_unique_goals(src_p)
    print(f"[plans] {len(goals)} unique goals from {src_p}", flush=True)
    if args.limit > 0:
        goals = goals[:args.limit]

    done: set[str] = set()
    if args.resume and out_p.exists():
        with open(out_p) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["goal"])
                except Exception:
                    continue
        print(f"[plans] resume: {len(done)} goals already distilled", flush=True)

    pending = [g for g in goals if g not in done]
    print(f"[plans] {len(pending)} goals remain to distill", flush=True)
    if not pending:
        print("[plans] nothing to do; exit", flush=True)
        return

    model, processor = _load_teacher(args.teacher)

    n_ok = 0
    n_infeasible = 0
    n_fail = 0
    t_start = time.time()
    file_mode = "a" if args.resume else "w"
    out_f = open(out_p, file_mode)
    try:
        for i, goal in enumerate(pending):
            t0 = time.time()
            try:
                raw = _generate(model, processor, _build_prompt(goal),
                                args.max_new_tokens)
            except Exception as e:  # noqa: BLE001
                print(f"[plans] goal {i}: gen failed: {e}", flush=True)
                n_fail += 1
                continue
            plan = _clean_plan(raw)
            is_infeasible = "infeasible" in plan.lower()
            if is_infeasible:
                n_infeasible += 1
            else:
                n_ok += 1
            row = {"goal": goal, "plan": plan, "raw": raw,
                   "infeasible": is_infeasible}
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()

            dt = time.time() - t0
            if i < 5 or i % 25 == 0:
                done_n = n_ok + n_infeasible + n_fail
                eta = (time.time() - t_start) / done_n * (len(pending) - done_n)
                print(f"[plans] goal {i:4d}/{len(pending)}: ok={n_ok} "
                      f"infeasible={n_infeasible} fail={n_fail} "
                      f"({dt:.1f}s/goal, eta={eta/60:.1f}m)", flush=True)
    finally:
        out_f.close()

    total = n_ok + n_infeasible + n_fail
    print(f"[plans] done. ok={n_ok}/{total} infeasible={n_infeasible} fail={n_fail}",
          flush=True)
    print(f"METRIC plans_ok_rate={n_ok / max(1, total) * 100:.2f}", flush=True)
    print(f"METRIC plans_infeasible_rate={n_infeasible / max(1, total) * 100:.2f}",
          flush=True)
    print(f"METRIC plans_n_total={total}", flush=True)
    print(f"METRIC plans_runtime_s={time.time() - t_start:.1f}", flush=True)


if __name__ == "__main__":
    main()
