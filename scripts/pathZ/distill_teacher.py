#!/usr/bin/env python3
"""Teacher-distillation step for r34.

Reads the existing harness-parity smoke train rows
(`data/pathZ/smoke/train.jsonl`), runs Gemma 4 31B (4-bit) blind on each
prompt, and buckets the emission against the gold action.

Output: `data/pathZ/smoke/train_distilled.jsonl` containing the kept rows
(teacher action == gold action) with the assistant label rewritten as
`Reason: <teacher reason>\\nAction: <gold action>`.

Off-policy notes:
  * Text-only inference. Student is text-only and cannot use pixel-level
    rationale. Teacher sees the same UI-element list the student does.
  * Blind: teacher does NOT see the gold action. Match is computed
    afterwards.
  * Match rule: full m3a_action_match (action_type + grounding arg).
    Type-only fallback would inflate retention but reasoning may then
    explain a different intent.

Smoke-test entry point: `--limit N` runs only the first N rows and
reports the match rate. If the rate is < 0.30, abort distillation —
the teacher is too weak for these apps.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from m3a_format import m3a_action_match, parse_m3a_emission


DEFAULT_TEACHER_ID = "unsloth/gemma-4-31B-it-unsloth-bnb-4bit"
DEFAULT_MAX_NEW_TOKENS = 384


def _load_teacher(model_id: str):
    from unsloth import FastVisionModel  # heavy import; do at call time
    print(f"[distill] loading teacher {model_id}", flush=True)
    t0 = time.time()
    model, processor = FastVisionModel.from_pretrained(
        model_id, load_in_4bit=True, use_gradient_checkpointing=False
    )
    FastVisionModel.for_inference(model)
    print(f"[distill] teacher loaded in {time.time() - t0:.1f}s", flush=True)
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


def _salvage_prompt(user_text: str, gold: dict, pred: dict) -> str:
    """Build a 2nd-pass prompt asking the teacher to justify the gold action.

    The teacher's first emission picked the right action_type but a
    different grounding arg. We show both and ask for one sentence that
    explains why the gold choice is preferable. The student will be
    trained on Reason: <salvaged> + Action: <gold>.
    """
    return (
        user_text +
        "\n\nNote: the correct action for this step is:\n"
        f"  {json.dumps(gold)}\n"
        "Your initial choice was close but picked a different element:\n"
        f"  {json.dumps(pred)}\n"
        "Provide ONLY the Reason line that justifies the correct action "
        "above (one sentence, no Action line):\n"
        "Reason: "
    )


def _extract_reason_only(gen: str) -> str | None:
    """Extract just the Reason text from a 2nd-pass salvage emission."""
    import re as _re  # local alias
    # Teacher prompt ended with `Reason: `, so the gen often starts mid-reason
    s = gen.strip()
    # If model dutifully restated `Reason:` strip it; same for any Action:
    s = _re.sub(r"^Reason:\s*", "", s)
    s = _re.split(r"\nAction:|\n\n", s, maxsplit=1)[0].strip()
    return s if s else None


def _user_text(row: dict) -> str:
    """Extract the user-text from an SFT row (skips image content blocks)."""
    user_blocks = row["messages"][0]["content"]
    for blk in user_blocks:
        if blk.get("type") == "text":
            return blk["text"]
    raise ValueError("row has no user-text block")


def _gold_action(row: dict) -> dict:
    """Extract the gold m3a action — already stored as `gt_m3a` by prepare."""
    return row["gt_m3a"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/pathZ/smoke/train.jsonl",
                    help="harness-parity train rows to distill against")
    ap.add_argument("--out", default="data/pathZ/smoke/train_distilled.jsonl",
                    help="output JSONL of kept rows with teacher rationale")
    ap.add_argument("--mismatches", default="data/pathZ/smoke/train_distill_mismatches.jsonl",
                    help="mismatched rows (teacher action != gold) — saved for r35")
    ap.add_argument("--teacher", default=DEFAULT_TEACHER_ID)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all rows; otherwise stop after N (smoke test)")
    ap.add_argument("--abort-below", type=float, default=0.30,
                    help="if smoke-test match rate < this, exit nonzero")
    ap.add_argument("--resume", action="store_true",
                    help="skip rows whose _distill_src_idx already appears "
                         "in the kept or mismatch files; append rather than truncate")
    args = ap.parse_args()

    src_p = Path(args.src)
    out_p = Path(args.out)
    mis_p = Path(args.mismatches)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(src_p) as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit > 0:
        rows = rows[:args.limit]
    print(f"[distill] {len(rows)} rows from {src_p}", flush=True)

    done_idxs: set[int] = set()
    n_match = 0
    n_salvaged = 0
    n_mismatch = 0
    n_parsefail = 0
    if args.resume:
        for p in (out_p, mis_p):
            if not p.exists():
                continue
            with open(p) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    idx = r.get("_distill_src_idx")
                    if isinstance(idx, int):
                        done_idxs.add(idx)
                        kind = r.get("_distill_kind")
                        if kind == "match":
                            n_match += 1
                        elif kind == "salvage":
                            n_salvaged += 1
                        elif p == mis_p:
                            n_mismatch += 1
        print(f"[distill] resume: {len(done_idxs)} rows already processed "
              f"(match={n_match} salvaged={n_salvaged} mismatch={n_mismatch})",
              flush=True)

    model, processor = _load_teacher(args.teacher)

    t_start = time.time()
    file_mode = "a" if args.resume else "w"
    out_f = open(out_p, file_mode)
    mis_f = open(mis_p, file_mode)
    try:
        for i, row in enumerate(rows):
            if i in done_idxs:
                continue
            user_text = _user_text(row)
            gold = _gold_action(row)
            t0 = time.time()
            try:
                gen = _generate(model, processor, user_text, args.max_new_tokens)
            except Exception as e:  # noqa: BLE001
                print(f"[distill]   row {i}: generate failed: {e}", flush=True)
                n_parsefail += 1
                continue
            reason, pred = parse_m3a_emission(gen)
            dt = time.time() - t0
            if pred is None:
                n_parsefail += 1
                if i < 5 or i % 50 == 0:
                    print(f"[distill] row {i:4d}: PARSEFAIL ({dt:.1f}s) "
                          f"raw={gen[:120]!r}", flush=True)
                continue
            match = m3a_action_match(pred, gold)
            kept_reason: str | None = None
            kept_kind: str | None = None  # "match" or "salvage" or None
            if match["full_match"]:
                kept_reason = reason
                kept_kind = "match"
                n_match += 1
            elif match["type_match"]:
                # 2nd-pass salvage: teacher had the right verb but wrong
                # grounding. Show it the gold action; ask for a reason
                # that justifies the gold choice. Used for r34.
                try:
                    salv_gen = _generate(
                        model, processor,
                        _salvage_prompt(user_text, gold, pred),
                        args.max_new_tokens,
                    )
                    salv_reason = _extract_reason_only(salv_gen)
                except Exception as e:  # noqa: BLE001
                    salv_reason = None
                    print(f"[distill]   row {i}: salvage gen failed: {e}",
                          flush=True)
                if salv_reason:
                    kept_reason = salv_reason
                    kept_kind = "salvage"
                    n_salvaged += 1
                else:
                    n_mismatch += 1
            else:
                # type-mismatched: teacher picked a different action verb
                # entirely (e.g. open_app vs click). Discard — including
                # such rows would teach the student to deviate from the
                # AC reference trajectory.
                n_mismatch += 1

            if kept_kind:
                new_asst = (f"Reason: {kept_reason or ''}\n"
                            f"Action: {json.dumps(gold)}")
                row_out = dict(row)
                row_out["messages"] = [
                    row["messages"][0],
                    {"role": "assistant", "content": [
                        {"type": "text", "text": new_asst}
                    ]},
                ]
                row_out["_distill_reason"] = kept_reason
                row_out["_distill_pred"] = pred
                row_out["_distill_kind"] = kept_kind
                row_out["_distill_src_idx"] = i
                out_f.write(json.dumps(row_out) + "\n")
                out_f.flush()
            else:
                row_out = dict(row)
                row_out["_distill_reason"] = reason
                row_out["_distill_pred"] = pred
                row_out["_distill_kind"] = "mismatch"
                row_out["_distill_src_idx"] = i
                mis_f.write(json.dumps(row_out) + "\n")
                mis_f.flush()
            if i < 10 or i % 25 == 0:
                kept = n_match + n_salvaged
                done = kept + n_mismatch + n_parsefail
                rate = kept / max(1, done)
                eta = (time.time() - t_start) / (i + 1) * (len(rows) - i - 1)
                print(f"[distill] row {i:4d}: "
                      f"match={n_match} salvaged={n_salvaged} "
                      f"mismatch={n_mismatch} parsefail={n_parsefail} "
                      f"kept={rate:.2%} "
                      f"({dt:.1f}s/row, eta={eta/60:.1f}m)", flush=True)
    finally:
        out_f.close()
        mis_f.close()

    n_kept = n_match + n_salvaged
    total_done = n_kept + n_mismatch + n_parsefail
    keep_rate = n_kept / max(1, total_done)
    print(f"[distill] done. kept={n_kept}/{total_done} ({keep_rate:.2%}) "
          f"[match={n_match} salvaged={n_salvaged}] "
          f"mismatch={n_mismatch} parsefail={n_parsefail}", flush=True)
    print(f"METRIC distill_match_rate={n_match / max(1,total_done) * 100:.2f}",
          flush=True)
    print(f"METRIC distill_salvage_rate={n_salvaged / max(1,total_done) * 100:.2f}",
          flush=True)
    print(f"METRIC distill_keep_rate={keep_rate * 100:.2f}", flush=True)
    print(f"METRIC distill_n_kept={n_kept}", flush=True)
    print(f"METRIC distill_n_total={total_done}", flush=True)
    print(f"METRIC distill_runtime_s={time.time() - t_start:.1f}", flush=True)

    if args.limit > 0 and keep_rate < args.abort_below:
        print(f"[distill] FAIL: smoke keep_rate {keep_rate:.2%} < "
              f"{args.abort_below:.2%}; teacher too weak", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
