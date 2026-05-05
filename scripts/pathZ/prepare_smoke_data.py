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
from m3a_format import (
    HARNESS_APP_INVENTORY,
    harness_action_repr,
    pathw_to_m3a,
    render_m3a_prompt,
)


def _is_aw_app(name: str) -> bool:
    """True if `name` resolves to an app in the AW harness 19-app inventory.

    AC's `open_app` field is freeform and uses display names (e.g. "Chrome",
    "Maps", "Amazon"). AW's harness can only dispatch to its closed-set
    inventory. r34's data analysis showed 96% of AC `open_app` rows target
    apps that DO NOT EXIST in AW; training on those teaches the student to
    call non-existent apps. This filter keeps only AC rows whose target app
    matches the AW inventory (case-insensitive, substring tolerant).
    """
    n = (name or "").strip().lower()
    if not n:
        return False
    for app, _pkg in HARNESS_APP_INVENTORY:
        a = app.lower()
        if n == a or a in n or n in a:
            return True
    return False


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


def _build_row(src_row: dict, history_text: str,
               harness_parity: bool = False,
               compact_a11y: bool = False,
               plan: str | None = None,
               is_step0: bool | None = None) -> dict | None:
    """Convert one Path-W src row → one M3A-format SFT row.

    r41 plan support:
      - `plan`: teacher-generated 3-5 step plan for this row's goal.
      - `is_step0`: True if the row is the first step in its trajectory
        (history empty). Step 0 rows train the model to EMIT the plan
        at the top of its assistant text. Step 1+ rows train the model
        to USE a plan that's been re-injected in the user prompt.
    """
    try:
        gt_pathw = json.loads(src_row["messages"][1]["content"][0]["text"])
    except Exception:
        return None
    gt_m3a = pathw_to_m3a(gt_pathw)
    if gt_m3a is None:
        return None
    elements = src_row.get("elements") or []
    goal = src_row.get("goal", "")
    # If this row gets a plan AND it's not step 0, prepend the plan to
    # the user goal text. This mirrors what the harness will inject at
    # eval time after parsing the model's step-0 emission.
    plan_in_goal = ""
    if plan and is_step0 is False:
        plan_in_goal = f"Plan from step 0:\n{plan}\n\n"
    user_text = render_m3a_prompt(goal=plan_in_goal + goal,
                                  history=history_text,
                                  ui_elements=elements,
                                  harness_parity=harness_parity,
                                  compact_a11y=compact_a11y)
    reason = _synthesize_reason(gt_m3a, elements)
    asst_body = f'Reason: {reason}\nAction: {json.dumps(gt_m3a)}'
    # For step 0 rows with a plan, the assistant emits Plan first.
    # Train the model to ALWAYS emit the plan at step 0.
    if plan and is_step0 is True:
        asst_text = f'Plan:\n{plan}\n\n{asst_body}'
    else:
        asst_text = asst_body

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
    ap.add_argument("--boost-input-text", type=int, default=1,
                    help="Multiply per-class-target by this factor for the "
                         "input_text class only. r29 collapsed input_text "
                         "to 0% AC type-match — boosting it back into the "
                         "training mix is intended to recover that emission.")
    ap.add_argument("--androidlab-jsonl", type=Path, default=None,
                    help="If set, mix AndroidLab SoM-converted rows into "
                         "training (50/50 with AC for classes both have).")
    ap.add_argument("--include-status", action="store_true",
                    help="If set, also include a `status` class in the "
                         "balanced train mix (AndroidLab-only).")
    ap.add_argument("--harness-parity", action="store_true",
                    help="Emit prompts matching the M3AA11Y eval-time prompt "
                         "(indexed app inventory + det-history Step format). "
                         "Required for r33 schema-parity retrain.")
    ap.add_argument("--aw-apps-only", action="store_true",
                    help="Filter AC `open_app` rows to only those whose target "
                         "app exists in the AW harness 19-app inventory. r34 "
                         "data analysis: 96%% of AC open_app rows reference "
                         "apps that do not exist in AW.")
    ap.add_argument("--history-reasons", action="store_true",
                    help="In harness-parity mode, also emit a `  reason: ...` "
                         "line under each prior Step N: in the history block. "
                         "Required to train the model to use reasons that the "
                         "M3AA11Y harness carries through history at eval. "
                         "r36 evidenced that adding history reasons at eval "
                         "alone (without matching train) is net-negative.")
    ap.add_argument("--compact-a11y", action="store_true",
                    help="Drop UI elements with empty labels from the rendered "
                         "prompt. Indices preserved (skip lines, don't "
                         "renumber) so click(N) semantics unchanged. r39: "
                         "tests prompt-budget hypothesis without touching "
                         "history block.")
    ap.add_argument("--plans-jsonl", type=Path, default=None,
                    help="Path to teacher-generated plans (output of "
                         "distill_plans.py). When set, augment AL training "
                         "rows: step 0 emits Plan: header; step 1+ has "
                         "`Plan from step 0:` re-injected in user prompt. "
                         "M3AA11Y harness must mirror this format at eval.")
    ap.add_argument("--plan-align-filter", action="store_true",
                    help="When using plans, drop AL trajectories whose "
                         "teacher-plan step 1 does not align with the "
                         "trajectory's step-0 gold action. Currently checks "
                         "open_app: plan must reference the same app name "
                         "and start with 'Open'. Smaller training set, "
                         "cleaner step-0 supervision.")
    ap.add_argument("--synthesize-open-app", type=int, default=0,
                    help="Generate N synthetic `open_app` training rows per "
                         "AW-inventory app (goal='Open the X app', empty UI, "
                         "empty history, gold=open_app(X)). Adds N*19 rows "
                         "to the open_app bucket so balance-classes has "
                         "enough on-distribution open_app coverage.")
    ap.add_argument("--genesis-jsonl", type=Path, default=None,
                    help="Path to genesis vision training JSONL (output of "
                         "genesis_to_train.py or genesis_vision_train.py). "
                         "When set, genesis rows with screenshots are mixed "
                         "into the balanced training data alongside AC + AL. "
                         "Genesis rows must have `image` + `_image_root` "
                         "fields for vision training.")
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
            except Exception:
                continue
            if a is None:
                continue
            n = prev.get("step_index", 0) + 1
            if args.harness_parity:
                # Mirror M3AA11Y's deterministic per-step summary:
                #   Step N: <action_repr> -> ok; window changed
                # AC trajectories are gold so the executor feedback is `ok`;
                # the delta is a generic placeholder — the goal is to teach
                # the model the FORMAT, not pixel-accurate state diffs.
                prior_strs.append(
                    f"Step {n}: {harness_action_repr(a)} -> ok; window changed"
                )
                # r37: also carry the synthetic reason for the prior action
                # on a `  reason: ...` second line. Trains the model to
                # attend to prior-step reasoning when planning the next
                # action. Eval-time harness must mirror this format
                # (m3a_a11y.M3AA11Y).
                if args.history_reasons:
                    prev_elements = prev.get("elements") or []
                    prior_strs.append(
                        f"  reason: {_synthesize_reason(a, prev_elements)}"
                    )
            else:
                prior_strs.append(f"Step {n}: {json.dumps(a)}")
        return "\n".join(prior_strs[-6:])  # r37: last 3 steps × 2 lines each

    out_train = out / "train.jsonl"
    out_eval = out / "eval.jsonl"

    # Stamp absolute image roots so the trainer/evaluator can locate
    # images from multiple sources without juggling --data-dir flags.
    AC_IMG_ROOT = str(src.resolve())

    if args.balance_classes:
        # Bucket all source train rows by their post-conversion M3A
        # action_type, then sample per-class up to the target count
        # (with replacement when the source pool is short).
        from collections import defaultdict
        buckets: dict[str, list[dict]] = defaultdict(list)
        n_filtered_open_app = 0
        for r in train_src:
            row = _build_row(r, _history_for(r, train_by_ep),
                             harness_parity=args.harness_parity, compact_a11y=args.compact_a11y)
            if row is None:
                continue
            # r34 finding: 96% of AC open_app rows target apps that do not
            # exist in AW (Amazon/Maps/Drive/Gmail/eBay/Vimeo/Edmunds/etc.).
            # When --aw-apps-only is set, drop those rows; the model needs
            # to learn the closed-set AW inventory, not the open-set AC one.
            if args.aw_apps_only and row["gt_m3a"]["action_type"] == "open_app":
                if not _is_aw_app(row["gt_m3a"].get("app_name", "")):
                    n_filtered_open_app += 1
                    continue
            row["_image_root"] = AC_IMG_ROOT
            buckets[row["gt_m3a"]["action_type"]].append(row)
        if args.aw_apps_only:
            print(f"[prep] aw_apps_only filter dropped "
                  f"{n_filtered_open_app} AC open_app rows; kept "
                  f"{len(buckets.get('open_app', []))}")

        # Synthesize open_app rows for the closed-set AW inventory. Each
        # synthetic row teaches the model the action format and the goal
        # → app_name mapping; it does NOT memorize AW task templates because
        # the goal is a generic "Open the X app" string, not an AW task name.
        if args.synthesize_open_app > 0:
            syn = []
            for app, _pkg in HARNESS_APP_INVENTORY:
                for _ in range(args.synthesize_open_app):
                    gt = {"action_type": "open_app", "app_name": app}
                    user_text = render_m3a_prompt(
                        goal=f"Open the {app} app.",
                        history="",
                        ui_elements=[],
                        harness_parity=args.harness_parity,
                    )
                    asst = (f'Reason: Open the {app} app to begin the task.\n'
                            f'Action: {json.dumps(gt)}')
                    syn.append({
                        "messages": [
                            {"role": "user", "content": [
                                {"type": "image"},
                                {"type": "text", "text": user_text},
                            ]},
                            {"role": "assistant", "content": [
                                {"type": "text", "text": asst},
                            ]},
                        ],
                        "image": None,
                        "elements": [],
                        "gt_m3a": gt,
                        "_image_root": AC_IMG_ROOT,
                        "_synthetic": True,
                    })
            buckets["open_app"].extend(syn)
            print(f"[prep] synthesized {len(syn)} open_app rows "
                  f"({args.synthesize_open_app} per AW app)")

        # AndroidLab mix: load the converted AndroidLab SoM rows and add to
        # the same buckets. AndroidLab covers click/input_text/scroll/
        # navigate_back/status (no wait/open_app — those are AC-only).
        al_buckets: dict[str, list[dict]] = defaultdict(list)
        # r41 plan augmentation: load teacher-generated plans by goal.
        plans_by_goal: dict[str, str] = {}
        if args.plans_jsonl is not None and args.plans_jsonl.exists():
            with open(args.plans_jsonl) as f:
                for line in f:
                    p = json.loads(line)
                    if not p.get("infeasible"):
                        plans_by_goal[p["goal"]] = p["plan"]
            print(f"[prep] loaded {len(plans_by_goal)} teacher plans from "
                  f"{args.plans_jsonl}")

        def _augment_al_with_plan(r: dict) -> dict:
            """Inject r41 plan into an AL row's prompt + assistant text.

            AL rows are pre-built (have full M3A prompt). We extract the
            goal, look up the plan, then either:
              - step 0 (history_len == 0): assistant emits Plan: + reason+action
              - step 1+: prompt has `Plan from step 0:` injected before goal
            If no plan available for the goal, return row unchanged.
            """
            user_text = r["messages"][0]["content"][1]["text"]
            marker = "The current user goal/request is: "
            idx = user_text.find(marker)
            if idx < 0:
                return r
            goal_end = user_text.find("\n", idx + len(marker))
            goal = user_text[idx + len(marker):goal_end].strip()
            plan = plans_by_goal.get(goal)
            if not plan:
                return r
            step0 = (r.get("history_len", 0) == 0)
            new_r = json.loads(json.dumps(r))  # deep copy
            if step0:
                # Modify assistant: prepend Plan
                asst = new_r["messages"][1]["content"][0]["text"]
                new_r["messages"][1]["content"][0]["text"] = (
                    f"Plan:\n{plan}\n\n{asst}"
                )
            else:
                # Modify user prompt: inject Plan from step 0 before goal
                new_user = (user_text[:idx] +
                            f"Plan from step 0:\n{plan}\n\n" +
                            user_text[idx:])
                new_r["messages"][0]["content"][1]["text"] = new_user
            return new_r

        # r41 alignment filter: pre-pass to identify aligned goals.
        # Only enforce for trajectories whose step-0 action is open_app
        # (most common, easiest to align). For other start-action types,
        # default to "aligned" (no filter).
        aligned_goals: set[str] = set()
        misaligned_goals: set[str] = set()
        if plans_by_goal and args.plan_align_filter and args.androidlab_jsonl:
            marker = "The current user goal/request is: "
            with open(args.androidlab_jsonl) as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("history_len", -1) != 0:
                        continue
                    user_text = r["messages"][0]["content"][1]["text"]
                    idx = user_text.find(marker)
                    if idx < 0:
                        continue
                    end = user_text.find("\n", idx + len(marker))
                    g = user_text[idx + len(marker):end].strip()
                    plan = plans_by_goal.get(g)
                    if not plan:
                        continue
                    a0 = r.get("gt_m3a", {})
                    if a0.get("action_type") != "open_app":
                        # Don't filter non-open_app starts
                        aligned_goals.add(g)
                        continue
                    target_app = (a0.get("app_name") or "").strip().lower()
                    plan_first = plan.split("\n", 1)[0].lower()
                    # "1. Open the Markor app" — needs both "open" and target app
                    if "open" in plan_first and target_app and target_app in plan_first:
                        aligned_goals.add(g)
                    else:
                        misaligned_goals.add(g)
            print(f"[prep] plan alignment filter: {len(aligned_goals)} aligned, "
                  f"{len(misaligned_goals)} misaligned goals "
                  f"(of {len(plans_by_goal)} total plans)")

        if args.androidlab_jsonl is not None:
            n_planned = 0
            n_filtered = 0
            with open(args.androidlab_jsonl) as f:
                for line in f:
                    r = json.loads(line)
                    if plans_by_goal:
                        # Look up goal for filter check
                        ut = r["messages"][0]["content"][1]["text"]
                        m = "The current user goal/request is: "
                        i = ut.find(m)
                        gend = ut.find("\n", i + len(m)) if i >= 0 else -1
                        g = ut[i + len(m):gend].strip() if i >= 0 else ""
                        if (args.plan_align_filter and g in misaligned_goals):
                            n_filtered += 1
                            continue
                        before = r["messages"][1]["content"][0]["text"]
                        r = _augment_al_with_plan(r)
                        if r["messages"][1]["content"][0]["text"] != before:
                            n_planned += 1
                    al_buckets[r["gt_m3a"]["action_type"]].append(r)
            print(f"[prep] AndroidLab pool sizes: "
                  f"{ {k: len(v) for k, v in al_buckets.items()} }")
            if plans_by_goal:
                print(f"[prep] augmented {n_planned} AL rows with plans; "
                      f"filtered {n_filtered} rows from misaligned trajectories")
        # Genesis vision rows (teacher-rollout trajectories with screenshots).
        # These are NOT mixed with AC/AL data - indices are dataset-specific.
        # Train genesis data separately with its own pipeline.
        genesis_buckets: dict[str, list[dict]] = defaultdict(list)
        if args.genesis_jsonl is not None and args.genesis_jsonl.exists():
            with open(args.genesis_jsonl) as f:
                for line in f:
                    r = json.loads(line)
                    gt = r.get("gt_m3a", {})
                    if gt and gt.get("action_type") and r.get("image"):
                        genesis_buckets[gt["action_type"]].append(r)
            print(f"[prep] Genesis vision pool sizes: "
                  f"{ {k: len(v) for k, v in genesis_buckets.items()} }")
            # If genesis data is provided, write ONLY genesis rows (no mixing)
            print("[prep] WARNING: Genesis data provided - writing genesis-only training set")
            print("[prep] Genesis indices are dataset-specific and cannot be mixed with AC/AL")
        # Drop classes whose source pool is much smaller than the target —
        # otherwise we replay the same handful of rows many times, which
        # overfits to that class. Keep replay factor ≤ ~2.5x.
        min_pool = max(5, args.per_class_target // 3)
        skipped = [k for k, v in buckets.items() if len(v) < min_pool]
        for k in skipped:
            print(f"[prep] dropping action_type={k} ({len(buckets[k])} rows)")
            buckets.pop(k)
        rng2 = random.Random(args.seed + 1)
        n_train_written = 0
        def _target_for(at: str) -> int:
            if at == "input_text" and args.boost_input_text > 1:
                return args.per_class_target * args.boost_input_text
            return args.per_class_target
        # Optionally include status from AndroidLab as its own class.
        if args.include_status and "status" in al_buckets:
            classes_to_emit = list(buckets.keys()) + ["status"]
        else:
            classes_to_emit = list(buckets.keys())

        # Genesis vision training: write ONLY genesis rows (teacher distillation).
        # AC/AL data is kept completely separate - indices are dataset-specific.
        if genesis_buckets:
            print("[prep] Genesis vision mode - writing ONLY genesis rows")
            print("[prep] AC/AL data is NOT mixed (dataset-specific indices)")
            n_train_written = 0
            with open(out_train, "w") as f:
                for at in sorted(genesis_buckets.keys()):
                    gen_rows = genesis_buckets[at]
                    if not gen_rows:
                        continue
                    rng2.shuffle(gen_rows)
                    for row in gen_rows:
                        f.write(json.dumps(row) + "\n")
                        n_train_written += 1
                    print(f"[prep]   {at:18s} gen={len(gen_rows):4d} written")
            print(f"[prep] Genesis vision train: {n_train_written} rows total")
        else:
            # Text-only training: AC/AL mixing as before
            n_train_written = 0
            with open(out_train, "w") as f:
                for at in sorted(set(classes_to_emit)):
                    ac_rows = buckets.get(at, [])
                    al_rows = al_buckets.get(at, [])
                    target = _target_for(at)
                    if ac_rows and al_rows:
                        half = target // 2
                        s_ac = (rng2.sample(ac_rows, min(half, len(ac_rows)))
                                if len(ac_rows) >= half
                                else [rng2.choice(ac_rows) for _ in range(half)])
                        s_al = (rng2.sample(al_rows, min(target - half, len(al_rows)))
                                if len(al_rows) >= target - half
                                else [rng2.choice(al_rows) for _ in range(target - half)])
                        samp = s_ac + s_al
                        src_label = f"AC={len(s_ac)}+AL={len(s_al)}"
                    elif ac_rows or al_rows:
                        pool = ac_rows or al_rows
                        if not pool:
                            continue
                        if len(pool) >= target:
                            samp = rng2.sample(pool, target)
                        else:
                            samp = [rng2.choice(pool) for _ in range(target)]
                        src_label = f"{'AC' if ac_rows else 'AL'}={len(samp)}"
                    else:
                        continue

                    rng2.shuffle(samp)
                    for row in samp:
                        f.write(json.dumps(row) + "\n")
                        n_train_written += 1
                    print(f"[prep]   {at:18s} ac={len(ac_rows):4d} al={len(al_rows):4d} sampled={len(samp):4d} ({src_label})")
            print(f"[prep] AC/AL text-only train: {n_train_written} rows total")
    else:
        n_train_written = 0
        with open(out_train, "w") as f:
            for r in train_src:
                if n_train_written >= args.n_train:
                    break
                row = _build_row(r, _history_for(r, train_by_ep),
                                 harness_parity=args.harness_parity, compact_a11y=args.compact_a11y)
                if row is None:
                    continue
                f.write(json.dumps(row) + "\n")
                n_train_written += 1

    n_eval_written = 0
    with open(out_eval, "w") as f:
        for r in val_src:
            if n_eval_written >= args.n_eval:
                break
            row = _build_row(r, _history_for(r, val_by_ep),
                             harness_parity=args.harness_parity, compact_a11y=args.compact_a11y)
            if row is None:
                continue
            row["_image_root"] = AC_IMG_ROOT
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
