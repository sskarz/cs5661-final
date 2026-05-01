"""M3A-format conversion + parsing helpers used by smoke prep / train / eval.

Converts AndroidControl-v3 (Path W schema) rows to M3A's exact prompt and
action vocabulary, which is what `m3a_gemma_wrapper.GemmaMultimodalWrapper`
sees at AW eval time. Also provides an action-match scorer that consumes
M3A-format actions on both sides.
"""
from __future__ import annotations

import json
import re

# Pulled from android_world/agents/m3a.py (PROMPT_PREFIX + GUIDANCE).
# Trimmed: dropped the long Text Related Operations section since AC has
# no copy/paste/select trajectories — that text is dead weight at smoke.
M3A_PROMPT_PREFIX = (
    "You are an agent who can operate an Android phone on behalf of a user.\n"
    "Based on the user's goal, you choose ONE action per step from the list below.\n"
    '- Click an element: {"action_type": "click", "index": <int>}\n'
    '- Long press an element: {"action_type": "long_press", "index": <int>}\n'
    '- Type into a text field: {"action_type": "input_text", "text": "<text>", "index": <int>}\n'
    '- Press Enter: {"action_type": "keyboard_enter"}\n'
    '- Navigate home: {"action_type": "navigate_home"}\n'
    '- Navigate back: {"action_type": "navigate_back"}\n'
    '- Scroll: {"action_type": "scroll", "direction": "<up|down|left|right>"}\n'
    '- Open an app: {"action_type": "open_app", "app_name": "<name>"}\n'
    '- Wait for screen update: {"action_type": "wait"}\n'
    '- Mark task complete: {"action_type": "status", "goal_status": "complete"}\n'
    '- Mark task infeasible: {"action_type": "status", "goal_status": "infeasible"}\n'
    '- Answer the user: {"action_type": "answer", "text": "<answer>"}\n'
    "Respond with EXACTLY the format:\nReason: <one sentence>\n"
    'Action: {"action_type": ...}\n'
)


def render_m3a_prompt(goal: str, history: str, ui_elements: list[dict]) -> str:
    """Render the M3A-style action-selection prompt as a single user-text block.

    Mirrors the AW-eval-time prompt that the model will see through the
    GemmaMultimodalWrapper, but compact (no long Guidance section).

    `ui_elements` is the row's `elements` list: each dict has `id`, `bbox`,
    `label`. We render in the same order as the legend the runtime emits.
    """
    if not history:
        history = "You just started, no action has been performed yet."
    lines = [
        M3A_PROMPT_PREFIX,
        f"\nThe current user goal/request is: {goal}\n",
        f"Here is a history of what you have done so far:\n{history}\n",
        "Here is the list of UI elements visible on screen "
        "(numeric indexes match the labeled screenshot):\n",
    ]
    for e in ui_elements:
        eid = e.get("id")
        label = (e.get("label") or "").strip()
        lines.append(f'  UI element {eid}: {{"index": {eid}, "text": "{label}"}}')
    lines.append("\nNow output an action from the above list.\n")
    lines.append("Reason: ...\nAction: {\"action_type\":...}\n\nYour Answer:\n")
    return "".join(lines)


# Path W → M3A action conversion. Path W stores
# {"action_type": "tap"|"type"|..., "action_args": {...}}; M3A wants the
# args FLAT under action_type with renamed keys.
PATHW_TO_M3A = {
    "tap": "click",
    "type": "input_text",
    "scroll": "scroll",
    "open_app": "open_app",
    "wait": "wait",
    "navigate_back": "navigate_back",
    "navigate_home": "navigate_home",
    "long_press": "long_press",
    "keyboard_enter": "keyboard_enter",
    "status": "status",
    "answer": "answer",
}


def pathw_to_m3a(action: dict) -> dict | None:
    """Convert Path W (v2) {action_type, action_args} → flat M3A schema.

    Returns None if the action is malformed.
    """
    if not isinstance(action, dict):
        return None
    at = action.get("action_type")
    if at is None:
        return None
    m3a_type = PATHW_TO_M3A.get(at, at)
    args = action.get("action_args") or {}
    if not isinstance(args, dict):
        return None
    out: dict = {"action_type": m3a_type}
    if m3a_type in ("click", "long_press"):
        eid = args.get("element_id")
        if eid is None:
            return None
        out["index"] = int(eid)
    elif m3a_type == "input_text":
        eid = args.get("element_id")
        out["text"] = args.get("text", "")
        if eid is not None:
            out["index"] = int(eid)
    elif m3a_type == "scroll":
        d = args.get("direction")
        if d is None:
            return None
        out["direction"] = d
    elif m3a_type == "open_app":
        out["app_name"] = args.get("app_name", "")
    elif m3a_type == "answer":
        out["text"] = args.get("text", "")
    elif m3a_type == "status":
        out["goal_status"] = args.get("goal_status", "complete")
    # wait / navigate_back / navigate_home / keyboard_enter: no args
    return out


_ACTION_RE = re.compile(r"Action:\s*(\{.*\})", flags=re.DOTALL)


def parse_m3a_emission(text: str) -> tuple[str | None, dict | None]:
    """Returns (reason, action_dict). Mirrors m3a_utils.parse_reason_action_output.

    Tolerant: works whether or not 'Reason:' is present.
    """
    reason_m = re.search(r"Reason:(.*)Action:", text, flags=re.DOTALL)
    reason = reason_m.group(1).strip() if reason_m else None
    m = _ACTION_RE.search(text)
    if not m:
        return reason, None
    raw = m.group(1)
    # brace-balanced first JSON object scan
    depth = 0
    start = -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return reason, json.loads(raw[start:i + 1])
                except Exception:
                    return reason, None
    return reason, None


def m3a_action_match(pred: dict | None, gt: dict) -> dict:
    """Score one prediction against an M3A-format gt action.

    Returns a dict of booleans:
      type_match  — action_type matches
      arg_match   — relevant grounding arg matches (index / direction / app_name / text)
      full_match  — both
    """
    if pred is None:
        return {"type_match": False, "arg_match": False, "full_match": False}
    pt = (pred.get("action_type") or "").lower()
    gt_t = (gt.get("action_type") or "").lower()
    if not pt:
        return {"type_match": False, "arg_match": False, "full_match": False}
    type_match = pt == gt_t
    if not type_match:
        return {"type_match": False, "arg_match": False, "full_match": False}
    arg_ok = True
    if gt_t in ("click", "long_press"):
        arg_ok = pred.get("index") == gt.get("index")
    elif gt_t == "input_text":
        # primary signal is the index — text is harder to match exactly at this scale
        arg_ok = pred.get("index") == gt.get("index")
    elif gt_t == "scroll":
        arg_ok = (pred.get("direction") or "").lower() == (gt.get("direction") or "").lower()
    elif gt_t == "open_app":
        a = (pred.get("app_name") or "").strip().lower()
        b = (gt.get("app_name") or "").strip().lower()
        arg_ok = a == b or (b in a) or (a in b)
    elif gt_t == "status":
        arg_ok = pred.get("goal_status") == gt.get("goal_status")
    # answer / wait / navigate_*  / keyboard_enter: no grounding arg
    return {"type_match": True, "arg_match": bool(arg_ok), "full_match": bool(arg_ok)}
