#!/usr/bin/env python3
"""Filter Genesis vision trajectories into aligned vision SFT rows.

This script is intentionally Genesis-only. It reads raw Genesis trajectory
records with per-step screenshots and emits:
  - strict complete trajectories/train rows/report
  - expanded trajectories/train rows/report with conservative no-terminal
    prefix salvage

No AndroidControl, AndroidLab, or AW benchmark template data is read.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


ACTION_TYPES = {
    "click",
    "long_press",
    "input_text",
    "keyboard_enter",
    "navigate_home",
    "navigate_back",
    "scroll",
    "open_app",
    "wait",
    "status",
    "answer",
}
INDEXED_ACTIONS = {"click", "long_press", "input_text"}
NON_TERMINAL_ACTIONS = ACTION_TYPES - {"status", "answer"}
UI_INDEX_RE = re.compile(r"UI element\s+(\d+):")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            rec["_raw_index"] = len(rows)
            rows.append(rec)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def image_ok(image_root: Path, rel: str | None) -> bool:
    if not rel:
        return False
    p = image_root / rel
    return p.exists() and p.is_file() and p.stat().st_size >= 100


def visible_indexes(prompt: str | None) -> set[int]:
    if not prompt:
        return set()
    return {int(m.group(1)) for m in UI_INDEX_RE.finditer(prompt)}


def action_signature(action: dict[str, Any] | None) -> tuple[Any, ...]:
    action = action or {}
    return (
        action.get("action_type"),
        action.get("index"),
        action.get("app_name"),
        action.get("direction"),
        action.get("text"),
        action.get("goal_status"),
    )


def status_value(step: dict[str, Any]) -> str | None:
    action = step.get("action_output_json") or {}
    if action.get("action_type") == "status":
        return action.get("goal_status")
    return None


def is_parseable_action(step: dict[str, Any]) -> bool:
    action = step.get("action_output_json")
    return isinstance(action, dict) and action.get("action_type") in ACTION_TYPES


def invalid_step_reasons(step: dict[str, Any], image_root: Path) -> list[str]:
    reasons = []
    if not step.get("action_prompt"):
        reasons.append("missing_prompt")
    if not step.get("action_output"):
        reasons.append("missing_action_output")
    if not is_parseable_action(step):
        reasons.append("missing_or_unparseable_action")
    if not image_ok(image_root, step.get("image")):
        reasons.append("missing_or_invalid_image")

    action = step.get("action_output_json") or {}
    if action.get("action_type") in INDEXED_ACTIONS:
        idx = action.get("index")
        if not isinstance(idx, int):
            reasons.append("missing_action_index")
        elif idx not in visible_indexes(step.get("action_prompt")):
            reasons.append("index_not_in_prompt")
    if action.get("action_type") == "scroll" and action.get("direction") not in {
        "up",
        "down",
        "left",
        "right",
    }:
        reasons.append("bad_scroll_direction")
    if action.get("action_type") == "open_app" and not action.get("app_name"):
        reasons.append("missing_app_name")
    return reasons


def trim_at_first_status(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for step in steps:
        out.append(step)
        if status_value(step) is not None:
            break
    return out


def max_consecutive_run(signatures: list[tuple[Any, ...]]) -> int:
    best = cur = 1 if signatures else 0
    for prev, sig in zip(signatures, signatures[1:]):
        if sig == prev:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def has_alternating_loop(signatures: list[tuple[Any, ...]]) -> bool:
    if len(signatures) < 5:
        return False
    for start in range(0, len(signatures) - 4):
        window = signatures[start : start + 5]
        if window[0] == window[2] == window[4] and window[1] == window[3]:
            if window[0] != window[1]:
                return True
    for start in range(0, len(signatures) - 5):
        window = signatures[start : start + 6]
        if window[0] == window[2] == window[4] and window[1] == window[3] == window[5]:
            if window[0] != window[1]:
                return True
    return False


def first_loop_start(signatures: list[tuple[Any, ...]]) -> int | None:
    for i in range(2, len(signatures)):
        if signatures[i] == signatures[i - 1] == signatures[i - 2]:
            return i - 2
    for i in range(4, len(signatures)):
        window = signatures[i - 4 : i + 1]
        if window[0] == window[2] == window[4] and window[1] == window[3]:
            if window[0] != window[1]:
                return i - 4
    for i in range(5, len(signatures)):
        window = signatures[i - 5 : i + 1]
        if window[0] == window[2] == window[4] and window[1] == window[3] == window[5]:
            if window[0] != window[1]:
                return i - 5
    return None


def has_low_diversity(signatures: list[tuple[Any, ...]]) -> bool:
    if len(signatures) < 6:
        return False
    counts = Counter(signatures)
    if len(counts) <= 2:
        return True
    return counts.most_common(1)[0][1] / len(signatures) >= 0.70


def changed_steps(steps: list[dict[str, Any]]) -> int:
    return sum(
        1
        for step in steps
        if "window changed" in (step.get("summary") or "").lower()
    )


def no_visible_steps(steps: list[dict[str, Any]]) -> int:
    return sum(
        1
        for step in steps
        if "no visible change" in (step.get("summary") or "").lower()
    )


def useful_steps(steps: list[dict[str, Any]], include_status: bool) -> list[dict[str, Any]]:
    out = []
    for step in steps:
        action = step.get("action_output_json") or {}
        at = action.get("action_type")
        if at == "status" and not include_status:
            continue
        if at in {"answer", "wait"}:
            continue
        out.append(step)
    return out


def trajectory_reasons(
    rec: dict[str, Any],
    steps: list[dict[str, Any]],
    image_root: Path,
    min_non_terminal: int,
    require_complete: bool,
) -> tuple[str, list[str]]:
    reasons = []
    if not steps:
        return "drop", ["empty_trajectory"]

    for step in steps:
        reasons.extend(invalid_step_reasons(step, image_root))
    if reasons:
        return "drop", sorted(set(reasons))

    statuses = [(i, status_value(step)) for i, step in enumerate(steps) if status_value(step)]
    if any(value == "infeasible" for _, value in statuses):
        return "drop", ["status_infeasible"]
    if statuses and statuses[0][1] != "complete":
        return "drop", [f"status_{statuses[0][1]}"]
    if statuses and "refusing status" in (steps[statuses[0][0]].get("summary") or "").lower():
        return "drop", ["refused_terminal_status"]

    non_terminal = [s for s in steps if (s.get("action_output_json") or {}).get("action_type") != "status"]
    if require_complete:
        if not statuses:
            return "drop", ["no_terminal_status"]
        if statuses[0][0] <= 1:
            return "drop", ["premature_complete"]
        if len(non_terminal[: statuses[0][0]]) < min_non_terminal:
            return "drop", ["too_short"]
        check_steps = steps[: statuses[0][0]]
    else:
        if len(non_terminal) < min_non_terminal:
            return "drop", ["too_short"]
        check_steps = non_terminal

    signatures = [action_signature(s.get("action_output_json")) for s in check_steps]
    if max_consecutive_run(signatures) >= 3 or has_alternating_loop(signatures):
        return "drop", ["repeated_action_loop"]
    if has_low_diversity(signatures):
        return "drop", ["low_diversity_action_pattern"]
    if changed_steps(check_steps) < 2 and len(check_steps) >= 4:
        return "drop", ["low_progress"]
    if no_visible_steps(check_steps) / max(1, len(check_steps)) > 0.50:
        return "drop", ["low_progress"]

    return "keep", []


def build_train_rows(
    rec: dict[str, Any],
    steps: list[dict[str, Any]],
    bucket: str,
    image_root: Path,
    salvage: bool,
) -> list[dict[str, Any]]:
    rows = []
    for step in steps:
        action = step.get("action_output_json")
        at = action.get("action_type")
        if salvage and at in {"status", "answer"}:
            continue
        row = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": step["action_prompt"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": step["action_output"]}],
                },
            ],
            "image": step["image"],
            "_image_root": str(image_root.resolve()),
            "_source": "genesis",
            "_genesis_app": rec.get("app"),
            "_genesis_goal": rec.get("goal"),
            "_genesis_step": step.get("step"),
            "_genesis_raw_index": rec.get("_raw_index"),
            "_genesis_bucket": bucket,
            "_genesis_salvaged_no_terminal": salvage,
            "gt_m3a": action,
        }
        rows.append(row)
    return rows


def clean_record(rec: dict[str, Any], steps: list[dict[str, Any]], bucket: str) -> dict[str, Any]:
    out = deepcopy({k: v for k, v in rec.items() if k != "_raw_index"})
    out["steps"] = steps
    out["n_steps"] = len(steps)
    out["_filter"] = {
        "raw_index": rec.get("_raw_index"),
        "bucket": bucket,
        "salvaged_no_terminal": bucket == "salvaged_no_terminal_prefix",
        "synthetic_terminal_status": False,
    }
    return out


def no_terminal_prefix(rec: dict[str, Any], image_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    prefix = []
    signatures = []
    for step in rec.get("steps", []):
        bad = invalid_step_reasons(step, image_root)
        if bad:
            return [], sorted(set(bad))
        action = step.get("action_output_json") or {}
        at = action.get("action_type")
        if at == "status":
            return [], [f"status_{action.get('goal_status')}"]
        if at in {"answer", "wait"}:
            break
        if at not in NON_TERMINAL_ACTIONS:
            return [], ["missing_or_unparseable_action"]
        prefix.append(step)
        signatures.append(action_signature(action))
        loop_start = first_loop_start(signatures)
        if loop_start is not None:
            prefix = prefix[:loop_start]
            break
    return prefix, []


def sample_record(rec: dict[str, Any], reason: str | None = None) -> dict[str, Any]:
    actions = []
    for step in rec.get("steps", [])[:6]:
        action = step.get("action_output_json") or {}
        actions.append(
            {
                k: action.get(k)
                for k in ("action_type", "index", "app_name", "direction", "text", "goal_status")
                if action.get(k) is not None
            }
        )
    out = {
        "index": rec.get("_raw_index", (rec.get("_filter") or {}).get("raw_index")),
        "app": rec.get("app"),
        "goal": rec.get("goal"),
        "n_steps": len(rec.get("steps", [])),
        "first_actions": actions,
    }
    if reason:
        out["reason"] = reason
    return out


def summarize(
    src: Path,
    image_root: Path,
    out_paths: dict[str, Path],
    raw: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    reasons: Counter[str],
    samples_kept: list[dict[str, Any]],
    samples_dropped: list[dict[str, Any]],
    policy: str,
) -> dict[str, Any]:
    by_app = Counter(r.get("app", "?") for r in kept)
    train_actions = Counter((r.get("gt_m3a") or {}).get("action_type") for r in train_rows)
    buckets = Counter((r.get("_filter") or {}).get("bucket") for r in kept)
    return {
        "source": str(src.resolve()),
        "image_root": str(image_root.resolve()),
        "outputs": {k: str(v.resolve()) for k, v in out_paths.items()},
        "raw_trajectories": len(raw),
        "raw_steps": sum(len(r.get("steps", [])) for r in raw),
        "kept_trajectories": len(kept),
        "kept_steps_or_train_rows": len(train_rows),
        "dropped_trajectories": len(raw) - len(kept),
        "drop_reasons": dict(reasons),
        "kept_buckets": dict(buckets),
        "kept_by_app": dict(sorted(by_app.items())),
        "train_action_counts": dict(train_actions),
        "missing_train_images": sum(1 for r in train_rows if not image_ok(image_root, r.get("image"))),
        "synthetic_terminal_status_rows": 0,
        "filter_policy": policy,
        "kept_samples": samples_kept[:8],
        "dropped_samples": samples_dropped[:12],
    }


def run_filter(args: argparse.Namespace) -> None:
    src = args.src
    image_root = args.image_root or (src.parent / "screenshots")
    raw = load_jsonl(src)
    out_dir = args.out_dir or src.parent

    strict_traj: list[dict[str, Any]] = []
    strict_rows: list[dict[str, Any]] = []
    strict_reasons: Counter[str] = Counter()
    strict_kept_samples = []
    strict_dropped_samples = []

    for rec in raw:
        steps = trim_at_first_status(rec.get("steps", []))
        decision, why = trajectory_reasons(
            rec,
            steps,
            image_root,
            min_non_terminal=args.min_strict_actions,
            require_complete=True,
        )
        if decision == "keep":
            clean = clean_record(rec, steps, "strict_complete")
            strict_traj.append(clean)
            strict_rows.extend(build_train_rows(rec, steps, "strict_complete", image_root, False))
            if len(strict_kept_samples) < 8:
                strict_kept_samples.append(sample_record(clean))
        else:
            reason = why[0]
            strict_reasons[reason] += 1
            if len(strict_dropped_samples) < 12:
                strict_dropped_samples.append(sample_record(rec, reason))

    expanded_traj = list(strict_traj)
    expanded_rows = list(strict_rows)
    expanded_reasons: Counter[str] = Counter()
    expanded_kept_samples = list(strict_kept_samples)
    expanded_dropped_samples = []
    strict_raw_indexes = {(r.get("_filter") or {}).get("raw_index") for r in strict_traj}

    for rec in raw:
        if rec.get("_raw_index") in strict_raw_indexes:
            continue
        statuses = [status_value(s) for s in rec.get("steps", []) if status_value(s)]
        if statuses:
            terminal_steps = trim_at_first_status(rec.get("steps", []))
            _, why = trajectory_reasons(
                rec,
                terminal_steps,
                image_root,
                min_non_terminal=args.min_strict_actions,
                require_complete=True,
            )
            reason = why[0] if why else f"status_{statuses[0]}"
            expanded_reasons[reason] += 1
            if len(expanded_dropped_samples) < 12:
                expanded_dropped_samples.append(sample_record(rec, reason))
            continue

        prefix, prefix_bad = no_terminal_prefix(rec, image_root)
        if prefix_bad:
            reason = prefix_bad[0]
            expanded_reasons[reason] += 1
            if len(expanded_dropped_samples) < 12:
                expanded_dropped_samples.append(sample_record(rec, reason))
            continue
        decision, why = trajectory_reasons(
            rec,
            prefix,
            image_root,
            min_non_terminal=args.min_salvage_actions,
            require_complete=False,
        )
        if decision == "keep":
            clean = clean_record(rec, prefix, "salvaged_no_terminal_prefix")
            expanded_traj.append(clean)
            expanded_rows.extend(
                build_train_rows(rec, prefix, "salvaged_no_terminal_prefix", image_root, True)
            )
            if len(expanded_kept_samples) < 8:
                expanded_kept_samples.append(sample_record(clean))
        else:
            reason = why[0]
            expanded_reasons[reason] += 1
            if len(expanded_dropped_samples) < 12:
                expanded_dropped_samples.append(sample_record(rec, reason))

    strict_paths = {
        "trajectories": out_dir / args.strict_trajectories,
        "train": out_dir / args.strict_train,
        "report": out_dir / args.strict_report,
    }
    expanded_paths = {
        "trajectories": out_dir / args.expanded_trajectories,
        "train": out_dir / args.expanded_train,
        "report": out_dir / args.expanded_report,
    }
    strict_report = summarize(
        src,
        image_root,
        strict_paths,
        raw,
        strict_traj,
        strict_rows,
        strict_reasons,
        strict_kept_samples,
        strict_dropped_samples,
        "strict: keep only first terminal status=complete trajectories with valid screenshots/actions, "
        ">=2 non-terminal actions, no premature complete, no infeasible status, no loops, "
        "low-diversity patterns, or low-progress prefixes; truncate at first complete status.",
    )
    expanded_report = summarize(
        src,
        image_root,
        expanded_paths,
        raw,
        expanded_traj,
        expanded_rows,
        expanded_reasons,
        expanded_kept_samples,
        expanded_dropped_samples,
        "expanded: strict complete set plus conservative no-terminal prefixes only; salvage requires "
        "valid screenshots/actions, >=3 non-terminal non-answer/non-wait actions, at least two screen "
        "changes for longer prefixes, no explicit infeasible status, no repeated/alternating loops, "
        "and no low-diversity pattern. No synthetic terminal status rows are created.",
    )

    write_jsonl(strict_paths["trajectories"], strict_traj)
    write_jsonl(strict_paths["train"], strict_rows)
    strict_paths["report"].write_text(json.dumps(strict_report, indent=2, ensure_ascii=False) + "\n")

    write_jsonl(expanded_paths["trajectories"], expanded_traj)
    write_jsonl(expanded_paths["train"], expanded_rows)
    expanded_paths["report"].write_text(json.dumps(expanded_report, indent=2, ensure_ascii=False) + "\n")

    print(f"[filter_genesis_vision] raw trajectories: {len(raw)}")
    print(
        f"[filter_genesis_vision] strict: "
        f"{len(strict_traj)} trajectories, {len(strict_rows)} train rows -> {strict_paths['train']}"
    )
    print(
        f"[filter_genesis_vision] expanded: "
        f"{len(expanded_traj)} trajectories, {len(expanded_rows)} train rows -> {expanded_paths['train']}"
    )
    print("[filter_genesis_vision] expanded buckets:")
    for bucket, count in Counter((r.get("_filter") or {}).get("bucket") for r in expanded_traj).most_common():
        print(f"  {bucket:30s} {count:4d}")
    print("[filter_genesis_vision] expanded action counts:")
    for action, count in Counter((r.get("gt_m3a") or {}).get("action_type") for r in expanded_rows).most_common():
        print(f"  {action:30s} {count:4d}")
    print(f"METRIC genesis_strict_train_rows={len(strict_rows)}")
    print(f"METRIC genesis_expanded_train_rows={len(expanded_rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/pathZ/genesis_vision_rebuild/trajectories.jsonl"))
    ap.add_argument("--image-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--min-strict-actions", type=int, default=2)
    ap.add_argument("--min-salvage-actions", type=int, default=3)
    ap.add_argument("--strict-trajectories", default="trajectories.strict.jsonl")
    ap.add_argument("--strict-train", default="train.strict.jsonl")
    ap.add_argument("--strict-report", default="filter_report.strict.json")
    ap.add_argument("--expanded-trajectories", default="trajectories.expanded.jsonl")
    ap.add_argument("--expanded-train", default="train.expanded.jsonl")
    ap.add_argument("--expanded-report", default="filter_report.expanded.json")
    args = ap.parse_args()
    run_filter(args)


if __name__ == "__main__":
    main()
