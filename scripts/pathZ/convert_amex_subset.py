#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev


PROMPT_PREFIX = """You are an agent operating an Android phone for the user.
Choose exactly one action for the current screen.
Allowed actions:
- tap an element: {"action_type":"tap","action_args":{"element_id":<int>}}
- type text: {"action_type":"type","action_args":{"text":"<text>"}}
- scroll: {"action_type":"scroll","action_args":{"direction":"up|down|left|right"}}
- navigate_back: {"action_type":"navigate_back","action_args":{}}
- navigate_home: {"action_type":"navigate_home","action_args":{}}
- press_enter: {"action_type":"press_enter","action_args":{}}
- mark complete: {"action_type":"status","action_args":{"goal_status":"complete"}}
- impossible: {"action_type":"status","action_args":{"goal_status":"impossible"}}
"""


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def unique_nonempty(values: list) -> list[str]:
    seen = set()
    out = []
    for value in values or []:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def element_label(element: dict) -> str:
    parts = unique_nonempty(element.get("xml_desc") or [])
    functionality = str(element.get("functionality") or "").strip()
    if functionality and functionality not in parts:
        parts.append(functionality)
    return " | ".join(parts) if parts else "<unlabeled>"


def bbox_area(bbox: list[int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def construct_candidates(element_doc: dict, include_scrollable: bool = False) -> tuple[list[dict], dict]:
    raw = []
    for source, elements in (
        ("clickable", element_doc.get("clickable_elements") or []),
        ("scrollable", element_doc.get("scrollable_elements") or []),
    ):
        if source == "scrollable" and not include_scrollable:
            continue
        for source_index, element in enumerate(elements):
            bbox = [int(v) for v in element.get("bbox", [0, 0, 0, 0])]
            x1, y1, x2, y2 = bbox
            raw.append(
                {
                    "source": source,
                    "source_index": source_index,
                    "bbox": bbox,
                    "label": element_label(element),
                    "min_side": min(max(0, x2 - x1), max(0, y2 - y1)),
                    "area": bbox_area(bbox),
                }
            )

    kept = [e for e in raw if e["min_side"] >= 16]
    tiny_dropped = len(raw) - len(kept)
    if raw and not kept:
        kept = raw
        tiny_dropped = 0

    kept.sort(
        key=lambda e: (
            0 if e["source"] == "clickable" else 1,
            e["bbox"][1],
            e["bbox"][0],
            e["area"],
            e["source_index"],
        )
    )
    for idx, element in enumerate(kept):
        element["element_id"] = idx
    return kept, {"raw": len(raw), "kept": len(kept), "tiny_dropped": tiny_dropped}


def contains(bbox: list[int], point: tuple[float, float]) -> bool:
    x1, y1, x2, y2 = bbox
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2


def center_distance(bbox: list[int], point: tuple[float, float]) -> float:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return math.hypot(point[0] - cx, point[1] - cy)


def match_tap(step: dict, candidates: list[dict]) -> tuple[dict | None, dict]:
    point = tuple(step.get("touch_coord") or [0, 0])
    device = step.get("device_dim") or [0, 0]
    contained = [e for e in candidates if contains(e["bbox"], point)]
    if contained:
        contained.sort(key=lambda e: (e["area"], center_distance(e["bbox"], point), e["element_id"]))
        winner = contained[0]
        ambiguous = len(contained) > 1
        distance = center_distance(winner["bbox"], point)
        return winner, {
            "strategy": "contained",
            "contained": True,
            "ambiguous": ambiguous,
            "candidate_count": len(candidates),
            "contained_ids": [e["element_id"] for e in contained],
            "distance_px": round(distance, 3),
            "acceptable": True,
        }

    if not candidates:
        return None, {
            "strategy": "none",
            "contained": False,
            "ambiguous": False,
            "candidate_count": 0,
            "distance_px": None,
            "acceptable": False,
            "reason": "no_candidates",
        }

    nearest = min(candidates, key=lambda e: (center_distance(e["bbox"], point), e["element_id"]))
    distance = center_distance(nearest["bbox"], point)
    diag = math.hypot(float(device[0] or 0), float(device[1] or 0))
    threshold = max(80.0, diag * 0.04)
    acceptable = distance <= threshold
    return nearest if acceptable else None, {
        "strategy": "nearest",
        "contained": False,
        "ambiguous": False,
        "candidate_count": len(candidates),
        "nearest_id": nearest["element_id"],
        "distance_px": round(distance, 3),
        "acceptable": acceptable,
        "threshold_px": round(threshold, 3),
    }


def swipe_direction(step: dict) -> tuple[str | None, dict]:
    touch = step.get("touch_coord") or [0, 0]
    lift = step.get("lift_coord") or [0, 0]
    dx = float(lift[0] - touch[0])
    dy = float(lift[1] - touch[1])
    abs_dx, abs_dy = abs(dx), abs(dy)
    clear = max(abs_dx, abs_dy) >= 80 and max(abs_dx, abs_dy) >= min(abs_dx, abs_dy) * 1.5
    if not clear:
        return None, {"dx": dx, "dy": dy, "clear": False}
    if abs_dy >= abs_dx:
        direction = "up" if dy < 0 else "down"
    else:
        direction = "left" if dx < 0 else "right"
    return direction, {"dx": dx, "dy": dy, "clear": True, "direction_policy": "finger displacement"}


def render_prompt(instruction: str, previous_action: str, candidates: list[dict]) -> str:
    lines = [
        PROMPT_PREFIX,
        f"\nUser goal: {instruction}\n",
        f"Previous action: {previous_action}\n",
        "Elements:",
    ]
    if candidates:
        for element in candidates:
            bbox = element["bbox"]
            lines.append(
                f'{element["element_id"]}. {element["label"]} '
                f'[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}] ({element["source"]})'
            )
    else:
        lines.append("<none>")
    lines.append('Return only JSON with "action_type" and "action_args".')
    return "\n".join(lines)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = (len(values) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return round(values[lo], 6)
    return round(values[lo] * (hi - idx) + values[hi] * (idx - lo), 6)


def basic_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "p50": None, "max": None}
    return {
        "count": len(values),
        "mean": round(mean(values), 6),
        "std": round(pstdev(values), 6),
        "min": round(min(values), 6),
        "p50": percentile(values, 0.5),
        "max": round(max(values), 6),
    }


def find_element_doc(element_dir: Path) -> dict[str, dict]:
    docs = {}
    for path in sorted(element_dir.glob("*.json")):
        doc = load_json(path)
        image_path = doc.get("image_path")
        if image_path:
            docs[Path(image_path).name] = doc
    return docs


def verdict(report: dict) -> str:
    total = report["total_steps"]
    converted = report["converted_examples"]
    tap = report["tap_steps"]
    if total < 20:
        return "not enough data: subset is too small for a clean/usable dataset verdict"
    if converted / max(total, 1) >= 0.9 and tap["mapped"] / max(tap["total"], 1) >= 0.85:
        return "clean"
    if converted / max(total, 1) >= 0.7:
        return "usable"
    return "not enough data"


def convert(args: argparse.Namespace) -> dict:
    subset = args.subset_dir
    instruction_dir = subset / "sample" / "instruction_anno"
    element_docs = find_element_doc(subset / "sample" / "element_anno")
    screenshot_dir = subset / "sample" / "screenshot"

    examples = []
    rows = []
    skipped = []
    source_counts = Counter()
    target_counts = Counter()
    candidate_counts = []
    nearest_distances = []
    all_touch_norm_x = []
    all_touch_norm_y = []
    center_dist_norm = []
    tap_stats = Counter(total=0, mapped=0, contained=0, nearest=0, ambiguous=0, unmatched=0)
    tap_with_type_text = 0

    for instruction_path in sorted(instruction_dir.glob("*.json")):
        episode = load_json(instruction_path)
        instruction = episode.get("instruction", "")
        episode_id = str(episode.get("episode_id", instruction_path.stem))
        previous_action = "<none>"
        for step in sorted(episode.get("steps") or [], key=lambda s: int(s.get("step_id", 0))):
            step_id = int(step.get("step_id", 0))
            source_action = str(step.get("action", "")).upper()
            source_counts[source_action] += 1
            image_name = Path(step.get("image_path", "")).name
            element_doc = element_docs.get(image_name, {"clickable_elements": [], "scrollable_elements": []})
            include_scrollable = source_action in {"SWIPE", "TASK_COMPLETE"} or not element_doc.get("clickable_elements")
            candidates, candidate_diag = construct_candidates(element_doc, include_scrollable=include_scrollable)
            candidate_counts.append(len(candidates))

            touch = step.get("touch_coord") or [0, 0]
            device = step.get("device_dim") or [0, 0]
            if device and float(device[0] or 0) > 0 and float(device[1] or 0) > 0:
                nx = float(touch[0]) / float(device[0])
                ny = float(touch[1]) / float(device[1])
                all_touch_norm_x.append(nx)
                all_touch_norm_y.append(ny)
                center_dist_norm.append(math.hypot(nx - 0.5, ny - 0.5))

            target = None
            match = {"candidate_diagnostics": candidate_diag}
            skip_reason = None
            type_text = str(step.get("type_text") or "")

            if source_action == "TAP" and type_text:
                tap_with_type_text += 1
                target = {"action_type": "type", "action_args": {"text": type_text}}
                match["type_policy"] = "AMEX TAP with nonempty type_text mapped to type without element_id"
            elif source_action == "TYPE" or type_text:
                if type_text:
                    target = {"action_type": "type", "action_args": {"text": type_text}}
                else:
                    skip_reason = "type_without_text"
            elif source_action == "TAP":
                tap_stats["total"] += 1
                winner, tap_match = match_tap(step, candidates)
                match.update(tap_match)
                if tap_match.get("ambiguous"):
                    tap_stats["ambiguous"] += 1
                if tap_match.get("contained"):
                    tap_stats["contained"] += 1
                elif tap_match.get("strategy") == "nearest":
                    tap_stats["nearest"] += 1
                    if tap_match.get("distance_px") is not None:
                        nearest_distances.append(float(tap_match["distance_px"]))
                if winner is not None:
                    tap_stats["mapped"] += 1
                    target = {"action_type": "tap", "action_args": {"element_id": winner["element_id"]}}
                    match["matched_element"] = winner
                else:
                    tap_stats["unmatched"] += 1
                    skip_reason = "tap_unmatched_or_low_confidence"
            elif source_action == "SWIPE":
                direction, swipe_diag = swipe_direction(step)
                match.update(swipe_diag)
                if direction:
                    target = {"action_type": "scroll", "action_args": {"direction": direction}}
                else:
                    skip_reason = "unclear_swipe_displacement"
            elif source_action == "TASK_COMPLETE":
                target = {"action_type": "status", "action_args": {"goal_status": "complete"}}
                match["interest_region"] = step.get("interest_region")
                match["status_policy"] = "TASK_COMPLETE mapped to status complete"
            elif source_action == "PRESS_BACK":
                target = {"action_type": "navigate_back", "action_args": {}}
            elif source_action == "PRESS_HOME":
                target = {"action_type": "navigate_home", "action_args": {}}
            elif source_action == "PRESS_ENTER":
                target = {"action_type": "press_enter", "action_args": {}}
            elif source_action == "TASK_IMPOSSIBLE":
                target = {"action_type": "status", "action_args": {"goal_status": "impossible"}}
                match["status_policy"] = "TASK_IMPOSSIBLE mapped to status impossible"
            else:
                skip_reason = f"unsupported_action:{source_action}"

            row = {
                "episode_id": episode_id,
                "step_id": step_id,
                "image": str(screenshot_dir / image_name),
                "source_action": source_action,
                "target_action": target["action_type"] if target else None,
                "candidate_count": len(candidates),
                "skip_reason": skip_reason,
                "match": match,
            }
            rows.append(row)

            if target is None:
                skipped.append(row)
                continue

            target_counts[target["action_type"]] += 1
            example = {
                "episode_id": episode_id,
                "step_id": step_id,
                "image": str(screenshot_dir / image_name),
                "instruction": instruction,
                "previous_action": previous_action,
                "prompt": render_prompt(instruction, previous_action, candidates),
                "target": target,
                "source_action": step,
                "match": match,
            }
            examples.append(example)
            previous_action = json.dumps(target, sort_keys=True, separators=(",", ":"))

    report = {
        "subset_dir": str(subset),
        "total_steps": sum(source_counts.values()),
        "converted_examples": len(examples),
        "skipped_examples": len(skipped),
        "counts_by_source_action": dict(sorted(source_counts.items())),
        "counts_by_target_action": dict(sorted(target_counts.items())),
        "tap_steps": dict(tap_stats),
        "candidate_count_distribution": basic_stats([float(v) for v in candidate_counts]),
        "normalized_touch_coordinate_mean_std": {
            "x": {"mean": round(mean(all_touch_norm_x), 6), "std": round(pstdev(all_touch_norm_x), 6)}
            if all_touch_norm_x else {"mean": None, "std": None},
            "y": {"mean": round(mean(all_touch_norm_y), 6), "std": round(pstdev(all_touch_norm_y), 6)}
            if all_touch_norm_y else {"mean": None, "std": None},
        },
        "center_bias_metrics": {
            "mean_distance_from_screen_center_norm": round(mean(center_dist_norm), 6) if center_dist_norm else None,
            "within_center_half_screen_fraction": round(
                sum(d <= 0.25 for d in center_dist_norm) / len(center_dist_norm), 6
            ) if center_dist_norm else None,
        },
        "nearest_mapping_distance_px": basic_stats(nearest_distances),
        "tap_with_type_text_policy": {
            "count": tap_with_type_text,
            "policy": "map AMEX TAP with nonempty type_text to type(text) and omit element_id",
        },
        "action_policies": {
            "swipe_direction": "direction is the finger displacement from touch_coord to lift_coord",
            "task_complete": 'mapped to {"action_type":"status","action_args":{"goal_status":"complete"}}',
            "task_impossible": 'mapped to {"action_type":"status","action_args":{"goal_status":"impossible"}}',
            "press_back": 'mapped to {"action_type":"navigate_back","action_args":{}}',
            "press_home": 'mapped to {"action_type":"navigate_home","action_args":{}}',
            "press_enter": 'mapped to {"action_type":"press_enter","action_args":{}}',
            "tap_confidence": "containment is always acceptable; nearest requires distance <= max(80px, 4% of device diagonal)",
        },
        "per_step_summary_rows": rows,
    }
    report["verdict"] = verdict(report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / "amex_subset.converted.jsonl"
    report_json_path = args.out_dir / "amex_subset.report.json"
    report_md_path = args.out_dir / "amex_subset.report.md"
    with jsonl_path.open("w") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    with report_json_path.open("w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    report_md_path.write_text(render_markdown(report, jsonl_path, report_json_path), encoding="utf-8")
    return report


def render_markdown(report: dict, jsonl_path: Path, report_json_path: Path) -> str:
    lines = [
        "# AMEX Conversion Probe",
        "",
        f"- Converted JSONL: `{jsonl_path}`",
        f"- Report JSON: `{report_json_path}`",
        f"- Total steps: {report['total_steps']}",
        f"- Converted examples: {report['converted_examples']}",
        f"- Skipped examples: {report['skipped_examples']}",
        f"- Verdict: {report['verdict']}",
        "",
        "## Counts",
        "",
        f"- Source actions: `{json.dumps(report['counts_by_source_action'], sort_keys=True)}`",
        f"- Target actions: `{json.dumps(report['counts_by_target_action'], sort_keys=True)}`",
        "",
        "## Tap Matching",
        "",
        f"- Tap stats: `{json.dumps(report['tap_steps'], sort_keys=True)}`",
        f"- Nearest distance px: `{json.dumps(report['nearest_mapping_distance_px'], sort_keys=True)}`",
        "",
        "## Candidate And Coordinate Metrics",
        "",
        f"- Candidate count distribution: `{json.dumps(report['candidate_count_distribution'], sort_keys=True)}`",
        f"- Normalized touch mean/std: `{json.dumps(report['normalized_touch_coordinate_mean_std'], sort_keys=True)}`",
        f"- Center bias: `{json.dumps(report['center_bias_metrics'], sort_keys=True)}`",
        "",
        "## Policies",
        "",
    ]
    for key, value in report["action_policies"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Per-Step Summary", "", "| step | source | target | candidates | skip | match |", "| ---: | --- | --- | ---: | --- | --- |"])
    for row in report["per_step_summary_rows"]:
        match = row["match"]
        compact = {
            k: match.get(k)
            for k in ("strategy", "contained", "ambiguous", "distance_px", "direction", "clear", "interest_region")
            if k in match
        }
        lines.append(
            f"| {row['step_id']} | {row['source_action']} | {row['target_action'] or ''} | "
            f"{row['candidate_count']} | {row['skip_reason'] or ''} | "
            f"`{json.dumps(compact, ensure_ascii=False, sort_keys=True)}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-dir", type=Path, default=Path("data/amex_subset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/amex_conversion_probe"))
    args = parser.parse_args()
    report = convert(args)
    print(
        "converted={converted_examples} skipped={skipped_examples} verdict={verdict}".format(
            **report
        )
    )


if __name__ == "__main__":
    main()
