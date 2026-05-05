#!/usr/bin/env python3
"""Build a filtered AMEX SFT subset for pathZ/train_smoke.py."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ACTION_TYPES = ("tap", "scroll", "type", "status")
DEFAULT_RATIOS = {
    "tap": 0.45,
    "scroll": 0.25,
    "type": 0.15,
    "status": 0.15,
}
FALLBACK_IMAGE_DIR = Path("data/amex_full/extract_anchor/screenshot")


def read_jsonl(path: Path):
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                yield line_no, json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def action_type(row: dict[str, Any]) -> str | None:
    target = row.get("target") or row.get("gt_m3a")
    return target.get("action_type") if isinstance(target, dict) else None


def resolve_image(image: str | None, repo_root: Path) -> tuple[str | None, str]:
    if not image:
        return None, "empty"
    img_path = Path(image)
    if img_path.is_absolute():
        if img_path.is_file():
            try:
                return str(img_path.relative_to(repo_root)), "absolute_under_repo"
            except ValueError:
                return str(img_path), "absolute_external"
        return None, "missing"

    if (repo_root / img_path).is_file():
        return str(img_path), "original"

    fallback = FALLBACK_IMAGE_DIR / img_path.name
    if (repo_root / fallback).is_file():
        return str(fallback), "fallback_basename"

    return None, "missing"


def include_row(row: dict[str, Any], repo_root: Path,
                report: dict[str, Any]) -> tuple[bool, str | None]:
    at = action_type(row)
    report["source_action_distribution"][at or "<missing>"] += 1
    if at not in ACTION_TYPES:
        report["filter_drops"]["action_type"] += 1
        return False, None

    if at == "tap":
        match = row.get("match") or {}
        key = (
            f"strategy={match.get('strategy', '<missing>')};"
            f"ambiguous={bool(match.get('ambiguous'))}"
        )
        report["tap_filter_stats"]["seen"] += 1
        report["tap_filter_stats"]["by_strategy_ambiguity"][key] += 1
        if match.get("strategy") != "contained":
            report["tap_filter_stats"]["dropped_not_contained"] += 1
            report["filter_drops"]["tap_not_contained"] += 1
            return False, None
        if bool(match.get("ambiguous")):
            report["tap_filter_stats"]["dropped_ambiguous"] += 1
            report["filter_drops"]["tap_ambiguous"] += 1
            return False, None
        report["tap_filter_stats"]["kept"] += 1

    image, image_status = resolve_image(row.get("image"), repo_root)
    report["image_stats"][image_status] += 1
    if image is None:
        report["missing_images"] += 1
        report["filter_drops"]["missing_image"] += 1
        return False, None

    report["filtered_action_distribution"][at] += 1
    return True, image


def build_sft_row(src: dict[str, Any], image: str, repo_root: Path) -> dict[str, Any]:
    target = src["target"]
    assistant_text = f"Action: {compact_json(target)}"
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": src.get("prompt", "")},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": assistant_text},
            ]},
        ],
        "image": image,
        "_image_root": str(repo_root),
        "gt_m3a": target,
        "source": "amex_conversion_full",
        "source_action": src.get("source_action"),
        "match": src.get("match"),
        "episode_id": src.get("episode_id"),
        "step_id": src.get("step_id"),
    }


def desired_counts(total: int, available: dict[str, int]) -> dict[str, int]:
    counts = {k: min(available.get(k, 0), int(total * DEFAULT_RATIOS[k]))
              for k in ACTION_TYPES}
    while sum(counts.values()) < total:
        candidates = [
            k for k in ACTION_TYPES
            if counts[k] < available.get(k, 0)
        ]
        if not candidates:
            break
        candidates.sort(key=lambda k: (counts[k] / DEFAULT_RATIOS[k], k))
        counts[candidates[0]] += 1
    return counts


def take_split(buckets: dict[str, list[dict[str, Any]]], max_train: int,
               max_eval: int, seed: int) -> tuple[list[dict[str, Any]],
                                                  list[dict[str, Any]],
                                                  dict[str, Any]]:
    rng = random.Random(seed)
    shuffled: dict[str, list[dict[str, Any]]] = {}
    for at, rows in buckets.items():
        rows = sorted(rows, key=lambda r: (
            str(r.get("episode_id")),
            int(r.get("step_id") or -1),
            r.get("image") or "",
        ))
        rng.shuffle(rows)
        shuffled[at] = rows

    available = {at: len(shuffled.get(at, [])) for at in ACTION_TYPES}
    eval_counts = desired_counts(max_eval, available)
    remaining = {at: available[at] - eval_counts.get(at, 0)
                 for at in ACTION_TYPES}
    train_counts = desired_counts(max_train, remaining)

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for at in ACTION_TYPES:
        eval_n = eval_counts.get(at, 0)
        train_n = train_counts.get(at, 0)
        rows = shuffled.get(at, [])
        eval_rows.extend(rows[:eval_n])
        train_rows.extend(rows[eval_n:eval_n + train_n])

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    split_report = {
        "available": available,
        "requested_train": max_train,
        "requested_eval": max_eval,
        "selected_train": dict(Counter(action_type(r) for r in train_rows)),
        "selected_eval": dict(Counter(action_type(r) for r in eval_rows)),
    }
    return train_rows, eval_rows, split_report


def sample_for_report(row: dict[str, Any]) -> dict[str, Any]:
    user_text = row["messages"][0]["content"][1]["text"]
    assistant_text = row["messages"][1]["content"][0]["text"]
    return {
        "episode_id": row.get("episode_id"),
        "step_id": row.get("step_id"),
        "action_type": action_type(row),
        "image": row.get("image"),
        "assistant": assistant_text,
        "prompt_prefix": user_text[:300],
    }


def counter_report(obj: Any) -> Any:
    if isinstance(obj, Counter):
        return dict(obj)
    if isinstance(obj, defaultdict):
        return {k: counter_report(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: counter_report(v) for k, v in obj.items()}
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    default=Path("outputs/amex_conversion_full/amex_subset.converted.jsonl"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/pathZ/amex_sft_subset"))
    ap.add_argument("--max-train", type=int, default=15000)
    ap.add_argument("--max-eval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    if args.max_train < 0 or args.max_eval < 0:
        raise ValueError("--max-train and --max-eval must be non-negative")

    repo_root = Path.cwd().resolve()
    report: dict[str, Any] = {
        "src": str(args.src),
        "out_dir": str(args.out_dir),
        "seed": args.seed,
        "ratios": DEFAULT_RATIOS,
        "rows_read": 0,
        "missing_images": 0,
        "source_action_distribution": Counter(),
        "filtered_action_distribution": Counter(),
        "image_stats": Counter(),
        "filter_drops": Counter(),
        "tap_filter_stats": {
            "seen": 0,
            "kept": 0,
            "dropped_not_contained": 0,
            "dropped_ambiguous": 0,
            "by_strategy_ambiguity": Counter(),
        },
    }

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _line_no, src_row in read_jsonl(args.src):
        report["rows_read"] += 1
        ok, image = include_row(src_row, repo_root, report)
        if not ok or image is None:
            continue
        sft_row = build_sft_row(src_row, image, repo_root)
        buckets[action_type(sft_row)].append(sft_row)

    train_rows, eval_rows, split_report = take_split(
        buckets, args.max_train, args.max_eval, args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    eval_path = args.out_dir / "eval.jsonl"
    report_path = args.out_dir / "report.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    report["splits"] = split_report
    report["train_path"] = str(train_path)
    report["eval_path"] = str(eval_path)
    report["report_path"] = str(report_path)
    report["sample_rows"] = [
        sample_for_report(r) for r in (train_rows[:3] + eval_rows[:2])
    ]
    with report_path.open("w") as f:
        json.dump(counter_report(report), f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")

    print(f"[amex-sft-subset] read={report['rows_read']} "
          f"train={len(train_rows)} eval={len(eval_rows)}")
    print(f"[amex-sft-subset] train={split_report['selected_train']} "
          f"eval={split_report['selected_eval']}")
    print(f"[amex-sft-subset] wrote {train_path} {eval_path} {report_path}")


if __name__ == "__main__":
    main()
