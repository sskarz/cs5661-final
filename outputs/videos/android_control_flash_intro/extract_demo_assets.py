#!/usr/bin/env python3
import gzip
import json
import pickle
from pathlib import Path

from PIL import Image


ROOT = Path("/home/sanskar/Documents/Github/cs5661-final")
OUT = ROOT / "outputs/videos/android_control_flash_intro/assets"
FRAMES = OUT / "demo_frames"

RUNS = [
    {
        "kind": "trained",
        "label": "Trained r62 AW smoke",
        "path": Path("/home/sanskar/android_world/runs/r62_qwen35_vision_cont_aw_smoke/run_20260504T220820638384/RecipeDeleteSingleRecipe_0.pkl.gz"),
    },
    {
        "kind": "trained",
        "label": "Trained r62 AW-116",
        "path": Path("/home/sanskar/android_world/runs/r62_qwen35_vision_cont_aw116/run_20260505T011652695623/RecipeDeleteSingleRecipe_0.pkl.gz"),
    },
    {
        "kind": "baseline",
        "label": "Baseline AW smoke",
        "path": Path("/home/sanskar/android_world/runs/aw_smoke_baseline/run_20260501T054858070144/RecipeDeleteSingleRecipe_0.pkl.gz"),
    },
    {
        "kind": "baseline",
        "label": "Baseline m3a smoke",
        "path": Path("/home/sanskar/android_world/runs/m3a_baseline_smoke/run_20260430T175629176216/SystemWifiTurnOn_0.pkl.gz"),
    },
]


def load_pickle(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def get_episode(blob):
    if isinstance(blob, list) and blob:
        blob = blob[0]
    if isinstance(blob, dict):
        return blob.get("episode_data", blob)
    return {}


def first_value(data, keys, default=None):
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return default


def action_label(ep, idx):
    for key in ("action_output_json", "action_output", "action_reason", "summary"):
        values = ep.get(key)
        if isinstance(values, list) and idx < len(values) and values[idx]:
            value = values[idx]
            if isinstance(value, dict):
                action = value.get("action_type") or value.get("action") or "action"
                args = value.get("action_args") or value
                if isinstance(args, dict) and "index" in args:
                    return f"{action}(index={args['index']})"
                if isinstance(args, dict) and "element_id" in args:
                    return f"{action}(id={args['element_id']})"
                if isinstance(args, dict) and "text" in args:
                    return f"{action}(text)"
                return action
            text = str(value).replace("\n", " ")
            return text[:96]
    return "observe"


def save_frame(frame, path):
    if isinstance(frame, Image.Image):
        img = frame
    else:
        img = Image.fromarray(frame)
    img.save(path)
    return img.size


def main():
    FRAMES.mkdir(parents=True, exist_ok=True)
    manifest = {"episodes": []}

    for run in RUNS:
        if not run["path"].exists():
            continue
        blob = load_pickle(run["path"])
        ep = get_episode(blob)
        screenshots = ep.get("raw_screenshot") or ep.get("screenshots") or []
        if not screenshots:
            continue

        success = first_value(ep, ["success", "is_successful", "task_success"], None)
        goal = first_value(ep, ["goal", "task_goal", "instruction", "task_instruction"], run["path"].stem)
        episode_name = run["path"].name.replace(".pkl.gz", "")
        count = min(4, len(screenshots))
        if len(screenshots) > count:
            picks = sorted(set([0, len(screenshots) // 3, (2 * len(screenshots)) // 3, len(screenshots) - 1]))[:count]
        else:
            picks = list(range(count))

        frames = []
        for order, idx in enumerate(picks):
            name = f"{run['kind']}_{episode_name}_{order:02d}.png"
            rel = f"demo_frames/{name}"
            size = save_frame(screenshots[idx], FRAMES / name)
            frames.append(
                {
                    "file": rel,
                    "step_index": idx,
                    "action_label": action_label(ep, idx),
                    "size": size,
                }
            )

        manifest["episodes"].append(
            {
                "kind": run["kind"],
                "label": run["label"],
                "source_pickle": str(run["path"]),
                "episode_name": episode_name,
                "goal": str(goal),
                "success": bool(success) if success is not None else None,
                "frame_count_available": len(screenshots),
                "frames": frames,
            }
        )

    (OUT / "demo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2)[:4000])


if __name__ == "__main__":
    main()
