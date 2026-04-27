#!/usr/bin/env python3
"""
Step 1 — Data Preparation: Download and format AndroidControl for SFT training.

Downloads `smolagents/android-control` from HuggingFace, explodes episodes into
step-level samples, normalizes click coordinates per-image using actual PNG
dimensions, writes screenshots to disk, and emits JSONL training data ready for
Unsloth QLoRA fine-tuning.

Usage (Mac smoke test):
    uv run python scripts/prepare_androidcontrol.py \\
        --output-dir data/androidcontrol_test --max-episodes 5

Usage (NVIDIA training box, full):
    uv run python scripts/prepare_androidcontrol.py \\
        --output-dir data/androidcontrol --fetch-ood-splits

Schema notes (verified against smolagents/android-control 2026-04):
  - Episode keys: episode_id, goal, screenshots_b64, actions, step_instructions
  - actions[i] keys: action_type, app_name, direction, text, x, y
  - step_instructions is a TOP-LEVEL parallel list (not nested in actions)
  - Click x/y are raw pixels (e.g. 313 on a 1080x2400 image)
  - len(screenshots_b64) == len(actions) or len(actions)+1 (final state)
  - When a terminal (post-action) screenshot is provided, we emit an additional
    synthetic "done" step so the model learns when to stop without throwing
    away the real final action.
"""

import argparse
import base64
import json
import os
import sys
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path

import itertools

from datasets import get_dataset_split_names, load_dataset
from PIL import Image
from tqdm import tqdm


HF_REPO = "smolagents/android-control"

TERMINAL_ACTION = {"action": "done"}


def encode_discrete_xy(x: float, y: float, grid: int) -> dict:
    """Quantize normalized [0,1] coords to discrete <loc_x_K>/<loc_y_K> tokens."""
    kx = max(0, min(grid - 1, round(x * grid)))
    ky = max(0, min(grid - 1, round(y * grid)))
    return {"x": f"<loc_x_{kx}>", "y": f"<loc_y_{ky}>"}


def map_action(raw_step: dict, norm_coords: dict | None) -> dict:
    """Convert one raw AndroidControl action into our canonical SFT schema."""
    raw_type = (raw_step.get("action_type") or "").lower().strip()

    if raw_type == "click":
        out = {"action": "tap"}
        if norm_coords:
            out.update(norm_coords)
        return out

    if raw_type == "input_text":
        return {"action": "type", "text": raw_step.get("text") or ""}

    if raw_type == "open_app":
        return {"action": "open_app", "app_name": raw_step.get("app_name") or ""}

    if raw_type == "navigate_back":
        return {"action": "navigate_back"}

    if raw_type == "navigate_home":
        return {"action": "navigate_home"}

    if raw_type == "scroll":
        return {"action": "scroll", "direction": raw_step.get("direction") or "down"}

    if raw_type == "wait":
        return {"action": "wait"}

    print(
        f"WARNING: unknown action_type '{raw_type}' in episode "
        f"{raw_step.get('_episode_id', '?')} step {raw_step.get('_step_index', '?')}",
        file=sys.stderr,
    )
    return {"action": raw_type}


def decode_and_save_b64(
    b64_str: str, images_dir: Path, episode_id: int, step_index: int
) -> tuple[str | None, tuple[int, int] | None]:
    """Decode b64 screenshot, save as PNG, return (relative_path, (w, h))."""
    try:
        raw = base64.b64decode(b64_str)
        img = Image.open(BytesIO(raw))
        size = img.size
    except Exception as e:
        print(
            f"WARNING: failed to decode/open screenshot for ep {episode_id} "
            f"step {step_index}: {e}",
            file=sys.stderr,
        )
        return None, None

    images_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{episode_id:05d}_{step_index:02d}.png"
    dest = images_dir / filename

    try:
        img.save(dest, "PNG")
    except Exception as e:
        print(
            f"WARNING: failed to save image for ep {episode_id} step {step_index}: {e}",
            file=sys.stderr,
        )
        return None, None

    return f"images/{filename}", size


def build_sample(
    episode_id: str,
    step_index: int,
    total_steps: int,
    img_rel_path: str | None,
    instruction: str,
    action_obj: dict,
    granularity: str,
) -> dict:
    """One JSONL row: image referenced by path (Unsloth loads at train time)."""
    user_content = []
    if img_rel_path is not None:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": instruction})

    assistant_content = [{"type": "text", "text": json.dumps(action_obj)}]
    row = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "episode_id": episode_id,
        "step_index": step_index,
        "total_steps": total_steps,
        "granularity": granularity,
    }
    if img_rel_path is not None:
        row["image"] = img_rel_path
    return row


def process_episode(
    episode: dict, output_dir: Path, coord_encoding: str = "float", grid_size: int = 1024
) -> list[dict]:
    """Return all SFT rows for one episode (2 per step: goal + step_instruction)."""
    images_dir = output_dir / "images"

    episode_id = str(episode["episode_id"])
    goal = episode.get("goal") or ""
    actions = episode.get("actions") or []
    screenshots = episode.get("screenshots_b64") or []
    step_instructions = episode.get("step_instructions") or []

    if not actions:
        return []

    rows: list[dict] = []
    n = len(actions)

    for i, raw_step in enumerate(actions):
        # Per-step screenshot (if available)
        b64 = screenshots[i] if i < len(screenshots) else None
        img_rel, size = (None, None)
        if b64:
            img_rel, size = decode_and_save_b64(
                b64, images_dir, int(episode_id), i
            )

        # Normalize click coordinates against this image's actual dims
        norm_coords = None
        raw_type = (raw_step.get("action_type") or "").lower().strip()
        if raw_type == "click" and size is not None:
            x = raw_step.get("x")
            y = raw_step.get("y")
            if x is not None and y is not None:
                w, h = size
                nx = max(0.0, min(1.0, x / w))
                ny = max(0.0, min(1.0, y / h))
                if coord_encoding == "discrete":
                    norm_coords = encode_discrete_xy(nx, ny, grid_size)
                else:
                    norm_coords = {
                        "x": round(nx, 6),
                        "y": round(ny, 6),
                    }

        # Tag for warning messages
        raw_step["_episode_id"] = episode_id
        raw_step["_step_index"] = i

        action_obj = map_action(raw_step, norm_coords)

        # Two granularities: goal-level and per-step instruction
        rows.append(
            build_sample(episode_id, i, n, img_rel, goal, action_obj, "goal")
        )

        step_instr = step_instructions[i] if i < len(step_instructions) else goal
        rows.append(
            build_sample(
                episode_id, i, n, img_rel, step_instr, action_obj, "step_instruction"
            )
        )

    # Synthetic terminal "done" step — only when a post-action screenshot exists
    # (otherwise we have no image to ground the "stop" decision in).
    if len(screenshots) > n:
        terminal_img_rel, _ = decode_and_save_b64(
            screenshots[n], images_dir, int(episode_id), n
        )
        if terminal_img_rel is not None:
            done_action = dict(TERMINAL_ACTION)
            rows.append(
                build_sample(
                    episode_id, n, n + 1, terminal_img_rel, goal, done_action, "goal"
                )
            )
            rows.append(
                build_sample(
                    episode_id, n, n + 1, terminal_img_rel, goal, done_action,
                    "step_instruction",
                )
            )

    return rows


# Worker globals — populated via Pool initializer to avoid pickling per-task.
_WORKER_DATASET = None
_WORKER_OUTPUT_DIR: Path | None = None
_WORKER_COORD_ENCODING: str = "float"
_WORKER_GRID_SIZE: int = 1024


def _init_worker(
    dataset, output_dir: Path, coord_encoding: str = "float", grid_size: int = 1024
) -> None:
    global _WORKER_DATASET, _WORKER_OUTPUT_DIR, _WORKER_COORD_ENCODING, _WORKER_GRID_SIZE
    _WORKER_DATASET = dataset
    _WORKER_OUTPUT_DIR = output_dir
    _WORKER_COORD_ENCODING = coord_encoding
    _WORKER_GRID_SIZE = grid_size


def _process_idx(idx: int) -> list[dict]:
    return process_episode(
        _WORKER_DATASET[idx],
        _WORKER_OUTPUT_DIR,
        coord_encoding=_WORKER_COORD_ENCODING,
        grid_size=_WORKER_GRID_SIZE,
    )


def write_jsonl(split_name: str, rows: list[dict], output_dir: Path) -> Path:
    jsonl_path = output_dir / f"{split_name}.jsonl"
    rows.sort(key=lambda r: (int(r["episode_id"]), r["step_index"], r["granularity"]))
    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  → {jsonl_path} ({len(rows)} rows)")
    return jsonl_path


def fetch_ood_splits_hf() -> dict:
    """Best-effort fetch of OOD split labels from a split-enriched HF mirror."""
    try:
        ds = load_dataset("reece124/android_control", split="test")
        out = {}
        for item in ds:
            eid = item.get("episode_id") or item.get("id")
            if eid is None:
                continue
            out[str(eid)] = {
                "task_unseen": bool(item.get("task_unseen", False)),
                "app_unseen": bool(item.get("app_unseen", False)),
                "category_unseen": bool(item.get("category_unseen", False)),
            }
        return out
    except Exception as e:
        print(f"INFO: HF OOD split fetch failed: {e}", file=sys.stderr)
        return {}


def fetch_ood_splits_gcs() -> dict:
    """Official OOD splits from public GCS bucket (requires gsutil)."""
    try:
        import subprocess

        result = subprocess.run(
            ["gsutil", "cat", "gs://gresearch/android_control/splits.json"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(
                f"INFO: gsutil exit {result.returncode}, skipping",
                file=sys.stderr,
            )
            return {}

        raw = json.loads(result.stdout)
        out: dict = {}
        for split_name, episode_ids in raw.items():
            for eid in episode_ids:
                out.setdefault(str(eid), {})[split_name] = True
        return out
    except FileNotFoundError:
        print("INFO: gsutil not installed, skipping OOD split fetch", file=sys.stderr)
    except Exception as e:
        print(f"INFO: GCS OOD split fetch failed: {e}", file=sys.stderr)
    return {}


def write_ood_splits(output_dir: Path) -> None:
    splits = fetch_ood_splits_gcs() or fetch_ood_splits_hf()
    if not splits:
        print("INFO: no OOD splits fetched", file=sys.stderr)
        return
    path = output_dir / "ood_splits.json"
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved OOD split metadata → {path} ({len(splits)} episodes)")


def main():
    parser = argparse.ArgumentParser(
        description="Download and format AndroidControl for Unsloth QLoRA SFT.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/androidcontrol"),
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Limit episodes per split (for smoke testing).",
    )
    parser.add_argument(
        "--fetch-ood-splits", action="store_true", default=False,
    )
    parser.add_argument(
        "--num-workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
        help="Worker processes for episode decoding. 0 disables multiprocessing "
             "(serial mode, used automatically for streaming).",
    )
    parser.add_argument(
        "--coord-encoding", choices=["float", "discrete"], default="float",
        help="float: emit normalized floats. discrete: emit <loc_x_K>/<loc_y_K> tokens.",
    )
    parser.add_argument(
        "--grid-size", type=int, default=1024,
        help="Grid resolution per axis when --coord-encoding=discrete.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AndroidControl Data Preparation — Step 1")
    print("=" * 60)

    try:
        split_names = get_dataset_split_names(HF_REPO)
    except Exception as e:
        print(f"FATAL: could not list splits for '{HF_REPO}': {e}", file=sys.stderr)
        sys.exit(1)

    use_streaming = args.max_episodes is not None
    mode = "streaming" if use_streaming else "full download"
    print(f"\nLoading {HF_REPO} ({mode}). Splits: {split_names}")

    split_results: dict[str, list[dict]] = {}

    for split_name in split_names:
        try:
            split = load_dataset(HF_REPO, split=split_name, streaming=use_streaming)
        except Exception as e:
            print(f"FATAL: could not load split '{split_name}': {e}", file=sys.stderr)
            sys.exit(1)

        rows: list[dict] = []

        if use_streaming or args.num_workers <= 0:
            # Serial path: streaming datasets aren't index-addressable, and the
            # user can opt out of multiprocessing with --num-workers 0.
            iterator = (
                itertools.islice(split, args.max_episodes) if use_streaming else split
            )
            mode_desc = (
                f"streaming up to {args.max_episodes}" if use_streaming
                else f"{len(split)} episodes (serial)"
            )
            print(f"\nSplit '{split_name}': {mode_desc}")
            for episode in tqdm(iterator, desc=f"  {split_name}", unit="ep"):
                rows.extend(
                    process_episode(
                        episode, output_dir,
                        coord_encoding=args.coord_encoding,
                        grid_size=args.grid_size,
                    )
                )
        else:
            n = len(split)
            print(f"\nSplit '{split_name}': {n} episodes "
                  f"({args.num_workers} workers)")
            with Pool(
                processes=args.num_workers,
                initializer=_init_worker,
                initargs=(split, output_dir, args.coord_encoding, args.grid_size),
            ) as pool:
                for ep_rows in tqdm(
                    pool.imap_unordered(_process_idx, range(n), chunksize=4),
                    total=n, desc=f"  {split_name}", unit="ep",
                ):
                    rows.extend(ep_rows)

        split_results[split_name] = rows

    print("\nWriting JSONL files:")
    for split_name, rows in split_results.items():
        write_jsonl(split_name, rows, output_dir)

    if args.fetch_ood_splits:
        print("\nFetching OOD split metadata...")
        write_ood_splits(output_dir)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total = sum(len(r) for r in split_results.values())
    print(f"  Total step-level samples: {total}")
    for split_name, rows in split_results.items():
        goals = sum(1 for r in rows if r["granularity"] == "goal")
        steps = sum(1 for r in rows if r["granularity"] == "step_instruction")
        print(f"  {split_name}: {len(rows)} rows ({goals} goal + {steps} step_instruction)")
    images_dir = output_dir / "images"
    if images_dir.exists():
        print(f"  Screenshots saved: {len(list(images_dir.glob('*.png')))} PNGs")
    print(f"\nOutput directory: {output_dir.resolve()}")
    print("Ready for Unsloth QLoRA SFT training on NVIDIA GPU.")


if __name__ == "__main__":
    main()
