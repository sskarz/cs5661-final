#!/usr/bin/env python3
"""
Step 1 — Data Preparation: Download and format AndroidControl for SFT training.

Downloads the official AndroidControl dataset from HuggingFace, explodes episodes
into step-level samples, normalizes coordinates per-image, writes PNG screenshots
to disk, and emits JSONL files ready for Unsloth QLoRA fine-tuning.

Usage (local data prep on Mac):
    uv run python scripts/prepare_androidcontrol.py --output-dir data/androidcontrol

Usage (full download + OOD splits on NVIDIA training box):
    uv run --extra train python scripts/prepare_androidcontrol.py \\\\
        --output-dir data/androidcontrol \\\\
        --fetch-ood-splits

Key design decisions:
  - HF repo = "smolagents/android-control" (exact, no fallback) — ~15,283 episodes
  - Each step = one row (not full episode) — avoids blowing context windows
  - Two samples per step: "goal" mode (episode-level) + "step_instruction" mode
  - Coordinates normalized per-image using actual PNG dimensions (not hardcoded)
  - Official train/test split preserved — no invented split ratios
  - OOD split metadata (task_unseen, app_unseen, etc.) fetched from s3 + HF join
"""

import argparse
import base64
import json
import os
import sys
from io import BytesIO
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration — pin exact HF repo, no fallbacks (reproducibility)
# ---------------------------------------------------------------------------
HF_REPO = "smolagents/android-control"

# Default split sizes (from HF card): ~12,232 train / ~3,051 test
# We preserve the official split — do not re-split.

# Canonical action vocabulary mapping (AndroidControl raw → our SFT schema)
ACTION_SCHEMA = {
    "click": {"action": "tap", "x": "<norm_x>", "y": "<norm_y>"},
    "input_text": {"action": "type", "text": "<text>"},
    "open_app": {"action": "open_app", "app_name": "<app_name>"},
    "navigate_back": {"action": "navigate_back"},
    "navigate_home": {"action": "navigate_home"},
    "scroll": {
        "action": "scroll",
        "direction": "<direction>",  # "up" | "down" | "left" | "right"
    },
    "wait": {"action": "wait"},
}

# Terminal action — final step signals completion
TERMINAL_ACTION = {"action": "done"}


# ---------------------------------------------------------------------------
# Coordinate normalization — per-image, not hardcoded resolution
# ---------------------------------------------------------------------------

def normalize_coordinates(raw_x: float, raw_y: float, img_width: int, img_height: int) -> dict:
    """Normalize raw pixel coordinates to [0, 1] relative to this specific image."""
    return {
        "x": round(raw_x / img_width, 6),
        "y": round(raw_y / img_height, 6),
    }


# ---------------------------------------------------------------------------
# Action mapping — raw AndroidControl actions → canonical SFT schema
# ---------------------------------------------------------------------------

def map_action(raw_step: dict, norm_coords: dict | None = None) -> dict:
    """Convert a raw AndroidControl step into our canonical action schema."""
    raw_type = raw_step.get("type", "").lower().strip()

    if raw_type == "click":
        result = dict(ACTION_SCHEMA["click"])
        result.update(norm_coords) if norm_coords else None
        return result

    elif raw_type == "input_text":
        result = dict(ACTION_SCHEMA["input_text"])
        result["text"] = raw_step.get("text", "")
        return result

    elif raw_type == "open_app":
        result = dict(ACTION_SCHEMA["open_app"])
        result["app_name"] = raw_step.get("app", raw_step.get("app_name", ""))
        return result

    elif raw_type == "navigate_back":
        return dict(ACTION_SCHEMA["navigate_back"])

    elif raw_type == "navigate_home":
        return dict(ACTION_SCHEMA["navigate_home"])

    elif raw_type == "scroll":
        result = dict(ACTION_SCHEMA["scroll"])
        result["direction"] = raw_step.get("direction", "down")
        return result

    elif raw_type == "wait":
        return dict(ACTION_SCHEMA["wait"])

    else:
        # Unknown action — emit as-is, log a warning via stderr
        print(
            f"WARNING: unknown action type '{raw_type}' in step "
            f"{raw_step.get('step_index', '?')} of episode "
            f"{raw_step.get('episode_id', '?')}",
            file=sys.stderr,
        )
        return {"action": raw_type}


# ---------------------------------------------------------------------------
# Image decoding — write PNG to disk, return relative path
# ---------------------------------------------------------------------------

def decode_and_save_b64(
    b64_str: str, output_dir: Path, episode_id: int, step_index: int
) -> str | None:
    """Decode base64 screenshot, save as PNG, return relative path."""
    try:
        raw = base64.b64decode(b64_str)
    except Exception as e:
        print(
            f"WARNING: failed to decode screenshot for episode {episode_id} "
            f"step {step_index}: {e}",
            file=sys.stderr,
        )
        return None

    try:
        img = Image.open(BytesIO(raw))
    except Exception as e:
        print(
            f"WARNING: failed to open image for episode {episode_id} "
            f"step {step_index}: {e}",
            file=sys.stderr,
        )
        return None

    # Get actual dimensions for coordinate normalization (per-image!)
    img_width, img_height = img.size

    # Save PNG — relative path inside output_dir/images/
    rel_subdir = output_dir / "images"
    rel_subdir.mkdir(parents=True, exist_ok=True)

    filename = f"{episode_id:05d}_{step_index:02d}.png"
    dest = rel_subdir / filename

    try:
        img.save(dest, "PNG")
    except Exception as e:
        print(
            f"WARNING: failed to save image for episode {episode_id} "
            f"step {step_index}: {e}",
            file=sys.stderr,
        )
        return None

    # Return relative path from output_dir (what JSONL references)
    try:
        return str(dest.relative_to(output_dir))
    except ValueError:
        # Fallback if relative_to fails (shouldn't happen, but be safe)
        return f"images/{filename}"


# ---------------------------------------------------------------------------
# Build SFT message — one step → one training sample (two granularities)
# ---------------------------------------------------------------------------

def build_messages(
    episode: dict, step_index: int, img_path: str | None, norm_coords: dict | None
) -> list[dict]:
    """Build a single SFT message pair (user + assistant) for one step."""

    raw_step = episode["actions"][step_index]
    screenshots_b64 = episode.get("screenshots_b64", [])

    # Build user message: image + text instruction
    user_content = []

    if img_path is not None and os.path.isfile(
        str(Path(output_dir) / img_path) if isinstance(output_dir, Path) else img_path
    ):
        # For Unsloth SFTTrainer: embed image as base64 data URI in JSONL
        # (Unsloth's collator handles this)
        try:
            full_path = Path(output_dir) / img_path if isinstance(output_dir, Path) else Path(img_path)
            with open(full_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("ascii")
            user_content.append({
                "type": "image",
                "data": img_b64,
            })
        except Exception:
            pass  # No image — still produce text-only sample

    # Build assistant message: mapped action JSON string
    raw_action = episode["actions"][step_index]
    final_step = step_index == len(episode["actions"]) - 1

    if final_step:
        action_obj = dict(TERMINAL_ACTION)
    else:
        action_obj = map_action(raw_step, norm_coords)

    # Include any additional raw fields (e.g., text for input_text)
    if "text" in raw_step and raw_step.get("type", "").lower() == "input_text":
        action_obj["text"] = raw_step.get("text", "")

    return user_content, json.dumps(action_obj)


# ---------------------------------------------------------------------------
# Build OOD split metadata — join s3 splits + HF data
# ---------------------------------------------------------------------------

def fetch_ood_splits_hf() -> dict:
    """Fetch OOD split labels from a secondary HF dataset that has them."""
    try:
        # Try the split-enriched variant that includes OOD labels
        ds = load_dataset("reece124/android_control", split="test")
        splits = {}
        for item in ds:
            eid = item.get("episode_id") or item.get("id", "")
            splits[str(eid)] = {
                "task_unseen": item.get("task_unseen", False),
                "app_unseen": item.get("app_unseen", False),
                "category_unseen": item.get("category_unseen", False),
            }
        return splits
    except Exception as e:
        print(
            f"INFO: could not fetch OOD splits from HF "
            f"(reece124/android_control): {e}",
            file=sys.stderr,
        )
        return {}


def fetch_ood_splits_gcs() -> dict:
    """Fetch OOD split labels from Google Cloud Storage (official source)."""
    try:
        import subprocess

        result = subprocess.run(
            ["gsutil", "cat", "gs://gresearch/android_control/splits.json"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(
                f"INFO: gsutil failed ({result.returncode}), "
                "skipping OOD split fetch",
                file=sys.stderr,
            )
            return {}

        raw = json.loads(result.stdout)
        splits = {}
        for split_name, episodes in raw.items():
            # split_name like "task_unseen_test" → extract bucket + split_type
            parts = split_name.split("_")
            split_type = parts[0] if len(parts) > 1 else "unknown"

            for eid in episodes:
                split_key = str(eid)
                if split_key not in splits:
                    splits[split_key] = {}
                # Each split is mutually exclusive — mark which bucket this episode belongs to
                splits[split_key][f"{split_type}_test"] = True

        return splits
    except FileNotFoundError:
        print(
            "INFO: gsutil not found, skipping OOD split fetch. "
            "Install with 'gcloud auth application-default login && gcloud components install gsutil'",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"INFO: could not fetch OOD splits from GCS: {e}", file=sys.stderr)

    return {}


def build_ood_metadata(train_jsonl: Path, test_jsonl: Path) -> dict:
    """Build OOD split metadata by joining s3 + HF data."""
    # Try s3 first (official source, has full split labels)
    splits = fetch_ood_splits_gcs()

    # If s3 unavailable, try HF split-enriched dataset
    if not splits:
        splits = fetch_ood_splits_hf()

    # Build a lookup from episode_id → split labels
    split_lookup = {}
    for split_type, episodes in splits.items():
        if isinstance(episodes, dict):
            # s3 format: split_type → {episode_id: True/False}
            for eid, present in episodes.items():
                split_lookup[str(eid)] = split_lookup.get(str(eid), {})
                split_lookup[str(eid)][split_type] = present
        elif isinstance(episodes, list):
            # HF format: split_type → [episode_ids]
            for eid in episodes:
                split_lookup[str(eid)] = split_lookup.get(str(eid), {})
                split_lookup[str(eid)][split_type] = True

    # Save OOD metadata as JSON (not JSONL — one entry per split, not per row)
    ood_path = output_dir / "ood_splits.json"
    try:
        with open(ood_path, "w") as f:
            json.dump(split_lookup, f, indent=2)
        print(f"Saved OOD split metadata → {ood_path}")
    except Exception as e:
        print(f"WARNING: could not save OOD splits: {e}", file=sys.stderr)

    return split_lookup


# ---------------------------------------------------------------------------
# Main — download, explode, normalize, write JSONL + PNGs
# ---------------------------------------------------------------------------

def process_episode(episode: dict, output_dir: Path) -> list[dict]:
    """Process one episode into step-level SFT samples."""
    episodes = [episode]  # For iteration consistency
    return process_episodes(episodes, output_dir)


def process_episodes(
    episodes: list[dict], output_dir: Path, max_episodes: int | None = None
) -> dict[str, list[dict]]:
    """Process a batch of episodes into SFT training data.

    Returns dict split_name → list of JSONL rows.
    Each step produces TWO samples: "goal" mode + "step_instruction" mode.
    """

    # Collect split names from the dataset (official splits)
    split_names = list(set(episode.get("split", "train") for episode in episodes))

    split_data = {name: [] for name in split_names}
    total_steps = 0
    skipped = 0

    for episode in tqdm(episodes, desc="Processing episodes", unit="ep"):
        if max_episodes and split_data.get(episode.get("split", "train"), []) is not None:
            # Count only episodes from the current split being processed
            pass

        episode_id = str(episode.get("episode_id", f"ep_{total_steps}"))
        actions = episode.get("actions", [])
        screenshots_b64 = episode.get("screenshots_b64", [])

        if not actions:
            skipped += 1
            continue

        goal = episode.get("goal", "")
        split_name = episode.get("split", "train")

        for step_index, raw_step in enumerate(actions):
            # Decode screenshot (base64 → PNG on disk)
            b64_str = screenshots_b64[step_index] if step_index < len(screenshots_b64) else None

            img_path = None
            norm_coords = None

            if b64_str:
                img_path = decode_and_save_b64(
                    b64_str, output_dir, int(episode_id), step_index
                )

            # Get actual image dimensions for normalization (per-image!)
            if img_path is not None:
                try:
                    full_img = output_dir / img_path if isinstance(output_dir, Path) else Path(img_path)
                    with Image.open(full_img) as img:
                        w, h = img.size
                except Exception:
                    # If we can't open the saved image, skip normalization
                    w, h = None, None

                raw_x = raw_step.get("x", 0.5)
                raw_y = raw_step.get("y", 0.5)

                if w and h:
                    norm_coords = normalize_coordinates(raw_x, raw_y, w, h)

            # Build user message content (image + text)
            user_content = []

            if img_path is not None:
                try:
                    full_img = output_dir / img_path if isinstance(output_dir, Path) else Path(img_path)
                    with open(full_img, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode("ascii")
                    user_content.append({
                        "type": "image",
                        "data": img_b64,
                    })
                except Exception:
                    pass

            # Build assistant message (mapped action)
            final_step = step_index == len(actions) - 1

            if final_step:
                action_obj = dict(TERMINAL_ACTION)
            else:
                raw_type = raw_step.get("type", "").lower().strip()

                if raw_type == "click":
                    action_obj = dict(ACTION_SCHEMA["click"])
                    if norm_coords:
                        action_obj.update(norm_coords)

                elif raw_type == "input_text":
                    action_obj = dict(ACTION_SCHEMA["input_text"])
                    action_obj["text"] = raw_step.get("text", "")

                elif raw_type == "open_app":
                    action_obj = dict(ACTION_SCHEMA["open_app"])
                    action_obj["app_name"] = raw_step.get(
                        "app", raw_step.get("app_name", "")
                    )

                elif raw_type == "navigate_back":
                    action_obj = dict(ACTION_SCHEMA["navigate_back"])

                elif raw_type == "navigate_home":
                    action_obj = dict(ACTION_SCHEMA["navigate_home"])

                elif raw_type == "scroll":
                    action_obj = dict(ACTION_SCHEMA["scroll"])
                    action_obj["direction"] = raw_step.get("direction", "down")

                elif raw_type == "wait":
                    action_obj = dict(ACTION_SCHEMA["wait"])

                else:
                    print(
                        f"WARNING: unknown action type '{raw_type}' in step "
                        f"{step_index} of episode {episode_id}",
                        file=sys.stderr,
                    )
                    action_obj = {"action": raw_type}

                # Preserve additional fields from raw step
                if "text" in raw_step and raw_type == "input_text":
                    action_obj["text"] = raw_step.get("text", "")

            actions_json = json.dumps(action_obj)

            # Build metadata for this step
            split_data.setdefault(split_name, [])

            # Sample 1: "goal" mode — full episode goal (standard eval protocol)
            step1 = {
                "messages": [
                    {"role": "user", "content": user_content + [{"type": "text", "text": goal}]}
                    if user_content else [{"type": "text", "text": goal}],
                    {"role": "assistant", "content": actions_json},
                ],
                "episode_id": episode_id,
                "step_index": step_index,
                "total_steps": len(actions),
                "granularity": "goal",  # Full episode goal
            }

            split_data[split_name].append(step1)

            # Sample 2: "step_instruction" mode — per-step instructions (low-level eval)
            step_instr = raw_step.get("step_instruction", goal)

            if user_content:
                step2_messages = [{"type": "text", "text": step_instr}]
                # Insert image at the right position (after text, before assistant)
                final_user = user_content + step2_messages
            else:
                final_user = [{"type": "text", "text": step_instr}]

            step2 = {
                "messages": [
                    {"role": "user", "content": final_user},
                    {"role": "assistant", "content": actions_json},
                ],
                "episode_id": episode_id,
                "step_index": step_index,
                "total_steps": len(actions),
                "granularity": "step_instruction",  # Per-step instructions
            }

            split_data[split_name].append(step2)
            total_steps += 1

    if skipped:
        print(f"Skipped {skipped} episodes with no actions", file=sys.stderr)

    print(f"Total step-level samples: {total_steps}")
    return split_data


def write_jsonl(split_name: str, rows: list[dict], output_dir: Path):
    """Write a split's data as JSONL."""
    jsonl_path = output_dir / f"{split_name}.jsonl"

    # Sort by episode_id, then step_index for reproducibility
    rows.sort(key=lambda r: (r.get("episode_id", "zzz"), r.get("step_index", 0)))

    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  → {jsonl_path} ({len(rows)} rows)")


def main():
    global output_dir

    parser = argparse.ArgumentParser(
        description="Download and format AndroidControl for SFT training.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/androidcontrol"),
        help="Output directory for JSONL + PNG images (default: data/androidcontrol)",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Process only N episodes (for smoke testing on Mac)",
    )
    parser.add_argument(
        "--fetch-ood-splits", action="store_true", default=False,
        help="Fetch OOD split metadata from s3 + HF (NVIDIA training box only)",
    )

    args = parser.parse_args()
    output_dir = args.output_dir

    print("=" * 60)
    print("AndroidControl Data Preparation — Step 1")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Download from HuggingFace (exact repo, no fallback)
    # ------------------------------------------------------------------
    print(f"\nDownloading {HF_REPO} from HuggingFace...")

    try:
        ds = load_dataset(HF_REPO)
    except Exception as e:
        print(f"FATAL: could not load '{HF_REPO}': {e}", file=sys.stderr)
        print(
            "Verify the repo exists at https://huggingface.co/datasets/" + HF_REPO,
            file=sys.stderr,
        )
        sys.exit(1)

    # Collect all episodes across splits into a single list, tagged by split
    all_episodes = []

    for split_name in ds.keys():
        split_data = ds[split_name]
        print(f"  Split '{split_name}': {len(split_data)} episodes")

        for episode in split_data:
            # Ensure split field is set (some HF datasets don't include it)
            episode["split"] = split_name
            all_episodes.append(episode)

    print(f"Total episodes: {len(all_episodes)}")

    # ------------------------------------------------------------------
    # Process episodes → step-level SFT samples + PNGs + JSONL
    # ------------------------------------------------------------------
    print(f"\nProcessing → {output_dir}")

    split_results = process_episodes(all_episodes, output_dir, args.max_episodes)

    # ------------------------------------------------------------------
    # Write JSONL files (one per split, preserving official split)
    # ------------------------------------------------------------------
    print("\nWriting JSONL files:")

    for split_name, rows in split_results.items():
        write_jsonl(split_name, rows, output_dir)

    # ------------------------------------------------------------------
    # OOD split metadata (NVIDIA training box only, not local Mac)
    # ------------------------------------------------------------------
    if args.fetch_ood_splits:
        print("\nFetching OOD split metadata...")
        build_ood_metadata(
            output_dir / "train.jsonl", output_dir / "test.jsonl"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    total_rows = sum(len(rows) for rows in split_results.values())
    print(f"  Total step-level samples: {total_rows}")

    for split_name, rows in split_results.items():
        goal_count = sum(1 for r in rows if r.get("granularity") == "goal")
        step_count = sum(1 for r in rows if r.get("granularity") == "step_instruction")
        print(f"  {split_name}: {len(rows)} rows ({goal_count} goal + {step_count} step_instruction)")

    images_dir = output_dir / "images"
    if images_dir.exists():
        png_count = len(list(images_dir.glob("*.png")))
        print(f"  Screenshots saved: {png_count} PNGs")

    print(f"\nOutput directory: {output_dir.absolute()}")
    print("Ready for Unsloth QLoRA SFT training on NVIDIA GPU.")


if __name__ == "__main__":
    main()
