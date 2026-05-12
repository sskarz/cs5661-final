"""Convert genesis trajectories → M3A SFT training rows.

Reads `data/pathZ/genesis/trajectories.jsonl` (one record per (app, goal)
rollout) and emits per-step training rows compatible with
`scripts/pathZ/train_smoke.py`.

Filtering:
  - drop steps with no parseable action (`action_output_json` is None)
  - truncate trajectories at the first `status:*` emission (don't train
    the student to spam status; the harness already suppresses these
    in r43+ but we don't want them in train data)
  - drop trajectories that yield 0 useful steps after filtering
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _classify_trajectory(rec: dict) -> tuple[str, int]:
    """Bucket the trajectory.

    Returns (bucket, real_n) where real_n is the truncated step count
    (steps 0..first_status-1 if status emitted, else len(steps)).

    Buckets:
      success         — status emitted, real_n >= 2, no `no visible change`
      recovery        — status emitted, real_n >= 2, has at least one no-op
      premature       — status emitted at step <= 1
      no_status_long  — never emitted status, hit max_steps, no repeated actions
      loop            — never emitted status, hit max_steps, >=3 consecutive
                        identical actions
      other           — anything else (parse failures, very short with no status)
    """
    steps = rec.get("steps", [])
    n = len(steps)
    first_status = None
    summaries = []
    actions = []
    for s in steps:
        a = s.get("action_output_json") or {}
        sumt = (s.get("summary") or "").lower()
        summaries.append(sumt)
        actions.append((a.get("action_type"), a.get("index"),
                        a.get("app_name"), a.get("direction"),
                        (a.get("text") or "")[:20]))
        if a.get("action_type") == "status" and first_status is None:
            first_status = s["step"]

    real_n = first_status if first_status is not None else n
    if first_status is not None and first_status <= 1:
        return "premature", real_n
    if first_status is None and n >= 12:
        max_run = 1
        cur_run = 1
        for i in range(1, n):
            if actions[i] == actions[i - 1]:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 1
        return ("loop" if max_run >= 3 else "no_status_long"), real_n
    if first_status is not None and real_n >= 2:
        n_noop = sum(1 for s in summaries[:real_n] if "no visible change" in s)
        return ("success" if n_noop == 0 else "recovery"), real_n
    return "other", real_n


# Buckets to include. r50: also drop no_status_long. Investigation of
# r48 AW-116 CameraTakePhoto failure showed the Camera no_status_long
# trajectory ("Take a photo of a 'Blue Vase' by tapping the Shutter
# button") trained the model to click shutter 12× without ever emitting
# status — directly poisoning AW eval behavior.
KEEP_BUCKETS = {"success", "recovery"}


def _convert_record(rec: dict, bucket: str) -> list[dict]:
    """One trajectory → list of training rows.

    Keeps through the FIRST status:complete emission (so the model learns
    when to stop). Drops any subsequent status spam. Drops parse-failure
    steps.
    """
    rows = []
    seen_status = False
    for s in rec.get("steps", []):
        prompt = s.get("action_prompt")
        emission = s.get("action_output")
        action = s.get("action_output_json")
        if not prompt or not emission:
            continue
        at = (action or {}).get("action_type")
        # r48: keep the first status emission as the trajectory's end-marker,
        # then stop. Drops trailing status-spam from teacher loops.
        if at == "status":
            if seen_status:
                break
            seen_status = True
        if action is None:
            continue
        row = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": emission}
                ]},
            ],
            "_genesis_app": rec.get("app"),
            "_genesis_goal": rec.get("goal"),
            "_genesis_step": s.get("step"),
            "_genesis_bucket": bucket,
            "gt_m3a": action,
        }
        # r53: pass through screenshot path if present (vision training)
        if s.get("image"):
            row["image"] = s["image"]
        rows.append(row)
        if seen_status:
            break  # status was the last useful step
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/pathZ/genesis/trajectories.jsonl")
    ap.add_argument("--out", default="data/pathZ/genesis/train.jsonl")
    ap.add_argument("--min-steps", type=int, default=2,
                    help="drop trajectories with fewer than N useful steps")
    ap.add_argument("--image-root", type=Path, default=None,
                    help="Absolute path to the screenshots directory. "
                         "When set, all rows with an `image` field get "
                         "`_image_root` so train_smoke.py can resolve "
                         "relative PNG paths. Auto-detected from --src "
                         "if not provided (expects <src_parent>/screenshots).")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Resolve image root for screenshot-backed rows
    image_root = args.image_root
    if image_root is None and src.parent.name == "genesis":
        # Default: screenshots live next to trajectories.jsonl
        image_root = src.parent / "screenshots"
    if image_root is not None:
        image_root = image_root.resolve()

    from collections import Counter
    n_traj = n_kept_traj = n_rows = 0
    by_app: dict[str, int] = {}
    bucket_counts = Counter()
    bucket_rows = Counter()
    with open(src) as fin, open(out, "w") as fout:
        for line in fin:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n_traj += 1
            bucket, _ = _classify_trajectory(rec)
            bucket_counts[bucket] += 1
            if bucket not in KEEP_BUCKETS:
                continue
            rows = _convert_record(rec, bucket)
            if len(rows) < args.min_steps:
                continue
            n_kept_traj += 1
            bucket_rows[bucket] += len(rows)
            by_app[rec.get("app", "?")] = by_app.get(rec.get("app", "?"), 0) + len(rows)
            for r in rows:
                # r53: stamp image root so train_smoke.py can resolve PNGs
                if image_root is not None and r.get("image"):
                    r["_image_root"] = str(image_root)
                fout.write(json.dumps(r) + "\n")
                n_rows += 1

    print(f"[genesis_to_train] trajectories: kept {n_kept_traj}/{n_traj}")
    print(f"[genesis_to_train] training rows: {n_rows}")
    print("[genesis_to_train] bucket distribution (all trajectories):")
    for b, c in bucket_counts.most_common():
        kept = "KEEP" if b in KEEP_BUCKETS else "DROP"
        rows_b = bucket_rows.get(b, 0)
        print(f"  {b:18s}  n_traj={c:3d}  rows={rows_b:4d}  [{kept}]")
    print("[genesis_to_train] per-app row counts (kept only):")
    for app, c in sorted(by_app.items(), key=lambda kv: -kv[1]):
        print(f"  {app:32s}  {c:4d}")
    print(f"METRIC genesis_train_rows={n_rows}")
    print(f"METRIC genesis_train_trajectories={n_kept_traj}")


if __name__ == "__main__":
    main()
