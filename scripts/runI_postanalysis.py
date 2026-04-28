#!/usr/bin/env python3
"""Post-training analysis for Run I (Path W SFT).

Reads every outputs/eval/runI_ckpt*.json sweep result, plus the Path W
baseline (outputs/eval/native_baseline.json), and produces:

1. A per-checkpoint summary table (full_match, parse_rate, oracle ceiling,
   per-action-type acc deltas vs baseline).
2. Best-checkpoint failure analysis:
   - Action-type confusion matrix
   - Of wrong tap predictions, how many picked a "near" element (within
     tap_radius of GT) — tells us "almost right" vs "totally lost"
   - Top-3 over-predicted action types per GT type
3. Writes:
   - WAKE_UP_SUMMARY.md (top-level repo root)
   - Appends a §30 section to TRAINING_LOG.md
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

TAP_RADIUS = 0.14
BASELINE_NATIVE = "outputs/eval/native_baseline.json"
BASELINE_HF = 0.288  # coord-regression HF mirror baseline
RUNH_BEST = 0.275  # Run H ckpt-1000


def load_metrics(path: str) -> dict:
    return json.load(open(path))["metrics"]


def near_miss_rate(all_preds: list[dict]) -> tuple[int, int]:
    """Of tap GT rows where pred was wrong, how many picked an element whose
    bbox center is within tap_radius of GT (x,y)?"""
    near = 0
    far = 0
    for p in all_preds:
        gt = p.get("gt") or {}
        gt_xy = p.get("gt_xy")
        if gt_xy is None or (gt.get("action") or "").lower() != "tap":
            continue
        if p.get("ok"):
            continue  # only count wrong predictions
        pred = p.get("pred") or {}
        rx = pred.get("_resolved_x")
        ry = pred.get("_resolved_y")
        if rx is None or ry is None:
            far += 1
            continue
        dx, dy = rx - gt_xy[0], ry - gt_xy[1]
        if (dx * dx + dy * dy) <= (TAP_RADIUS * TAP_RADIUS):
            near += 1
        else:
            far += 1
    return near, far


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-glob", default="outputs/eval/runI_ckpt*.json")
    ap.add_argument("--baseline", default=BASELINE_NATIVE)
    ap.add_argument("--out-summary", type=Path,
                    default=Path("WAKE_UP_SUMMARY.md"))
    ap.add_argument("--training-log", type=Path,
                    default=Path("TRAINING_LOG.md"))
    args = ap.parse_args()

    # --- Load eval results ---
    paths = sorted(glob.glob(args.eval_glob),
                   key=lambda x: int(x.split("ckpt")[1].split(".")[0]))
    if not paths:
        print(f"[postanalysis] no Run I evals found at {args.eval_glob}")
        return
    print(f"[postanalysis] loaded {len(paths)} checkpoint evals")

    rows = []
    base_m = load_metrics(args.baseline) if Path(args.baseline).exists() else None
    base_per_type = base_m["per_type"] if base_m else {}

    for p in paths:
        m = load_metrics(p)
        n = int(p.split("ckpt")[1].split(".")[0])
        rows.append((n, p, m))

    rows.sort(key=lambda r: r[0])

    # --- Find best checkpoint ---
    best_idx = max(range(len(rows)), key=lambda i: rows[i][2]["full_match"])
    best_n, best_path, best_m = rows[best_idx]
    print(f"[postanalysis] best ckpt = {best_n} (full_match={best_m['full_match']:.3f})")

    # --- Build summary table ---
    table_lines = [
        "| step | full_match | parse | oracle | tap acc | scroll | type | open_app | nav_back |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    if base_m is not None:
        pt = base_m["per_type"]
        get = lambda k: pt.get(k, {}).get("accuracy", 0.0)
        table_lines.append(
            f"| **baseline** | **{base_m['full_match']:.3f}** | "
            f"{base_m['parse_rate']:.3f} | {base_m.get('tap_oracle_reachability', 0):.3f} | "
            f"{get('tap'):.3f} | {get('scroll'):.3f} | {get('type'):.3f} | "
            f"{get('open_app'):.3f} | {get('navigate_back'):.3f} |"
        )
    for n, _, m in rows:
        pt = m["per_type"]
        get = lambda k: pt.get(k, {}).get("accuracy", 0.0)
        bold = "**" if n == best_n else ""
        table_lines.append(
            f"| {bold}{n}{bold} | {bold}{m['full_match']:.3f}{bold} | "
            f"{m['parse_rate']:.3f} | {m.get('tap_oracle_reachability', 0):.3f} | "
            f"{get('tap'):.3f} | {get('scroll'):.3f} | {get('type'):.3f} | "
            f"{get('open_app'):.3f} | {get('navigate_back'):.3f} |"
        )

    # --- Failure analysis on best checkpoint ---
    best_full = json.load(open(best_path))
    all_preds = best_full.get("all_predictions") or []

    confusion = best_m.get("confusion") or {}
    near, far = near_miss_rate(all_preds)
    n_tap_wrong = near + far
    near_pct = near / max(n_tap_wrong, 1)

    fail_lines = [
        f"**Best checkpoint: ckpt-{best_n}, full_match = {best_m['full_match']:.3f}**",
        "",
        "Confusion matrix (rows = GT action, cols = predicted):",
        "",
    ]
    if confusion:
        all_preds_types = sorted({pt for v in confusion.values() for pt in v})
        header = "| GT \\ pred | " + " | ".join(all_preds_types) + " |"
        fail_lines.append(header)
        fail_lines.append("|---" * (len(all_preds_types) + 1) + "|")
        for gt_t in sorted(confusion):
            row = f"| **{gt_t}** | "
            row += " | ".join(str(confusion[gt_t].get(pt, 0)) for pt in all_preds_types)
            row += " |"
            fail_lines.append(row)
        fail_lines.append("")

    fail_lines.extend([
        f"Of wrong tap predictions: **{near}/{n_tap_wrong} ({near_pct:.1%}) were 'near misses'** "
        f"(picked a different element whose bbox center is within {TAP_RADIUS} of GT). The rest "
        f"({far}) were structurally wrong picks.",
        "",
    ])

    # Compare per-type vs baseline
    if base_m:
        delta_lines = ["Per-action-type lift over zero-shot baseline (best ckpt):", ""]
        delta_lines.append("| action | n | baseline acc | best ckpt acc | Δ |")
        delta_lines.append("|---|---|---|---|---|")
        for k, v in best_m["per_type"].items():
            base_acc = base_per_type.get(k, {}).get("accuracy", 0.0)
            d = v["accuracy"] - base_acc
            arrow = "↑" if d >= 0 else "↓"
            delta_lines.append(
                f"| {k} | {v['n']} | {base_acc:.3f} | {v['accuracy']:.3f} | {arrow} {d:+.3f} |"
            )
        fail_lines.extend(delta_lines)
        fail_lines.append("")

    # --- Build wake-up summary ---
    headline_full = best_m["full_match"]
    summary = [
        "# Wake-up summary — Run I (Path W SFT) overnight",
        "",
        "## Headline",
        "",
        f"- **Best checkpoint: `ckpt-{best_n}` at full_match = {headline_full:.3f}**",
        f"- Path W zero-shot baseline: {base_m['full_match']:.3f} (no training)" if base_m else "",
        f"- Coord baseline (HF, all 8 prior LoRA runs failed to clear): {BASELINE_HF}",
        f"- Run H best (LoRA SFT, projector unlocked, coord regression): {RUNH_BEST}",
        "",
        f"- Lift over coord baseline: **{headline_full - BASELINE_HF:+.3f}** "
        f"({(headline_full - BASELINE_HF) / BASELINE_HF:+.1%} relative)",
        f"- Lift over Run H best: **{headline_full - RUNH_BEST:+.3f}**",
        "",
        "## Decision tree",
        "",
        f"- If best ckpt > {base_m['full_match']:.3f} (baseline) → SFT is helping; "
        "ship the best checkpoint, optionally try DPO."
        if base_m else "",
        f"- If best ckpt < baseline → SFT is hurting; ship the zero-shot baseline.",
        f"- If best ckpt is at oracle ceiling ({best_m.get('tap_oracle_reachability', 0):.3f}): "
        "we've saturated the projection-center scoring; need element-level metric to gain more.",
        "",
        "## Per-checkpoint sweep",
        "",
    ] + table_lines + [
        "",
        "## Failure analysis",
        "",
    ] + fail_lines + [
        "## Files",
        "",
        f"- Run I checkpoints: `outputs/gemma4-e2b-pathW-lora-runI/`",
        f"- Eval JSONs: `outputs/eval/runI_ckpt*.json`",
        f"- Best-ckpt full predictions (with all_predictions): `{best_path}`",
        f"- Path W baseline: `outputs/eval/native_baseline.json`",
        "",
        "## Suggested next step",
        "",
    ]
    if base_m and headline_full > base_m["full_match"]:
        summary.append("- Path W SFT cleared zero-shot. Pick best checkpoint, run on full test set "
                       "(8,217 rows, ~30 min), publish the number. Consider DPO if time permits.")
    elif base_m and headline_full > BASELINE_HF:
        summary.append("- Path W SFT cleared the coord baseline but not zero-shot. Either ship zero-shot "
                       "(simpler, no overfitting risk) or investigate response-only mask integrity.")
    else:
        summary.append("- Path W SFT did not clear baseline. Investigate response-only loss config "
                       "and potentially re-train with adjusted hyperparameters. See failure analysis above.")

    args.out_summary.write_text("\n".join(summary) + "\n")
    print(f"[postanalysis] wrote {args.out_summary}")

    # --- Append to training log ---
    log_section = [
        "",
        "## 30. Run I — Path W (a11y-native) 1-epoch SFT, results",
        "",
        f"Best checkpoint: **`ckpt-{best_n}` at full_match = {headline_full:.3f}** "
        f"(Path W baseline {base_m['full_match']:.3f}, "
        f"Run H best {RUNH_BEST}, "
        f"coord baseline {BASELINE_HF}).",
        "",
        "### Per-checkpoint sweep",
        "",
    ] + table_lines + [
        "",
        "### Best-checkpoint failure analysis",
        "",
    ] + fail_lines + [""]

    with open(args.training_log, "a") as f:
        f.write("\n".join(log_section) + "\n")
    print(f"[postanalysis] appended §30 to {args.training_log}")


if __name__ == "__main__":
    main()
