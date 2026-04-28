#!/usr/bin/env python3
"""Aggregate Run I + Run J results into FINAL_SUMMARY.md."""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rescore_native_element import element_match  # noqa: E402

FILES = {
    "runI_test_sweep": "outputs/eval/runI_ckpt*.json",
    "runI_val_sweep": "outputs/eval/runI_val_ckpt*.json",
    "runJ_val_sweep": "outputs/eval/runJ_val_ckpt*.json",
    "runI_fulltest": "outputs/eval/runI_ckpt*_fulltest.json",
    "runJ_fulltest": "outputs/eval/runJ_ckpt*_fulltest.json",
    "baseline_fulltest": "outputs/eval/native_baseline_fulltest.json",
    "baseline_200": "outputs/eval/native_baseline.json",
}


def load_metrics(path: Path) -> dict | None:
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return None
    return raw.get("metrics")


def element_acc(path: Path) -> tuple[float, int]:
    raw = json.loads(path.read_text())
    preds = raw.get("all_predictions") or []
    if not preds:
        return 0.0, 0
    ok = sum(1 for p in preds if element_match(p.get("pred") or {}, p.get("gt") or {}))
    return ok / len(preds), len(preds)


def ckpt_num(path: str) -> int | None:
    m = re.search(r"ckpt(\d+)", path)
    return int(m.group(1)) if m else None


def best_by_element(paths: list[Path]) -> tuple[Path, float, int] | None:
    if not paths:
        return None
    scored = []
    for p in paths:
        try:
            acc, n = element_acc(p)
            scored.append((acc, n, p))
        except Exception:
            continue
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][2], scored[0][0], scored[0][1]


def sweep_table(paths: list[Path], label: str) -> list[str]:
    rows = []
    for p in paths:
        m = load_metrics(p)
        if not m:
            continue
        ea, _ = element_acc(p)
        n = ckpt_num(str(p))
        rows.append((n, m["full_match"], ea, m.get("parse_rate", 0)))
    rows.sort(key=lambda r: r[0] if r[0] is not None else 1 << 30)

    out = [f"### {label}", "",
           "| ckpt | tap-radius | element-acc | parse |",
           "|---|---|---|---|"]
    for n, fm, ea, pr in rows:
        nstr = str(n) if n is not None else "—"
        out.append(f"| {nstr} | {fm:.4f} | {ea:.4f} | {pr:.3f} |")
    out.append("")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("FINAL_SUMMARY.md"))
    args = ap.parse_args()

    lines = ["# Final summary — Run I + lifts (rescore, val sweep, Run J)", ""]
    lines.append("## Headline numbers (full test set, 8,217 rows)")
    lines.append("")

    fulltest_paths = {
        "baseline": Path("outputs/eval/native_baseline_fulltest.json"),
    }
    for p in sorted(glob.glob("outputs/eval/runI_ckpt*_fulltest.json")):
        n = ckpt_num(p)
        fulltest_paths[f"Run I ckpt-{n}"] = Path(p)
    for p in sorted(glob.glob("outputs/eval/runJ_ckpt*_fulltest.json")):
        n = ckpt_num(p)
        fulltest_paths[f"Run J ckpt-{n}"] = Path(p)

    lines.append("| run | tap-radius | element-acc | n |")
    lines.append("|---|---|---|---|")
    for name, p in fulltest_paths.items():
        if not p.exists():
            lines.append(f"| {name} | (missing) | — | — |")
            continue
        m = load_metrics(p)
        ea, n = element_acc(p)
        if m:
            lines.append(f"| {name} | {m['full_match']:.4f} | {ea:.4f} | {n} |")
    lines.append("")

    # Comparisons
    base_p = Path("outputs/eval/native_baseline_fulltest.json")
    if base_p.exists():
        base_m = load_metrics(base_p)
        base_ea, _ = element_acc(base_p)
        lines.append("## Lift over baseline (full test)")
        lines.append("")
        lines.append("| run | tap-radius Δ | element-acc Δ |")
        lines.append("|---|---|---|")
        for name, p in fulltest_paths.items():
            if name == "baseline" or not p.exists():
                continue
            m = load_metrics(p)
            ea, _ = element_acc(p)
            d_radius = m["full_match"] - base_m["full_match"]
            d_element = ea - base_ea
            lines.append(f"| {name} | {d_radius:+.4f} | {d_element:+.4f} |")
        lines.append("")

    # Best-by-val ckpt picks
    lines.append("## Best ckpts (selected by val element-accuracy)")
    lines.append("")
    for run, glob_pat in [("Run I", "outputs/eval/runI_val_ckpt*.json"),
                          ("Run J", "outputs/eval/runJ_val_ckpt*.json")]:
        paths = [Path(p) for p in sorted(glob.glob(glob_pat))]
        best = best_by_element(paths)
        if best is None:
            lines.append(f"- **{run}**: no val sweep results.")
            continue
        bp, bacc, bn = best
        lines.append(f"- **{run}** best val element-accuracy: ckpt-{ckpt_num(str(bp))} "
                     f"@ {bacc:.4f} ({bn} val rows).")
    lines.append("")

    # Sweep tables
    lines.append("## Run I val sweep")
    lines.append("")
    paths = [Path(p) for p in sorted(glob.glob("outputs/eval/runI_val_ckpt*.json"))]
    if paths:
        lines.extend(sweep_table(paths, "Val (686 rows, seed 3407)"))
    else:
        lines.append("(no Run I val sweep results)")
        lines.append("")

    lines.append("## Run I test sweep (200-sample, original)")
    lines.append("")
    paths = [Path(p) for p in sorted(glob.glob("outputs/eval/runI_ckpt*.json"))
             if "fulltest" not in p]
    if paths:
        lines.extend(sweep_table(paths, "Test 200-sample, seed 3407"))

    lines.append("## Run J val sweep")
    lines.append("")
    paths = [Path(p) for p in sorted(glob.glob("outputs/eval/runJ_val_ckpt*.json"))]
    if paths:
        lines.extend(sweep_table(paths, "Run J val (686 rows, seed 3407)"))
    else:
        lines.append("(Run J did not run or has no val results)")
        lines.append("")

    args.out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
