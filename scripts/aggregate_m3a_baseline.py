#!/usr/bin/env python
"""Aggregate AndroidWorld M3A baseline sweep results.

Reads every `<task>_<i>.pkl.gz` in the run directory, computes:
  - success rate (overall, by app cluster)
  - mean episode length and wall time
  - failure breakdown: parse-fail, action-format-fail, infeasible-status,
    max-steps-without-status, runtime exception

Usage:
  uv run python scripts/aggregate_m3a_baseline.py \\
      --run-dir /home/sanskar/android_world/runs/m3a_baseline_full
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# Which app a task name belongs to. Cheap heuristic: prefix before a CamelCase
# verb. Falls back to "Other" so unknown tasks still aggregate.
_APP_PREFIX_RE = re.compile(r"^([A-Z][a-z]+(?:[A-Z][a-z]+)?)")


def app_cluster(task_template: str) -> str:
    m = _APP_PREFIX_RE.match(task_template)
    return m.group(1) if m else "Other"


_REASON_ACTION_RE = re.compile(
    r"Action:\s*(\{.*\})", flags=re.DOTALL
)


def classify_failure(ep: dict) -> str:
    """Returns a short failure-mode tag for a non-successful episode."""
    if ep.get("exception_info"):
        return "runtime_exception"
    ed = ep.get("episode_data") or {}
    outs = ed.get("action_output") or []
    jsons = ed.get("action_output_json") or []
    n = len(outs)
    if n == 0:
        return "no_steps"

    # Check if last action was 'status' / 'answer' (model thought it was done)
    last_json = jsons[-1] if jsons else None
    if last_json is not None:
        at = getattr(last_json, "action_type", None)
        if at == "status":
            gs = getattr(last_json, "goal_status", None)
            if gs == "infeasible":
                return "model_gave_up_infeasible"
            return "model_thought_complete_but_wrong"
        if at == "answer":
            return "answered_but_wrong"

    # No status terminal: parse-fail or maxed steps
    parse_fails = sum(
        1 for o in outs
        if not _REASON_ACTION_RE.search(o or "")
    )
    if parse_fails >= max(1, n // 2):
        return "parse_fail_majority"
    return "max_steps_no_terminate"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    args = p.parse_args()

    pkls = sorted(args.run_dir.glob("*.pkl.gz"))
    if not pkls:
        # Look one level deeper for run_<timestamp> subdir
        sub = sorted(args.run_dir.glob("run_*"))
        if sub:
            pkls = sorted(sub[0].glob("*.pkl.gz"))
            args.run_dir = sub[0]
    if not pkls:
        sys.exit(f"No .pkl.gz files under {args.run_dir}")

    n_total = 0
    n_success = 0
    n_with_exc = 0
    by_app: dict[str, list[float]] = defaultdict(list)
    fail_modes: Counter[str] = Counter()
    times: list[float] = []
    lengths: list[int] = []
    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    for pkl in pkls:
        with gzip.open(pkl, "rb") as h:
            d = pickle.load(h)
        # The checkpointer writes a list with one element per repeat.
        for ep in d:
            n_total += 1
            tt = ep.get("task_template", pkl.stem)
            ok = float(ep.get("is_successful") or 0.0) >= 0.999
            n_success += int(ok)
            n_with_exc += int(ep.get("exception_info") is not None)
            times.append(float(ep.get("run_time") or 0.0))
            _len = ep.get("episode_length")
            try:
                lengths.append(int(_len) if _len and _len == _len else 0)
            except (TypeError, ValueError):
                lengths.append(0)
            by_app[app_cluster(tt)].append(1.0 if ok else 0.0)
            if ok:
                successes.append(tt)
            else:
                fm = classify_failure(ep)
                fail_modes[fm] += 1
                failures.append((tt, fm))

    print(f"# AndroidWorld M3A baseline (Gemma 4 E2B) — aggregated")
    print(f"run_dir: {args.run_dir}")
    print(f"episodes: {n_total}")
    print(f"success: {n_success}/{n_total} = {100.0 * n_success / max(1, n_total):.2f}%")
    print(f"runtime exceptions: {n_with_exc}")
    print(f"mean episode length: {sum(lengths) / max(1, len(lengths)):.1f} steps")
    print(f"mean run time: {sum(times) / max(1, len(times)):.1f} s")
    print(f"total wall: {sum(times):.0f} s ({sum(times) / 3600:.2f} h)")

    print()
    print("# Failure modes")
    for mode, n in fail_modes.most_common():
        print(f"  {mode:34s} {n:4d}  ({100.0*n/max(1,n_total):.1f}%)")

    print()
    print("# Per-app cluster")
    rows = []
    for app, vals in sorted(by_app.items()):
        rows.append((app, sum(vals), len(vals), 100.0 * sum(vals) / max(1, len(vals))))
    rows.sort(key=lambda r: -r[3])
    for app, k, n, pct in rows:
        print(f"  {app:24s} {int(k):3d}/{n:3d} = {pct:5.1f}%")

    print()
    print("# Successes")
    for t in successes:
        print(f"  + {t}")
    print()
    print(f"# Failures (n={len(failures)})")
    for t, fm in failures:
        print(f"  - {t:38s} [{fm}]")


if __name__ == "__main__":
    main()
