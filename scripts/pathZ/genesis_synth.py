"""r47: Genesis-style teacher-rolled trajectories on AW-19 apps.

Two-pass pipeline:
1. Goal synthesis: per app, launch app → capture a11y tree → teacher emits
   N goals derived from observed UI capabilities (NOT AW task templates).
2. Rollout: per (app, goal), reset env home → launch app → run M3AA11Y
   with the 31B teacher as the LLM until the agent emits status (which is
   suppressed by the harness for r43+, so really until max_steps) or the
   step cap is hit.

Outputs:
  data/pathZ/genesis/goals.jsonl — one record per app with synthesized goals
  data/pathZ/genesis/trajectories.jsonl — one record per agent step

Run on a live AW emulator (console_port=5554). Teacher in 4-bit ≈ 16GB,
fits in 24GB alongside the emulator.

Cost (rough): teacher 31B 4-bit ≈ 5–10s/generation × ~2 generations/step ×
~10 steps/rollout × 19 apps × 5 goals ≈ 3 hours.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# android_world lives outside this repo
AW_REPO = os.environ.get(
    "ANDROID_WORLD_REPO",
    "/home/sanskar/Documents/Github/android_world",
)
if AW_REPO not in sys.path:
    sys.path.insert(0, AW_REPO)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "pathZ"))

from m3a_format import HARNESS_APP_INVENTORY  # noqa: E402

DEFAULT_TEACHER = "unsloth/gemma-4-31B-it-unsloth-bnb-4bit"
DEFAULT_OUT = "data/pathZ/genesis"
DEFAULT_GOALS_PER_APP = 5
DEFAULT_MAX_STEPS = 12
DEFAULT_CONSOLE_PORT = 5554


GOAL_SYNTH_PROMPT = (
    "You are looking at the home screen of the Android app '{app}'. The "
    "visible UI elements are listed below.\n\n"
    "{ui_block}\n\n"
    "Generate a numbered list of {n} short, distinct, plausible user "
    "goals that a real user might want to accomplish in this app. Each "
    "goal should:\n"
    "  - Be a single concrete task (not a vague intent like 'use the app')\n"
    "  - Take 2-6 steps to complete\n"
    "  - Mention specific made-up content (titles, names, numbers, dates) "
    "where the task involves creating or finding something\n"
    "  - Be discoverable from the elements visible above (don't invent "
    "features that aren't suggested by the UI)\n"
    "  - NOT be generic system tasks like 'go home' or 'navigate back'\n\n"
    "Format: just the numbered list, one goal per line, no preamble.\n\n"
    "Goals:\n"
)


def _synth_goals(llm, app_name: str, ui_block: str, n: int) -> list[str]:
    prompt = GOAL_SYNTH_PROMPT.format(app=app_name, ui_block=ui_block, n=n)
    raw, _, _ = llm.predict_mm(prompt, [])
    # Parse numbered list. Tolerant of bullets and dashes.
    goals: list[str] = []
    for line in raw.splitlines():
        m = re.match(r"^\s*(?:\d+[.)]|[-*])\s*(.+?)\s*$", line)
        if m:
            g = m.group(1).strip()
            if 5 < len(g) < 200:
                goals.append(g)
    return goals[:n]


def _capture_ui_block(env, max_elements: int = 40) -> str:
    """Return a textual list of visible UI elements from the current state."""
    state = env.get_state()
    elems = state.ui_elements
    lines = []
    for i, e in enumerate(elems[:max_elements]):
        label = (
            getattr(e, "text", None)
            or getattr(e, "content_description", None)
            or getattr(e, "class_name", None)
            or "?"
        )
        label = str(label).strip()[:80]
        lines.append(f'  {i}: "{label}"')
    return "\n".join(lines) if lines else "(empty)"


def _go_home(env):
    from android_world.env import adb_utils
    adb_utils.press_home_button(env.controller)


def _launch_app(env, app_name: str) -> bool:
    from android_world.env import adb_utils
    res = adb_utils.launch_app(app_name, env.controller)
    return res is not None


def _action_to_dict(a) -> dict | None:
    """Convert a JSONAction (or already-dict action) to a plain dict."""
    if a is None:
        return None
    if isinstance(a, dict):
        return a
    # JSONAction has .json_str() or attrs we can extract
    for meth in ("to_dict", "as_dict"):
        if hasattr(a, meth):
            try:
                return getattr(a, meth)()
            except Exception:
                pass
    if hasattr(a, "json_str"):
        try:
            return json.loads(a.json_str())
        except Exception:
            pass
    # Fallback: pull common attrs
    try:
        d = {}
        for k in ("action_type", "index", "text", "direction", "app_name",
                 "goal_status", "x", "y"):
            v = getattr(a, k, None)
            if v is not None:
                d[k] = v
        return d or None
    except Exception:
        return None


def _trajectory_record(app_name: str, goal: str, agent) -> dict:
    """Extract per-step training data from agent.history."""
    steps = []
    for i, h in enumerate(agent.history):
        steps.append({
            "step": i,
            "action_prompt": h.get("action_prompt"),
            "action_output": h.get("action_output"),
            "action_reason": h.get("action_reason"),
            "action_output_json": _action_to_dict(h.get("action_output_json")),
            "summary": h.get("summary"),
        })
    return {"app": app_name, "goal": goal, "steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=DEFAULT_TEACHER)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--goals-per-app", type=int, default=DEFAULT_GOALS_PER_APP)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--console-port", type=int, default=DEFAULT_CONSOLE_PORT)
    ap.add_argument("--apps", default="",
                    help="comma-separated AW app display names; default = all 19")
    ap.add_argument("--skip-rollout", action="store_true",
                    help="only run goal synthesis (pass 1)")
    ap.add_argument("--resume", action="store_true",
                    help="append to existing files; skip already-rolled (app,goal) pairs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    goals_p = out_dir / "goals.jsonl"
    traj_p = out_dir / "trajectories.jsonl"

    apps = [a.strip() for a in args.apps.split(",") if a.strip()] or [
        n for n, _ in HARNESS_APP_INVENTORY
    ]
    print(f"[genesis] {len(apps)} apps; teacher={args.teacher}", flush=True)

    # Resume tracking
    done_pairs: set[tuple[str, str]] = set()
    existing_goals: dict[str, list[str]] = {}
    if args.resume:
        if goals_p.exists():
            with open(goals_p) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        existing_goals[r["app"]] = r["goals"]
                    except Exception:
                        continue
        if traj_p.exists():
            with open(traj_p) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        done_pairs.add((r["app"], r["goal"]))
                    except Exception:
                        continue
        print(f"[genesis] resume: {len(existing_goals)} apps with goals; "
              f"{len(done_pairs)} (app,goal) trajectories already saved",
              flush=True)

    # Lazy imports — heavy
    print("[genesis] importing env + agent ...", flush=True)
    from android_world.env import env_launcher
    from android_world.agents import m3a_a11y
    from android_world.agents import m3a_gemma_wrapper

    print("[genesis] connecting to emulator ...", flush=True)
    env = env_launcher.load_and_setup_env(
        console_port=args.console_port,
        emulator_setup=False,
        adb_path=os.environ.get(
            "ADB_PATH",
            f"{os.path.expanduser('~')}/Android/Sdk/platform-tools/adb",
        ),
    )

    print("[genesis] loading teacher ...", flush=True)
    llm = m3a_gemma_wrapper.GemmaMultimodalWrapper(
        model_id=args.teacher,
        adapter_path=None,
        text_only=True,
    )
    # Trigger lazy load with a tiny call
    _, _, _ = llm.predict_mm("hello", [])
    print("[genesis] teacher ready", flush=True)

    agent = m3a_a11y.M3AA11Y(env, llm, name="genesis-teacher")

    # ---------- Pass 1: goal synthesis ----------
    goals_f = open(goals_p, "a" if args.resume else "w")
    apps_to_synth = [a for a in apps if a not in existing_goals]
    print(f"[genesis] pass 1: synthesizing goals for {len(apps_to_synth)} apps "
          f"(skipping {len(apps) - len(apps_to_synth)} cached)", flush=True)
    for app_name in apps_to_synth:
        try:
            _go_home(env)
            time.sleep(1.0)
            ok = _launch_app(env, app_name)
            if not ok:
                print(f"[genesis]   {app_name}: launch FAILED, skipping", flush=True)
                continue
            time.sleep(2.5)
            ui_block = _capture_ui_block(env)
            t0 = time.time()
            goals = _synth_goals(llm, app_name, ui_block, args.goals_per_app)
            print(f"[genesis]   {app_name}: {len(goals)} goals "
                  f"({time.time()-t0:.1f}s)", flush=True)
            for g in goals:
                print(f"[genesis]     - {g}", flush=True)
            rec = {"app": app_name, "ui_block": ui_block, "goals": goals}
            goals_f.write(json.dumps(rec) + "\n")
            goals_f.flush()
            existing_goals[app_name] = goals
        except Exception as e:  # noqa: BLE001
            print(f"[genesis]   {app_name}: synth ERROR: {e}", flush=True)
    goals_f.close()
    print(f"[genesis] pass 1 done; {sum(len(v) for v in existing_goals.values())} "
          f"goals total", flush=True)

    if args.skip_rollout:
        return

    # ---------- Pass 2: rollouts ----------
    traj_f = open(traj_p, "a" if args.resume else "w")
    n_traj = 0
    n_step = 0
    t_start = time.time()
    pairs = []
    for app_name in apps:
        for g in existing_goals.get(app_name, []):
            if (app_name, g) in done_pairs:
                continue
            pairs.append((app_name, g))
    print(f"[genesis] pass 2: {len(pairs)} (app,goal) rollouts "
          f"× max_steps={args.max_steps}", flush=True)

    for idx, (app_name, goal) in enumerate(pairs):
        try:
            agent.reset(go_home_on_reset=True)
            time.sleep(0.8)
            _go_home(env)
            time.sleep(0.8)
            ok = _launch_app(env, app_name)
            if not ok:
                print(f"[genesis]   {idx} {app_name!r}: launch failed", flush=True)
                continue
            time.sleep(2.0)
            for s in range(args.max_steps):
                t0 = time.time()
                try:
                    res = agent.step(goal)
                except Exception as e:  # noqa: BLE001
                    print(f"[genesis]   step {s} ERROR: {e}", flush=True)
                    break
                n_step += 1
                if res.done:
                    print(f"[genesis]   {idx}.{s}: DONE ({time.time()-t0:.1f}s)",
                          flush=True)
                    break
            rec = _trajectory_record(app_name, goal, agent)
            rec["n_steps"] = len(agent.history)
            try:
                traj_f.write(json.dumps(rec) + "\n")
                traj_f.flush()
                n_traj += 1
            except (TypeError, ValueError) as e:
                print(f"[genesis]   serialize ERROR: {e}; trying default=str", flush=True)
                traj_f.write(json.dumps(rec, default=str) + "\n")
                traj_f.flush()
                n_traj += 1
            elapsed = time.time() - t_start
            eta = elapsed / max(1, idx + 1) * (len(pairs) - idx - 1)
            print(f"[genesis] traj {n_traj}/{len(pairs)}: {app_name} | "
                  f"{goal[:60]!r} | steps={len(agent.history)} | "
                  f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[genesis]   rollout ERROR ({app_name}): {e}", flush=True)
    traj_f.close()
    print(f"[genesis] DONE. trajectories={n_traj} steps={n_step} "
          f"runtime={(time.time()-t_start)/60:.1f}m", flush=True)
    print(f"METRIC genesis_n_trajectories={n_traj}", flush=True)
    print(f"METRIC genesis_n_steps={n_step}", flush=True)


if __name__ == "__main__":
    main()
