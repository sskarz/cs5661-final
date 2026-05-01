# AndroidWorld plan — AndroidLab recipe on Gemma 4 E2B

**Deadline:** 2026-05-10 (10 days from 2026-04-30)
**Hardware:** 1× RTX 4090 (24 GB), 1 emulator
**Base model:** `unsloth/gemma-4-E2B-it` (continuity with Run I/J/K/L)
**Harness for evaluation:** standard M3A (`--agent_name=m3a_gemma4_lora`, wired the same way as `m3a_gemma4_baseline`, with `--adapter_path` set to the new SFT checkpoint)
**Baseline score-to-beat:** the M3A Gemma 4 E2B baseline currently sweeping (in flight at 0/59 with 0 successes; final lands ~00:30 PDT 2026-04-30; tracking near 0% AW SR)
**Success criterion:** **≥ baseline + 15.0pp absolute on full AW (116 tasks).** If the baseline finishes at 0%, the bar is **AW SR ≥ 15.0%**.

## Why AndroidLab and not Ferret-UI Lite

The previous version of this plan targeted Apple's Ferret-UI Lite recipe (point-coordinate grounding on screenshots, SFT on ~28M grounding rows, then GRPO RL). After comparing methodologies, we chose AndroidLab instead because:

1. **AndroidLab's paradigm matches what we already built.** Path W is a11y-tree-conditioned, JSON-action-emitting, element-id-grounded, SFT-trained — these are *exactly* AndroidLab's design choices in XML mode. Apple's recipe inverts every one of these. Continuing the AndroidLab line lets us reuse Run L's training pipeline almost verbatim.
2. **AndroidLab's data is open and trajectory-shaped.** Their Android Instruction dataset (`THUDM/Android-Lab`) is 6K multi-step trajectories with history baked into the prompt — directly attacks the prompt-format narrowness we identified in Run K (87% mode collapse on AW because the LoRA only saw single-step screens).
3. **AndroidLab's published lift validates the path.** Llama-3.1-8B XML mode SFT lifts 2.17 → 23.91% on the AndroidLab benchmark with **SFT alone, no RL**. That single result is closer to Apple's full 28% AW number than Apple's SFT-only baseline (25%) — and it was achieved on a comparable parameter scale with a far smaller infrastructure footprint.
4. **AndroidLab's recipe fits a 4090 with margin.** No GRPO, no emulator-driven training, no point-coordinate regression, no 28M-row data prep. Just QLoRA SFT on a 6K-trajectory dataset + AndroidControl + a small mobile-grounding tail.

The key shift: we are no longer trying to clone Apple's *data scale*. We are reproducing AndroidLab's *training-distribution shape* (multi-step, history-aware, M3A-format) on the data we already have plus AndroidLab's release.

## Realistic outcome on Gemma 4 E2B

AndroidLab's 23.91% point is an 8B model. Gemma 4 E2B is 2B. Expect a base-size penalty of ~5–10pp. Honest target band:
- **15–18%**: most likely if the data is right and the harness is matched
- **18–22%**: optimistic; requires CoT prompting + zoom-in + constrained decoding all paying off
- **<10%**: indicates a data-format or harness-format bug, triggers the Day-6 abort gate

The 15pp lift target is set deliberately at the floor of this band so we have margin.

## Strategy in one paragraph

Build an SFT mixture from **AndroidLab's Android Instruction dataset (multi-step trajectories with history)** + **AndroidControl-train (single-step grounding, our existing data)** + a small grounding tail (mobile slices of OS-Atlas / ShowUI for element-recognition robustness). Convert all sources to a unified action vocabulary that matches **M3A's exact action schema** so the trained model speaks M3A's language at eval time. QLoRA SFT on Gemma 4 E2B with the multimodal projector unlocked. Evaluate with the M3A harness we already wired. Stretch goals if SFT clears the bar early: **V-Droid-style constrained decoding** (free at inference, kills parse-fails by construction) and **zoom-in inference** (free, +1–2pp on grounding).

## What success looks like

- A single LoRA checkpoint that, loaded into `--agent_name=m3a_gemma4_lora --adapter_path=<ckpt>` and run on the full 116-task AW suite, produces **AW SR ≥ baseline + 15pp**.
- The same `m3a_gemma_wrapper.py` and `run.py` we shipped (no harness changes for the headline number — same M3A code path, only the adapter differs).
- A reproducible `scripts/pathZ/*` chain (data prep → SFT → eval) someone else can re-run from a clean repo.

## Day-by-day plan (10 days)

### Day 1 (Fri 2026-05-01) — pull AndroidLab + write unified action schema

**Output:** `data/pathZ/raw/{androidlab,androidcontrol,os_atlas_mobile,showui_mobile}/` + `ACTION_SCHEMA_v3.md`.

- Clone `THUDM/Android-Lab` (HF + GitHub) — this is the headline data source. Pull the **Android Instruction dataset**: ~6K multi-step trajectories with history baked into the prompt. Read their training code to understand exactly how they format the prompt — we will mirror it.
- Verify the AndroidControl-train data we already prepped (Run I/J/K/L) — 159K step-level rows in v2 schema. This is our second-largest source.
- Pull **mobile-only slices** of two grounding sources for element-recognition robustness:
  - **OS-Atlas mobile** (HF `OS-Copilot/OS-Atlas-Data`, mobile filter) — sub-sample to ~200K rows
  - **ShowUI** (HF `showlab/ShowUI-2B-data`, mobile-rich) — ~30K rows
- **Skip** OS-Atlas web/desktop, UGround, AGUVIS heavy-grounding, WaveUI — Apple-recipe sources that don't transfer to AW the way trajectory data does.
- **Write the unified action schema** in `ACTION_SCHEMA_v3.md`, matching M3A's vocabulary verbatim:
  - `click` / `long_press` → `{"action_type": "click", "index": N}` (M3A's exact form)
  - `input_text` → `{"action_type": "input_text", "text": "...", "index": N}`
  - `scroll` → `{"action_type": "scroll", "direction": "<dir>", "index": <optional>}`
  - `open_app`, `navigate_back`, `navigate_home`, `wait`, `keyboard_enter` — passthrough
  - `status` → `{"action_type": "status", "goal_status": "complete"}` or `infeasible`
  - `answer` → `{"action_type": "answer", "text": "..."}`
- **Critical**: train on **all** these action types. Path W LoRA was never trained to emit `status`/`answer`/`keyboard_enter`/`long_press` — that single fact accounts for a huge chunk of Run K's 0% AW. AndroidLab's data already contains these actions; AndroidControl does not for `status`/`answer`.

### Day 2 (Sat 2026-05-02) — converters + train/val/test split + sanity check

**Output:** `data/pathZ/train.jsonl` (~600–800K rows) + `data/pathZ/val.jsonl` (~5K rows) + `data/pathZ/test.jsonl` (~5K rows).

- One Python converter per source under `scripts/pathZ/convert_<source>.py`. Each emits the unified schema.
- **Source weighting** (smaller than the Ferret-UI version of this plan — quality > quantity for trajectory training):
  - **AndroidLab Instruction trajectories** (multi-step, history-aware, AW-format-matched): **45%** (~300K rows after step-explosion)
  - **AndroidControl-train** (single-step grounding, our existing pipeline): **35%** (~250K rows, weighted toward goal-mode rows)
  - **OS-Atlas mobile** (grounding-only, element recognition): **15%** (~100K rows)
  - **ShowUI mobile**: **5%** (~30K rows)
- **Val split**: 5K rows from AndroidLab held-out + AndroidControl-val, AW-format only — used for ckpt selection.
- **Test split**: 5K rows from AndroidControl-test, AW-format only — never look at this until day 9.
- Run `scripts/pathZ/sanity_check_data.py`:
  - Random 200 rows, render the prompt and gt-action, eyeball that the schema is consistent
  - Assert that ≥10% of training rows include a `status` action (AndroidLab trajectories should bring this naturally)
  - Assert that ≥40% of training rows have a non-empty action history block (multi-step, not iid screens)
  - If either assertion fails, re-mix before training.

### Day 3 (Sun 2026-05-03) — Gemma 4 E2B SFT setup + first 1K steps

**Output:** `outputs/gemma4-e2b-pathZ-sft-runM/checkpoint-1000` + a 200-task subset M3A smoke eval (run on a held-out 5-task slice of AW for fast feedback).

- New trainer `scripts/pathZ/train_sft_pathZ.py` based on `train_sft.py` but:
  - **Multi-source data loader** with the day-2 weighting (use `datasets.interleave_datasets`)
  - **Prompt template = M3A's verbatim action-selection prompt** (see `android_world/agents/m3a.py:ACTION_SELECTION_PROMPT_TEMPLATE`). Training labels are `Reason: ... Action: {...}` strings — the *exact* format M3A's parser consumes at eval time.
  - Same Unsloth QLoRA config as Run L (rank 16, alpha 32, target modules: all linear, **multimodal projector unlocked** — we know from Run H/I that this matters)
  - LR schedule: cosine, peak 1e-4, warmup 500 steps, **8K total steps**
  - `--save-steps 500`, `--save-total-limit 20`
- Launch the first 1K steps with the emulator off (saves ~10% throughput).
- At ckpt-1000, run `eval_a11y_native.py` on AndroidControl-val (offline) to confirm SFT is converging on training-distribution.

### Days 4–5 (Mon–Tue 2026-05-04 / 05) — full SFT + first AW eval

**Output:** `outputs/gemma4-e2b-pathZ-sft-runM/checkpoint-{best-by-val}` + one full-116 M3A AW eval.

- Continue SFT through 8K steps. Wall clock estimate: ~24h with batch 4 + grad accum 4 (effective batch 16, ~2K examples/min including image preprocessing).
- Sweep checkpoints every 500 steps via offline AndroidControl-val eval (reuse `lifts_chain.sh` pattern as `pathZ_val_chain.sh`).
- **Day 5 evening**: pick best-by-val checkpoint, run **full 116-task M3A AW eval** (~5h overnight Tue→Wed).
- **Day 5 gate**: if AW SR < baseline + 5pp, abort and debug (Day 6 State C). If ≥ baseline + 5pp, stay on plan.

### Day 6 (Wed 2026-05-06) — read AW results, branch decision

**Output:** decision document in `TRAINING_LOG.md §53` + a planned Day 7–9 ablation.

Three states:

**State A — AW SR ≥ baseline + 15pp already.** Spend remaining days on robustness:
- Add V-Droid-style constrained decoding (Day 7)
- Add zoom-in inference protocol (Day 7)
- Three-seed AW eval to confirm the number isn't lucky (Day 10)

**State B — AW SR in [baseline+5pp, baseline+15pp).** Most-likely outcome. Days 7–9 add the cheap gains:
- **Day 7**: V-Droid-style constrained decoding via `outlines` or `llguidance` — extract candidate actions from the a11y tree, mask logits to enforce that set. ~4 hours of code, no training. Likely +2–4pp by killing parse-fails (currently 20.9% in baseline) and schema hallucinations.
- **Day 7**: zoom-in inference protocol — predict, crop around prediction, re-predict. Half-day. Likely +1–2pp.
- **Days 8–9**: continue SFT another 2K steps with **format-augmented data** — explicitly oversample rows where the gt action is `status` (which models almost never emit) and rows where the prior action was wrong (recovery exposure). Targets the remaining failure modes.

**State C — AW SR < baseline + 5pp.** SFT didn't transfer. Diagnose:
- Inspect AW eval pickles: parse-fail rate, action-type distribution vs gt, whether `status` is ever emitted
- Verify the M3A prompt format in training matches the M3A parser at eval time exactly (literally diff the strings)
- Check the AndroidLab data was loaded right — sanity_check_data should have caught most issues by day 2
- Re-mix data, re-train days 7–8.

### Days 7–9 — execute the Day-6 branch

State B planning baseline. State A swaps the day-8/9 work for robustness; State C is a re-train cycle.

### Day 10 (Sun 2026-05-10) — final AW eval + write-up

**Output:** final 116-task M3A AW eval (single seed; three-seed only if Day 9 finished early) + `FINAL_PATHZ.md`.

- Run `aggregate_m3a_baseline.py` (already written — handles per-app / failure-mode breakdown).
- Write the result + ablation contribution table into `FINAL_PATHZ.md`.
- Update `FUTURE_WORK.md` "score to beat" section with the new headline and pin the M3A baseline number alongside it.

## File / artifact layout (planned)

```
ANDROID_WORLD_PLAN.md                      # this file
ACTION_SCHEMA_v3.md                        # day-1 output
data/pathZ/
├── raw/{androidlab,androidcontrol,os_atlas_mobile,showui_mobile}/
├── train.jsonl                            # ~600-800K rows
├── val.jsonl                              # 5K rows
└── test.jsonl                             # 5K rows
scripts/pathZ/
├── convert_androidlab.py
├── convert_androidcontrol_to_m3a.py        # reformat existing AC data into M3A format
├── convert_os_atlas_mobile.py
├── convert_showui_mobile.py
├── sanity_check_data.py
├── train_sft_pathZ.py
├── eval_pathZ_offline.py                   # AndroidControl-val/test offline
├── pathZ_val_chain.sh                      # ckpt sweep + best-by-val pick
├── run_pathZ_aw_full.sh                    # M3A AW full eval
├── constrained_decoding_hook.py            # day-7 V-Droid-style action masking
└── zoom_inference.py                       # day-7 stretch
outputs/gemma4-e2b-pathZ-sft-runM/          # SFT checkpoints
```

## Risk register (in order of likelihood × impact)

| Risk | Likelihood | Mitigation |
|---|---|---|
| **AndroidLab data has format quirks** that don't map cleanly to M3A's prompt | High | Day 1 reads their training code first; Day 2 sanity_check_data catches mismatches before training |
| **`status` action under-represented in training mix** | Medium | Day 2 hard-asserts ≥10% of training rows emit `status`; oversample if not |
| **AW eval throughput too slow** (full 116 takes 5h+) | High (we already see this) | Run AW evals overnight; smoke on 5–10 representative tasks for fast feedback during Days 3–5 |
| **CoT in training labels confuses Gemma at this scale** | Medium | M3A's prompt template *requires* `Reason: ... Action: {...}` format. If Gemma at 2B can't manage the reason line, drop to short reasons (≤20 tokens) on Day 6 State B |
| **SFT loss flatlines** (data mix incompatibility) | Medium | Day 3 ckpt-1000 offline eval; if AndroidControl-val accuracy hasn't moved by 1K steps, debug before continuing |
| **Constrained decoding breaks on M3A's prompt** (different output format than Path W) | Low-Medium | Test on smoke set Day 7 before depending on it for the headline |
| **Emulator instability across 5h runs** | Medium | Current sweep is proving stability; keep launch flags identical |
| **AndroidLab dataset is bigger than expected and exceeds disk** | Low | Their release is ~5–10 GB based on ACL paper sizes; ample disk |

## Cuts if behind schedule

In drop-first order:
1. **Day 7 zoom-in inference** — saves 0.5 days, costs 1–2pp
2. **Days 8–9 format-augmented continuation SFT** — saves 2 days, costs 2–4pp
3. **OS-Atlas + ShowUI grounding tail** (Day 1) — saves 0.5 days; expect <1pp impact since AndroidLab is already grounding-aware
4. **Three-seed final eval** — saves 0.5 days, costs only confidence (still report 1-seed)
5. **Day 7 constrained decoding** — saves 0.5 days, costs 2–4pp. **Drop only if the model's parse rate is already >95% on the smoke set** — measure before deciding.

Order is: drop free-but-small wins first; keep the load-bearing AndroidLab+AndroidControl+M3A SFT through every cut scenario.

## What this plan deliberately is not

- **Not a Ferret-UI Lite reproduction.** Apple's screenshot-only point-coord paradigm is a 90° turn from Path W. Their proprietary 3B base + GRPO infrastructure are out of scope.
- **Not a V-Droid reproduction.** V-Droid's verifier paradigm + pair-wise preference data collection is multi-week infra. We are stealing only their inference-time constrained decoding idea (Day 7 hook).
- **Not a navigation-RL plan.** RL with one emulator on a 4090 is the wrong cost/benefit at this budget.
- **Not a multi-agent framework plan.** Mobile-Agent-E / GUI-Owl-style scaffolds need a frontier-model brain; wrapping a 2B Gemma in many calls multiplies its errors instead of correcting them.

## What we keep from prior runs

- **M3A wrapper + `run.py` dispatch** (`m3a_gemma_wrapper.py`, `m3a_gemma4_baseline` / `m3a_gemma4_lora`) — both adapters share that code path; comparison is fair by construction.
- **AndroidControl data prep + element grounding** from Run I (`prepare_a11y_native.py`, `add_prior_action.py`).
- **Unsloth QLoRA training config** from Run L (rank 16, alpha 32, projector unlocked, 1 epoch).
- **Eval chain pattern** from `lifts_chain.sh` — adapt to `pathZ_val_chain.sh`.

## Stolen ideas table

| Idea | Source | What we steal |
|---|---|---|
| Multi-step trajectories with history in training | AndroidLab (Xu et al. 2024) | The training-distribution shape (entire core of this plan) |
| Constrained decoding from a11y-extracted candidates | V-Droid (Dai et al. 2025) | Inference-time action masking via `outlines` / `llguidance` (Day 7) |
| CoT prompting before action emission | Ferret-UI Lite (Apple 2025) | Use `Reason: ... Action:` format in training labels — already required by M3A |
| Zoom-in inference (predict → crop → re-predict) | Ferret-UI Lite (Apple 2025) | Day 7 stretch on grounding-heavy tasks |
| Unified action vocabulary across data sources | Ferret-UI Lite (Apple 2025) | Unified schema that matches M3A's vocab verbatim (Day 1) |

## Single-line summary

Single-stage QLoRA SFT on Gemma 4 E2B over a 600–800K-row mobile-weighted GUI mixture (AndroidLab Instruction trajectories 45% + AndroidControl-train 35% + OS-Atlas/ShowUI mobile grounding 20%) with **M3A's exact prompt and action vocabulary**, evaluated on the standard M3A harness we already wired. Stretch: V-Droid-style constrained decoding + zoom-in inference if the headline lands early. Target: AW SR ≥ baseline + 15pp by 2026-05-10.
