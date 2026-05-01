# Autoresearch ideas backlog (pathZ smoke — Phase 2 AndroidLab-faithful)

## After r31: harness rewrite for pure-a11y (user steer 2026-05-01)
Make the harness more robust now that we're committed to a11y-only.
Order by ROI:
1. **Indexed app inventory in prompt**. Read `_PATTERN_TO_ACTIVITY`
   from `adb_utils.py`, render as `Installed apps: 0: "Files"
   (com.android.documentsui) ...` and prepend to action_prompt.
   Constrain `open_app` to require an integer index. Mirror in
   training prompts so the model learns to consume it. Largest
   expected lift — "wrong app name at step 1" is our dominant
   failure mode.
2. **Deterministic history summaries**. Replace the second LLM call
   per step (m3a.py:559) with a template: `Step N: {action} →
   {executor_feedback}` where the feedback half is derived from a
   small a11y-tree diff (focused window changed, list count
   delta, etc.), NOT from another LLM call. Halves per-step
   latency and removes a hallucination source.
3. **Executor feedback in history**. Once history is deterministic,
   surface index-validation, monkey launch result, and
   package-name-not-found errors back into next step's prompt.
4. **No-op detection**. After 2 consecutive identical
   before-states, force `status(infeasible)` or a "try a different
   approach" branch. Stops the 30-step retry-the-same-action
   failure mode.
5. **Teacher/student schema parity**. Make `prepare_smoke_data.py`
   emit the EXACT new student prompt — indexed inventory +
   deterministic history + UI elements. AC has step_index/
   episode_id metadata to compute deterministic summaries.

## NEXT (after r30): pure-a11y mode (user steer 2026-05-01)
AndroidLab paper found XML/a11y mode beats SoM for fine-tuned ≤9B models
(+3-6pp AW SR on their 9B variants; gap larger at smaller scale per
literature on SLMs). Worth direct comparison on our 2B Gemma 4.

**Audit verdict: GO-WITH-FIX.** Concrete patch set:
1. **Data**: drop AL rows from training (625/1750 = 36%) since they have
   `elements=[]` and no UI-element text in the prompt — they are SoM-only.
   Train pure-a11y on AC-only (~1500 rows balanced 250/cls × 6, +
   input_text boost stays). Run-16 recipe was AC-only and topped AC
   offline at 23.4%; we never tested it on AW.
2. **Harness**: patch `android_world/agents/m3a.py` line 419 (and the
   summary call at line 571) to call `predict_mm(prompt, [])` — drop
   both screenshots. The action_prompt already includes the rendered
   UI element list, so the model retains full a11y grounding.
3. **Wrapper**: no change — `m3a_gemma_wrapper.predict_mm` already
   degrades gracefully when `images=[]`.
4. **Schema**: no change. Same JSONL format, same train/eval scripts.

Expected outcome: if AndroidLab's trend holds at 2B, pure-a11y should
match or beat r22's 50% (with much lower variance since one input
modality removes a major source of distribution noise).

## Phase 2 immediate (high priority)
- **Pull `THUDM/Android-Lab` Instruction dataset** — HF or GitHub release.
  Inspect schema: trajectories vs flat steps, action vocab, history format.
- **Write `convert_androidlab.py`** — map their action schema to M3A:
  - `tap` → `click(index=N)`
  - `swipe` → `scroll(direction=...)`
  - `text` → `input_text(index, text)`
  - `back/home` → `navigate_back/navigate_home`
  - `finish(success)` → `status(goal_status="complete")`
  - `finish(impossible)` → `status(goal_status="infeasible")`
- **Multi-step prompt rendering** — bake the prior step summaries into
  the user prompt, mirroring how M3A passes history at AW eval time.
- **ReAct-style Reason** — convert AndroidLab's "Thought" field to our
  Reason. Their thoughts are 1-3 sentences explaining the next action;
  much richer than the synthetic one-liners we've been using.
- **Mixed-source eval set** — sample held-out rows from BOTH
  AndroidLab-val and AndroidControl-val. AC-only eval may understate
  the impact of trajectory training.

## Phase 2 follow-on (medium priority)
- **Mixing weight sweep**: AndroidLab vs AndroidControl at 45/35,
  60/30, 100/0 to see how much each contributes.
- **Trajectory-aware loss masking**: in multi-step rows, do we mask
  loss on prior-action history (response-only) or compute full-sequence
  loss including thoughts at every prior step?
- **Length scaling**: AndroidLab trajectories are longer than AC
  single-steps. May need max_length=8192 (currently 4096).

## Phase 1 levers we can still re-test once Phase 2 lands
- **lora_r 32 + 300 steps** = run 16 = current best (23.40%) on AC-only
- Per-class target 250 vs 400 — found 250 was sweet spot on AC
- Schema-anchored Reason — discarded on AC because it interferes with
  open_app args; AndroidLab's natural Thought field may sidestep this
- Class balancing — the AC fix may not be needed at all once
  AndroidLab's natural multi-action mix is in (their trajectories
  contain finish actions which AC lacks entirely)

## Teacher-distilled CoT (Gemini suggestion, deferred to Phase 3)
- Pass AC trajectories through a frontier model (Gemini 1.5 Pro / GPT-4o)
  to generate richer (state → thought → action) rationales than AC's
  terse instruction field. Train Gemma 4 E2B on the augmented set.
- **Caveats from our own runs**:
  - Run 9 (schema-anchored Reason) DISCARDED: longer reason hurt full-
    match -2pp on the 2B model; reason text crowded out open_app args.
  - 2B may lack the capacity AndroidLab's 9B used to absorb CoT.
  - Cost: 2100 rows × frontier API call w/ images = ~$30-100 + hours.
- **Defer until**: variance is reduced (20-task slice), input_text/
  status emission is fixed. CoT helps after grounding is saturated.

## Long-horizon context management (user-requested follow-up)
- **max_length 4096 → 8192** at train time. AW tasks can hit 30+ steps;
  m3a.py rebuilds history as `Step N- <summary>` with no truncation, so
  late-step prompts may exceed train-time max and degrade LoRA behavior.
- **Train an explicit memory field**. AndroidLab prompts elicit a
  per-step "Memory" string for info that needs persistence; M3A's
  `summary` field is the rough equivalent but is generated by an
  unsupervised summary call at eval time. Surface AndroidLab's memory
  annotations in the SFT label so the model learns to write durable
  summaries rather than verbose recaps. Pair with the max_length bump
  since structured memory keeps history per-step short.
- **History compaction policy**. If we keep observed reason+action
  full-text in history, M3A prompts blow up fast. Either teach the
  model to emit terse memory lines, or add a sliding-window /
  K-most-recent-steps cap in the wrapper.

## Architectural changes (only if simple variations cap)
- **Constrained decoding** (outlines/llguidance) at AW eval to enforce
  the M3A schema vocabulary
- **AC trajectory reconstruction** — AC has episode_id/step_index
  metadata; we could group AC rows back into trajectories and add
  multi-step history that way. Backup if AndroidLab data is hard to
  use.
- **Add long_press / status / answer synthetic rows** to teach actions
  AC doesn't have. AndroidLab Instruction trajectories should bring
  status naturally.
