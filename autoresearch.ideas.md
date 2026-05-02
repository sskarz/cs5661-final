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

## r36+ harness-context backlog (user steer 2026-05-01)

User principle: "the reasoning is important when paired with teacher-student
distillation"; "we need to focus more on the harness — what context we pass
through (past 5 actions, candidate clickable elements, compact a11y tree).
Some actions the model never decides on doing because of the context it
has." → reasoning stays in labels; iterate on harness context.

Independent, stackable, in priority order:

- **Reason-preserving history**: M3AA11Y currently emits
  `Step N: <action_repr> -> ok; window changed`. Strip the action repr,
  carry the model's prior `Reason:` text instead (or alongside). Direct fix
  for navigate_back / wait collapse — those decisions are step-to-step
  coherence problems, not perception problems. Cheapest harness change;
  reuse existing adapter, no retrain. Implement in
  `android_world/agents/m3a_a11y.py` history-rendering path.

- **Candidate-action shortlist header**: emit per-step
  "Available actions: click(0..N) / scroll(up|down|left|right) /
  open_app({inventory}) / wait" rendered from the parsed a11y tree.
  Currently the model derives the action space implicitly from the element
  list. Making it explicit reduces verb-confusion (click→scroll,
  click→wait, status spam). Also eval-only first; if it lifts SR,
  retrain with the same header at train time.

- **Compact a11y tree**: drop elements with empty labels, group by
  container, mark interactive vs non-interactive, optional truncation by
  bbox size. Shorter prompt → better attention budget for relevant
  elements. Plus: a11y tree could be too long for the model context
  on certain screens (untested hypothesis).

- **Goal-substep planner**: at step 0, model emits numbered plan;
  history carries it forward. Standard literature pattern for
  long-horizon agents; directly addresses "model never plans to back
  out of a dialog" and the navigate_back floor. Larger change, requires
  new train-time format (synthesized plan or teacher-generated plan).

- **State diff signal**: currently "window changed" is a generic
  placeholder. Track what actually changed (new elements appeared,
  page text changed). Real signal for whether `wait` succeeded vs needs
  retry. Requires harness-side state tracking; cheap to implement on
  `m3a_a11y.M3AA11Y._step_summary`.

- **Dedup consecutive observations** (AppVLM trick): when two
  consecutive a11y trees are identical, drop the duplicate from
  training/history. Free data-quality win; orthogonal to everything
  above. Implement in prepare_smoke_data and m3a_a11y.

Off-the-table for now (per user steers): RFT/RL on-policy collection
(AppVLM's headline lever), drop-CoT label format (user vetoed; reasoning
is needed for teacher distillation alignment).

## r41+ plan-distillation design (drafted 2026-05-02)

User-approved direction following r40 launch. Goal: address goal-decomposition
gap revealed by r39 failure analysis (premature `status:complete` after
opening app, model can't sequence multi-step actions).

### Key insight from r36/r37
At 2B (E2B), in-context reasoning has a sharp toxicity cliff — model
can't disambiguate own-prior-thoughts from user goal text.
**Mitigation**: scope reasoning to PLAN (set once at step 0, role-clear,
in dedicated prompt slot) rather than per-step Reason in history. Plan
is task-scoped, not step-by-step.

### Design

**At step 0** (no history):
```
User: <goal>
       <inventory>
       <elements>
Assistant: Plan:
  1. Open the X app.
  2. Navigate to Y.
  3. Perform Z action.
  4. Confirm.
Reason: First step is to open X.
Action: {"action_type":"open_app","app_name":"X"}
```

**At step 1+** (harness re-injects plan):
```
User: <goal>
       Plan from step 0:
       1. Open the X app.
       2. ...
       History:
       Step 1: open_app(X) -> ok
       <elements>
Assistant: Reason: Plan step 2: navigate to Y. I see Y at index 5.
Action: {"action_type":"click","index":5}
```

### Data sources (cheapest first)

**Option A: AndroidLab plan augmentation (PREFERRED)**
- AL has 6053 multi-step trajectories with goal + action sequences
- Send each unique AL goal to teacher (Gemma 4 31B 4-bit) → get plan
- Augment AL training rows with the plan; gold actions stay AL's
- Reuses existing trajectories; only need teacher to generate plans

**Option B: Synthesize AW-style task descriptions**
- Templated goals over 19 AW apps ("Delete X from Y", "Find Z in W")
- Teacher generates plan + first action for each
- New trajectory data, more on-distribution but no real action sequences
- DO NOT use actual AW task names (contamination risk)

Start with Option A. If AL plan distillation lifts SR, augment with Option B.

### Pipeline

1. `scripts/pathZ/distill_plans.py`:
   - Load AL trajectories, dedup by goal
   - For each unique goal, send to teacher with AW inventory
   - Teacher emits 3-5 step plan
   - Output: `data/pathZ/al_plans.jsonl` mapping goal → plan
   - ~6053 unique goals (or fewer after dedup); ~3-5s/teacher call → ~5-8 hours
   - Could use existing distill_teacher.py scaffolding

2. `prepare_smoke_data.py` --use-plans flag:
   - Look up plan for each AL row's goal
   - At step 0: prepend `Plan: ...` to assistant text
   - At step 1+: inject `Plan from step 0:` block in user prompt above history

3. `m3a_a11y.py` plan handling:
   - At step 0: parse Plan from model's emission, store in self.plan
   - At step 1+: inject self.plan into action_prompt above history block

### Risks

- Teacher's plan may be wrong (no validation against ground truth actions)
- Student may overfit to plan format; lose flexibility
- Plan adds prompt budget; may dilute attention on UI elements
- 2B model still might be too small to USE the plan even if shown one
  (E4B is the right base model to test this on)

### Gating rule

- Only build this pipeline if r40 (E4B) shows AW SR ≥ 10% (recovers
  baseline), proving capacity is the lever.
- If r40 stays at 5-10%, plan distillation alone unlikely to lift past
  the structural ceiling.
- If r40 ≥ 15%, plan distillation may be the multiplier to break 25%+.

## r42 (conditional): teacher rollouts on AndroidLab tasks

User-approved conditional plan: trigger if r41 doesn't lift AW-20 SR
past 10% floor. Source clean on-distribution data without AW
contamination by collecting teacher trajectories on AL tasks (NOT AW).

**Why AL not AW**: AW-116 is the held-out test set; touching it (even
"hold out 20") leaks task templates because AW tasks are parameterized
families. AL's 138-task benchmark is disjoint from AW-116 task families
and runs on the same emulator, so trajectories are naturally on-AW-
distribution without overlap.

**Scope estimate**:
1. AL emulator setup (if not already running) — separate Android image,
   ~1 day
2. Teacher inference loop wrapper — Gemma 4 31B 4-bit running through
   M3AA11Y agent on AL tasks; record per-step (UI state, action, reason)
   ~half day
3. Trajectory→training-row converter — already partially exists in
   convert_androidlab_som.py
4. Train + eval cycle — same as r35-r41
~ Total: 1-2 days engineering before any training

**Risks**:
- Teacher may fail many AL tasks (per r34, teacher is also confused by
  text-only context). Failed trajectories aren't useful for SFT.
- AL 138 tasks × ~10 steps avg = ~1380 trajectory steps. Smaller than
  the 6053 AL Instruct rows we already have, but on-policy (real teacher
  rollouts) rather than off-policy (recorded human trajectories).
- Cost: teacher inference ~3s/step × ~1380 steps = ~70 min, but only if
  the loop completes. With max_steps=30 per task and many failures, real
  cost likely 4-6 hours.

**Gating**:
- Only build if r41 lands at ≤ 12% AW-20 (no meaningful lift over r40)
- Skip if r41 ≥ 15% (already shipping)

## r43 (conditional): drop status emission

User-approved if r41 ≤10% AW-20 (confirms premature `status:complete`
is the dominant failure mode in r41).

**Diagnosis**: r41's plan-distilled training rewards the model for
emitting `status: complete` as the last plan step. The model
generalizes this and emits status:complete on AW tasks BEFORE the
task is actually finished — the harness scores false. Same trajectory
shape across all 18 r41 failures.

**Fix**: drop `status` from action vocab entirely. AW's success check
fires post-trajectory based on env state, not on agent assertion.
The harness max_steps cap (default 30) becomes the only termination.

**Changes**:
1. `m3a_format.py M3A_PROMPT_PREFIX`: remove the two status lines
   (Mark task complete / infeasible)
2. `m3a_a11y.py`: clip emissions where action_type=='status' to a
   no-op (e.g. `wait`) at the executor; log but don't terminate
3. `prepare_smoke_data.py`: drop status class from balance set
4. AL plan postprocess: strip the final `X. status: complete` step
   from each teacher-generated plan (so the plan doesn't contain a
   directive to emit status)
5. Retrain E4B + plans on v7 data; eval AW-20

**Risk**: model may stop performing actions late in trajectories (not
sure what to do once "done"). max_steps cap absorbs this. Worst case
SR drops to 0 (everything timing out without success), which would
mean status drop wasn't the dominant problem and we need the verifier
approach (option 3 from premature-termination playbook).
