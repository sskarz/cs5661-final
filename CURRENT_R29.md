# Where we are at run 29

This document covers what the model is, how it's trained, what it sees at
test time, two concrete (anonymised, paraphrased) trajectories from the
in-flight r29 AW slice, and the levers most likely to lift the score next.

---

## 1. The model

- **Base**: `unsloth/gemma-4-E2B-it` — Gemma 4, ~2B parameters, vision +
  language, instruction-tuned. Loaded in 4-bit NF4 via Unsloth's
  `FastVisionModel.from_pretrained(load_in_4bit=True)`.
- **Adapter**: QLoRA on top, merged at eval time via PEFT.
  - `r=32`, `α=64`, target = all-linear projections in vision and language
    transformer blocks (regex covers q/k/v/o, gate/up/down, and Gemma's
    per_layer_input_gate, per_layer_projection, embedding_projection).
  - `modules_to_save=["embedding_projection"]` — the vision-to-language
    projector is unlocked so it can drift to fit the M3A image distribution
    (SoM-marked screenshots, a different visual style than the base's
    pretraining).

The total trainable parameter count is roughly a few percent of the base.
Loss only fires on the assistant's reply (`train_on_responses_only=True`)
so the user-side prompt + image acts as conditioning, not labels.

## 2. The training data

`data/pathZ/smoke/train.jsonl` — 1500 rows total, mixed from two sources at
roughly 1000 AC + 500 AL, balanced across action classes:

- **AndroidControl-v3** (Google): static `(screenshot, instruction) → action`
  pairs with rich text — both a low-level "tap the search icon" and a
  high-level goal. We pull 6 action classes balanced at 250 rows per class:
  `click`, `scroll`, `input_text`, `wait`, `open_app`, `navigate_back`.
- **AndroidLab SoM** (THUDM): full multi-step trajectories where the input
  image is already SoM-rendered (numeric indexes drawn on bounding boxes).
  We translate their action vocab to M3A's: `tap → click`, `swipe → scroll`,
  `text → input_text`, `back → navigate_back`, `finish(success) → status`.
  ~500 rows blended in for the four overlapping classes.

Each row is rendered into the **M3A prompt format**:

```
User:
  <image>            # SoM-marked screenshot (bbox + index)
  Goal: <task goal>
  History: Step 1 - <prior summary>; Step 2 - <prior summary>; …
  UI elements:
    0: TextView "Stopwatch", clickable=true, …
    1: Button "Start", clickable=true, …
    …  Output exactly:
    Reason: <one sentence>
    Action: {"action_type": "...", ...}
```

The label is just the trailing two lines: a one-sentence reason and a JSON
action object. Loss masking ensures the model is graded only on producing
that specific shape.

## 3. Training schedule

- 400 steps × batch 1 × grad-accum 4 → ~1.07 epochs over the 1500 rows.
- Optimizer: `adamw_8bit`, lr=2e-4 cosine, warmup_steps=12, wd=0.001.
- Seq length cap: 4096 tokens (a long step with deep history can clip).
- bf16 on RTX 4090.
- Single checkpoint saved at the end (no early stopping, no eval-during-
  training — too slow).
- ~10 minutes wall time per recipe variant.

## 4. What happens at test time on AndroidWorld

The M3A harness drives a real Android emulator running the AVD
`AndroidWorldAvd`. For each of the 20 tasks in the curated slice:

1. The harness resets the emulator to the task's initial state, declares a
   goal in natural language, and starts a step loop.
2. Per step:
   - Capture screenshot, run accessibility-tree extraction, build a numbered
     UI element list, render bounding boxes + numeric indexes onto the
     screenshot (Set-of-Marks).
   - Build the M3A prompt: `goal + history-of-prior-summaries + UI
     elements + format spec`.
   - Pass `(prompt, screenshot)` to our LoRA-loaded Gemma 4 E2B via
     `m3a_gemma_wrapper.predict_mm`.
   - The model emits `Reason: ...\nAction: {...}`. The harness parses the
     JSON, validates the index against the UI element list, executes the
     action via ADB.
   - The harness asks the model for a *summary* of the step (separate
     prompt, unsupervised at training time) and appends it to history.
3. Loop ends when the model emits `status` with `goal_status="complete"`,
   max steps (typically 30) is reached, or the harness's task evaluator
   detects success/failure from the device state.

Success is judged by **task-specific evaluators** built into AndroidWorld
that inspect the resulting device state — e.g., for "Take one photo", the
evaluator checks the camera roll for a new image with a recent timestamp.
Action-match accuracy on AC or AL is a *training* signal, not the eval
signal. The eval signal is "did the device end up in the right state?".

## 5. A successful trajectory: NotesIsTodo (run 28)

Goal (paraphrased): *"Is the note titled 'To-Do List' in the Joplin app
marked as a todo item? Respond with either 'True' if it is a todo or
'False' if not."*

Why this kind of task succeeds for our model:
- The goal contains the literal app name "Joplin", so step 1 produces
  `open_app("Joplin")` — and `open_app` is the action class with the
  highest type-match accuracy in our latest training (94% in r29, 70% in
  r28). The model has learned this lock-step pattern.
- Once Joplin is open, the visible UI lists notes by title. The task only
  requires reading text, not editing — a `click` plus an `answer` action.
- The decision boundary ("is this a todo?") is visible in a single
  screenshot once the note is opened.
- No `input_text` involved, no `status` required (the harness accepts the
  `answer` action as terminal).

Trace shape (compressed):
```
step 1: Reason: "I need to open Joplin to find the To-Do List note."
        Action: open_app("Joplin")          ✅
step 2: Reason: "I see the notes list; I'll tap on 'To-Do List'."
        Action: click(index=4)              ✅
step 3: Reason: "The note is shown as a checklist with checkboxes — that
                 means it is marked as a todo."
        Action: answer(text="True")         ✅
[task evaluator: True matches ground truth → Task Successful ✅]
```

## 6. An unsuccessful trajectory: FilesDeleteFile

Goal: *"Delete the file q2a8_fancy_banana.mp3 from the Android filesystem
located in the Notifications folder within the sdk_gphone_x86_64 storage
area."*

What happens, every run:
```
step 1: Reason: "I need to open the file manager to delete the file."
        Action: open_app("File Manager")    ⚠
        adb_utils: refusing to monkey-launch app_name with whitespace
        → no-op on the emulator
[harness retries the same step, loops 30 times, marks Task Failed ❌]
```

Why it fails:
- The instruction-tuned Gemma calls the app "File Manager" (two words),
  which Android does not accept directly via `monkey -p` because monkey
  parses arguments space-separated. We patched `adb_utils.launch_app` to
  refuse such names (otherwise the harness would deadlock waiting for an
  activity that never starts), but that only prevents the deadlock — it
  doesn't fix the failure.
- Our training data does not bridge "File Manager" → `com.android.documentsui`
  or `com.google.android.documentsui` (the actual package name). The model
  has no signal that *natural-language app names* must be translated to
  *package names* for `open_app`.
- Even if we taught the model the right package name, the AW slice task
  expects file-system actions on `Notifications/`, which on this emulator
  is reachable through the Files app's "Browse" view via several layers
  of menus. Our SoM training data has very few files-app trajectories.
- Worse: `input_text` (needed to type the filename in the search field)
  collapsed to **0% type-match** on AC in r29 — the model cannot reliably
  emit `input_text` at all.

So the failure mode is: open_app whiff at step 1 → wrong app or no app →
30 wasted steps → fail.

## 7. The leverage points

In rough order of expected ROI right now:

1. **Variance reduction.** The 10-task slice has σ ≈ 21pp from seed alone
   (r22=50%, r26=10%, r28=20% for the *same recipe*). r29 is the first
   run on a 20-task slice, which should roughly halve σ. Until the eval
   signal is reliable below ±10pp we cannot tell whether a +5pp recipe
   change is real. **This is the gating step.**

2. **Fix the `open_app` package-name gap.** Make a small synthetic table
   `{display_name → package_name}` for the AW apps (Files, Markor, Joplin,
   Broccoli, ProExpense, Camera, Clock, Calendar, SMS, Contacts) and
   inject ~50 training rows that show the model the canonical mapping.
   Cheaper alternative: pre-process the model's `open_app` output in the
   wrapper to do the mapping post-hoc. Either way, "wrong app name at
   step 1" is by far the dominant failure mode in our trajectories.

3. **Recover `input_text`.** It collapsed from 37% (r22) to 7% (r28) to
   0% (r29). Either the balanced 250/cls is now dominated by AL rows that
   use a different `text` field shape, or the additional AL rows are
   teaching the model to skip `input_text` in favour of `click`. Need to
   audit the 250 input_text rows in the latest training set.

4. **Memory-field training + max_length 8192.** Long AW tasks accumulate
   30+ history lines; our train-time max is 4096 which clips. AndroidLab
   prompts include an explicit "Memory" field that is currently unused at
   train time. (Logged in `autoresearch.ideas.md`.)

5. **Constrained decoding at AW eval.** Use outlines/llguidance to enforce
   the M3A schema at decoding time. This won't help the model *choose*
   the right action, but it will fix every parse-failure step (we already
   sit at 99%+ parse rate, so this is a smaller win).

6. **Teacher-distilled CoT** (Gemini 1.5 Pro / GPT-4o rationales over AC
   trajectories). Logged in the ideas backlog as Phase-3 with caveats —
   the 2B base may not have the capacity to absorb richer reasoning, and
   our schema-anchored Reason ablation in run 9 already DISCARDED for
   regressing full-match. Pursue only after grounding is fixed.

## 8. Where r29 sits

- Train loss: 0.7132 (r22 was 0.7262, r28 was 0.7307 — same recipe, CUDA
  non-determinism on the order of 1pp loss).
- AC offline: full=21.00%, type=55.80%; **open_app=93.94% (highest yet)**
  but **input_text=0.00% (collapsed)**.
- AL offline: full=4.38%, type=37.85%; status type-match=0% (still).
- AW-20 slice: in flight at the time of writing.
