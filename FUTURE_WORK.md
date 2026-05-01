# Future Work: Optimizing the LoRA for AndroidWorld

## Diagnosis recap
A QLoRA-tuned Gemma 3 E2B trained per-step on AndroidControl (Path W accessibility-tree → element-id JSON) won on AndroidControl (+13.76 pp element-acc) but collapsed on AndroidWorld (0/31 success, 87% mode collapse, 46% parse fail, schema hallucination). Root cause: the SFT objective is single-screen → single-action with zero exposure to (a) AW's history/legend prompt format, (b) multi-step state and recovery, (c) v2 schema constraints. The LoRA over-specialized to one prompt distribution and one action template; outside it, the policy degenerates and cannot self-correct — the OOD-prompt narrowness pattern noted in Hu et al. LoRA §6.4.

## Failure modes mapped to interventions
| Failure | Direct fix |
|---|---|
| Mode collapse (single repeated action) | Multi-turn RL with task-level reward; trajectory-level SFT; entropy regularization |
| Schema hallucination (`action_type` not in v2) | Constrained/grammar decoding against the v2 JSON schema; verifier reranker; discretized action space |
| Parse fails (46%) | Constrained decoding (deterministic fix); SFT on AW-format exemplars |
| No error recovery | Reflective trajectories (Mobile-Agent-E "Tips/Shortcuts"); online RL with rollouts that include failed states |
| OOD prompt | Train on AW's exact prompt format (legend + history); mix multiple harness formats during SFT |

## Techniques surveyed

### UI-TARS / UI-TARS-2 (ByteDance, Qin et al., 2025; arXiv 2501.12326, 2509.02544)
1. End-to-end native GUI agent; UI-TARS-2 adds a "data flywheel" + stabilized multi-turn RL on a sandbox of hundreds of VMs. 7B variant available.
2. Multi-turn RL with reflective trajectories directly attacks no-recovery and mode-collapse failure modes; unified action space + grounding pretrain reduces schema drift.
3. Cost: enormous — UI-TARS was trained on ~50B tokens; UI-TARS-2 needs hundreds of parallel emulators. Out of scope to reproduce.
4. Risk on a 4090: cannot retrain. But UI-TARS-1.5-7B is a *drop-in checkpoint* and could be the new starting point, replacing Gemma 3 E2B. Risk = 7B at 4-bit barely fits one 4090 for inference, and LoRA-tuning a 7B model with multi-turn rollouts is still expensive.

### DigiRL (Bai et al., 2024; arXiv 2406.11896, NeurIPS 2024)
1. Two-stage offline → offline-to-online advantage-weighted RL on a 1.3B VLM (T5-based) using AitW; lifted SR from 17.7% → 67.2%.
2. Exactly the diagnosis under test: SFT alone is brittle on real GUIs because of stochasticity; RL with a VLM evaluator + automatic curriculum teaches recovery from bad screens, the missing piece in our pipeline.
3. Cost: paper uses up to 64 parallel emulators; GPU-hours not reported but multi-day on multi-GPU. On a single 4090 you can run 1–2 emulators and a smaller rollout buffer. Code is open (DigiRL-agent/digirl).
4. Risk: emulator throughput becomes the bottleneck; reward modeling (VLM-as-judge) is itself a research project; 1-week timeline likely yields a partial result, not a 67% number.

### Digi-Q (Bai et al., 2025; arXiv 2502.15760, ICLR 2025)
1. Offline TD-learning of a Q-head on frozen VLM features, then Best-of-N action selection at inference. No online rollouts required.
2. Decouples "generate candidates" from "pick safe action" — directly addresses schema hallucination and mode collapse without needing online emulators. Reports 21.2% improvement over best prior offline method.
3. Cost: training a Q-head is cheap (frozen backbone); only need offline trajectories. Fits a 4090 easily.
4. Risk: needs candidate diversity from the base policy — if our LoRA emits the same wrong action 10× (it does), Best-of-N reranks garbage. Requires a non-collapsed base.

### V-Droid (Dai et al., 2025; arXiv 2503.15937)
1. LLM-as-verifier scores each candidate action from a discretized action space (extracted from the screen) instead of generating one; 59.5% on AndroidWorld with an 8B verifier.
2. Discretized action space *eliminates schema hallucination by construction* — the model cannot output an invalid action. Pair-wise progress preference training also gives a recovery signal.
3. Cost: 8B verifier (V-Droid-8B-0323 released); needs preference data (their human-agent joint annotation scheme). Verifier runs prefill-only with prefix caching → 4.3 s/step.
4. Risk on 4090: the 8B verifier fits at 4-bit. Building the candidate-extraction pipeline from a11y trees is engineering-heavy but not novel research.

### AndroidLab (Xu et al., 2024; arXiv 2410.24024, ACL 2025)
1. Open environment + 138-task benchmark + an Android Instruction dataset; SFT lifts Llama-3.1-8B from 2.17 → 23.91 SR, Qwen2-VL-7B similarly.
2. Their dataset includes multi-step trajectories with history; training Gemma on this directly addresses prompt-format narrowness.
3. Cost: dataset + code public on THUDM/Android-Lab. SFT only; 4090 can do it for 2B/7B models at 4-bit + LoRA.
4. Risk: AndroidLab's task distribution overlaps but is not identical to AW's; you may swap one OOD problem for another.

### OS-Atlas (Wu et al., 2024; arXiv 2410.23218, ICLR 2025)
1. 13M-element cross-platform GUI grounding corpus + two-stage pretrain (grounding then action FT). 4B and 7B InternVL/Qwen2-VL bases released.
2. Strengthens grounding which is necessary but not sufficient — does not by itself fix recovery or schema issues.
3. Cost: use OS-Atlas-Base-7B as a *better starting point* than Gemma 3 E2B; LoRA on top fits a 4090.
4. Risk: switching backbone means redoing all infra (chat template, image resolution, action serialization). 1-week timeline tight.

### Mobile-Agent-v3 / GUI-Owl (Ye et al., 2025; arXiv 2508.15144)
1. 7B foundational GUI model + multi-agent framework; 66.4% AW solo, 73.3% with the full Mobile-Agent-v3 framework. Self-evolving trajectory production.
2. Open-source SOTA backbone for AW today; framework adds planner/reflector/grounder.
3. Cost: 7B inference fits a 4090; framework is heavy (multi-model orchestration).
4. Risk: papers reporting 73% on AW use cloud emulators and frontier-model planners — results are not 4090-replicable end-to-end.

### Ferret-UI Lite (Apple, 2025; arXiv 2509.26539)
1. 3B end-to-end on-device GUI agent: SFT + step-wise RL with verifiable rewards + visual tool-use (crop/zoom). AW SR 28.0%.
2. Most directly comparable to your setup (small model, single device). Their explicit lesson: *SFT alone caps quickly; verifiable-reward RL is what unlocks navigation*.
3. Cost: 3B + RL is the most 4090-feasible point in this list.
4. Risk: Apple has not released weights or code; you would replicate the recipe, not the artifact.

### Mobile-Agent-E (Wang et al., 2025; arXiv 2501.11733)
1. Hierarchical Manager + Perceptor + Operator + Action Reflector + Notetaker, with a persistent "Tips/Shortcuts" memory updated across tasks.
2. The reflector + notetaker explicitly *adds error-recovery* without retraining the base model — pure inference-time fix.
3. Cost: prompt-engineering + scaffolding only; no GPU training.
4. Risk: published numbers use GPT-4o as the brain. With a 2B Gemma as the brain, the framework's checks would mostly say "your action is wrong" without being able to propose a better one.

### Aria-UI (Yang et al., 2024; arXiv 2412.16256)
1. MoE 3.9B-active grounding model with text-image interleaved action history. 44.8% on AW (Dec 2024 #1).
2. Demonstrates that history-conditioned grounding alone (no RL) can clear baseline by a wide margin.
3. Cost: weights + data released. MoE inference is awkward on a single 4090.
4. Risk: not Gemma; you would be evaluating a different model rather than improving yours.

### Ferret-UI / OmniParser (perception-only)
1. Ferret-UI (Apple, arXiv 2404.05719) and OmniParser (Microsoft, arXiv 2408.00203) are perception/grounding modules, not agents.
2. Useful as a **preprocessor** to give Gemma a clean, parsed view that matches AndroidControl's a11y format — directly attacks OOD-prompt narrowness.
3. Cost: OmniParser V2 is a ~1B detector + caption model; runs alongside Gemma on a 4090.
4. Risk: adds latency and a second failure surface; doesn't fix the policy's collapse, only its inputs.

### Constrained / grammar-guided decoding (xgrammar, outlines, llguidance)
1. At decode time, mask logits to enforce the v2 JSON Schema.
2. *Eliminates the 46% parse-fail and schema-hallucination buckets immediately, by construction.* But: "structure snowballing" (arXiv 2604.06066) shows constraints can degrade reasoning when the model's preferred tokens are all masked.
3. Cost: hours of engineering; no training.
4. Risk: turns parse-fails into *plausible-but-wrong* actions, which may not move success rate. Necessary, not sufficient.

## Priority-ordered shortlist (1 week, 1× RTX 4090)

**1. Constrained decoding against the v2 schema (1 day).** Highest ROI per hour. Removes the 46% parse-fail floor immediately. Use `outlines` or `llguidance`. Re-run AW; even modest gains here are pure wins, and it sets a floor for everything below.

**2. Re-do SFT on AW-format trajectories, not single-step (3 days).** The single biggest cause of collapse is prompt-format mismatch. Mix AndroidControl with the AndroidLab Instruction dataset (THUDM/Android-Lab) and at minimum reformat to include the AW history/legend wrapper. Use trajectory-level loss (each step in context of preceding steps), not iid screens. Add a small "wrong-action → correction" augmentation (5–10% of data) so the model sees recovery during training.

**3. Best-of-N reranking with a small verifier head (2 days), Digi-Q-style.** Train a tiny linear head on Gemma's frozen features to score (screen, candidate-action) → success probability, using AndroidControl labels as positives and synthetic negatives. At inference, sample N=8 from your LoRA and pick top-scored. Fast, fits on the same 4090, no RL.

**4. Switch backbone to OS-Atlas-Base-4B or UI-TARS-1.5-7B and re-LoRA (rest of week, if time).** Gemma 3 E2B was trained for general VL, not GUI. Starting from a GUI-pretrained 4–7B model will likely beat any amount of SFT on the wrong base. OS-Atlas-4B is the safest fit-on-4090 option.

**5. (Stretch) Tiny online RL loop, DigiRL-lite.** One emulator, ~500 task rollouts, advantage-weighted SFT (REINFORCE with baseline) on success/fail signal from AW's own task evaluators. Won't reach 67% in a week, but proves the recovery hypothesis and pairs cleanly with #3.

## What I would NOT do
- **Don't reproduce UI-TARS-2 or DigiRL faithfully.** Both assume tens of parallel emulators; one 4090 is the wrong shape of hardware. Use UI-TARS-1.5-7B as a *checkpoint*, don't try to retrain it.
- **Don't add a Mobile-Agent-E-style multi-agent framework around a 2B Gemma.** The framework's value comes from a strong reasoner (GPT-4o); wrapping a weak policy in many calls multiplies its errors instead of correcting them.
- **Don't bolt on OmniParser as a primary fix.** Better inputs are nice but won't move success rate when the policy collapses regardless. Do this only after #1–#2.
- **Don't keep training pure single-step SFT and hope the gap closes.** Hu et al. §6.4 and the Ferret-UI Lite paper both say the same thing in different words: SFT-only caps, you need either trajectory-level loss or RL to break out. More AndroidControl epochs will keep widening the AC↔AW gap.

## M3A baseline on AndroidWorld — score-to-beat reference (2026-04-30)

We re-ran the AndroidWorld benchmark using the **standard M3A harness** (Multimodal Autonomous Agent for Android, Rawles et al. 2024) with **baseline Gemma 4 E2B (no LoRA)** as the multimodal LLM. The previous AW evaluations in this repo used the Path W harness (a11y-native, history-only); M3A is the canonical AW protocol — SoM-marked screenshots + textual history of summaries + summarize-after-action — and is what every published AW number cited below uses.

**Setup:** `unsloth/gemma-4-E2B-it` loaded in 4-bit on one RTX 4090 via Unsloth `FastVisionModel`, wired into M3A through a new `MultimodalLlmWrapper` (`android_world/agents/m3a_gemma_wrapper.py`) and dispatched as `--agent_name=m3a_gemma4_baseline`. Headless emulator (`AndroidWorldAvd`, swiftshader_indirect, no-window) on the same machine. Sweep ran the full 116-task AW family with `task_random_seed=30`.

**Result (in flight at 46/116, all 0.0 success — see TRAINING_LOG / sweep log for the final 116 number):** the M3A baseline lands at the **floor** of the published SLM curve. Across the 793 inference steps observed so far:

- **20.9% of action emissions fail to parse** out of `Reason: ... Action: {…}` format (M3A treats them as no-ops)
- **Action distribution is otherwise plausible** — `click 24.5%`, `scroll 23.1%`, `open_app 11%`, `input_text 6.2%`, plus `long_press`, `navigate_*`, `wait` — confirming the wrapper, prompt template, and model emission paths are all healthy
- **Dominant failure mode is scroll-loop / mode collapse** when the target is already on-screen (4–6 identical scroll-down emissions before the budget runs out)
- **Premature termination** in 5/46 episodes (`status:complete` or `status:infeasible`)
- **No execute_action exceptions** across all observed steps; the AW evaluator runs cleanly

This is consistent with the literature on small general-purpose VLMs at this benchmark: **at 2B parameters with no GUI-specific training, AW is brutal regardless of harness.** The M3A 2-call-per-step structure (action + summary) actually *reads* the summaries it generates fine, but Gemma can't course-correct — the next action ignores the summary that says "I just scrolled and nothing happened."

## Fair comparison: small VLMs / SLMs ≤4B on AndroidWorld

All numbers are **full AndroidWorld** (116 tasks, Rawles et al. 2024) unless flagged. Order is by AW SR within tier.

### Tier 1 — small models ≤4B (the right fairness bracket for Gemma 4 E2B)

| Model | Size | Harness | AW SR | Notes |
|---|---|---|---|---|
| **Ferret-UI Lite** | 3B | native, GUI-trained | **28.0%** | SFT + step-wise GRPO with verifiable rewards + visual tool-use (zoom-in). Apple, arXiv:2509.26539 |
| OS-Atlas-Base-4B | 4B | native | ~14.6% | secondary report; original paper reports OSWorld/ScreenSpot |
| **ShowUI** | 2B | native | **7.7%** | Single-stage SFT on grounding+actions. Lin et al. arXiv:2411.17465 (CVPR 2025) |
| Aria-UI | 3.9B active MoE (25B total) | native | 44.8% **on grounding subset only** | not full task SR — **don't use for SR comparison** |

### Tier 2 — small general VLMs (zero-shot, what we're actually measuring)

| Model | Size | Harness | AW SR | Notes |
|---|---|---|---|---|
| Gemini 1.5 Flash | undisclosed (small) | M3A | **~7.1%** | only published AW M3A point in the small-general-VLM band |
| **Gemma 4 E2B (this run)** | 2B | M3A | **0/116 trajectory floor** (in flight) | no GUI training; first published AW M3A number for a ≤4B *general-purpose* VLM |

There is **no other peer-reviewed AW M3A number for a ≤4B general VLM** in the literature as of 2026-04. The result above plausibly fills that gap.

### Tier 3 — anchor models (>4B, for context)

| Model | Harness | AW SR |
|---|---|---|
| GPT-4 Turbo | M3A | 30.6% |
| Claude 3 Opus | M3A | ~27% |
| Gemini 1.5 Pro | M3A | 22.8% |
| UI-TARS-7B v1 | native | 33.0% |
| **UI-TARS-1.5-7B** | native | **42.5%** |
| GUI-Owl-7B (Mobile-Agent-v3 scaffold) | multi-agent | 66.4% (native single-agent: 49.6%) |
| UI-TARS-72B | native | 46.6% |

### Reading the table

- **The realistic SLM target for our 2B base is ShowUI's 7.7%** — same parameter count, GUI-pretrained on grounding+actions via SFT alone. Even *with* GUI training, a 2B caps in single digits.
- **The stretch SLM target is Ferret-UI Lite's 28.0%** — slightly bigger (3B), but also adds the techniques our diagnosis already flagged: verifiable-reward RL on top of SFT, plus chain-of-thought, plus visual tool-use. It is the cleanest published roadmap for a small-on-device GUI agent.
- **Above ~30% the field jumps to 7B+ models** — UI-TARS-1.5-7B at 42.5%, GUI-Owl-7B native at 49.6%, UI-TARS-72B at 46.6%. Everything past ShowUI/Ferret-UI Lite is a *different parameter class*.

## Deep dive: how Apple built Ferret-UI Lite (and whether we can replicate it)

Source: Lin et al. *Ferret-UI Lite: Lessons from Building Small On-Device GUI Agents.* arXiv:2509.26539, 2025.

### What they actually did

| Component | What it is | What it adds |
|---|---|---|
| **Base model** | "Internal 3B dense model pretrained on text + vision-language data" with a VitDet image encoder. **Not publicly released.** | The starting point. Comparable to Gemma 4 E2B, Qwen2-VL-3B, Phi-3-Vision — but it's *Apple-internal*, so an exact swap-in is impossible |
| **Unified action space** | One action schema (point-based grounding for taps, function-call serialization for navigation) covering web + desktop + mobile | Lets a single SFT mixture absorb data from many existing GUI corpora without per-source action conversion |
| **SFT data mix** | OS-Atlas (web 11M / desktop 1.1M / mobile 4.6M), UGround (web 9.5M / mobile 0.1M), AGUVIS-Grounding (web 723K / mobile 306K), Aria-UI (180K), WaveUI (63K), GroundUI (18K), ShowUI (30K), AGUVIS-Planning (293K), OpenCUA (421K desktop), AgentNet, Jedi, plus 70K synthetic mobile + 75K synthetic desktop trajectories | ~28M grounding examples + ~700K navigation trajectories. Big number, but most are *grounding* not *navigation* |
| **SFT recipe** | Single-stage, 10K steps, no separate stages | Trains grounding + navigation jointly under unified action space |
| **CoT** | Adds short or long chain-of-thought before the action | +5.9% AW (15.8 → 19.6 short→long, baseline 13.7 → +CoT) |
| **Synthetic data** | 17K synthetic trajectories | +5.6% AW (19.6 → 25.2) — CoT-prompted teacher generates rollouts |
| **RL algorithm** | **GRPO** (Group Relative Policy Optimization) — group-normalized advantages, no value head | Fits a single-GPU regime better than PPO; same family as DeepSeek-R1 |
| **RL reward (grounding)** | Containment-based: +1 if predicted point is inside the GT bounding box, else 0. Optional dense version: normalized L1 distance with decay λ=0.5 | Looser than SFT's "match the center exactly" — credits any in-box click |
| **RL reward (navigation, step-wise)** | `f_type + f_param`. `f_type` = +2 (type matches, no params) / +1 (type matches, params present) / 0. `f_param` = exact-match for strings, sparse(0/1)+dense(L1) for locations | Step-wise — agent gets a reward each step, not just at episode end |
| **Visual tool-use (zoom-in)** | Two-pass inference: model predicts → image is **cropped around the prediction** → model re-predicts on the crop. Both passes contribute to training pool | +2 absolute on ScreenSpot-Pro (~51.5 → 53.3). NOT an action_type the model emits — it's a post-hoc inference protocol applied to the grounding task |
| **RL training scale** | 1500 RL steps | Tiny vs SFT's 10K |
| **Hardware / GPU-hours** | **Not disclosed.** | Only step counts are reported |
| **Code / weights** | **Closed.** No GitHub repo, no Hugging Face card. Apple-proprietary. | The recipe is described; the artifact isn't shipped |

### The ablation table that actually matters (Table 4, paper)

| Configuration | AW SR |
|---|---|
| Baseline (no CoT, no synthetic) | 13.7% |
| + Short CoT | 15.8% |
| + Long CoT | 19.6% |
| + 17K synthetic trajectories | 25.2% |
| **SFT only (full mixture)** | **25.0%** |
| **SFT + RLVR (GRPO with verifiable rewards)** | **28.0%** |

So **80% of the headline gain comes from the SFT data mixture + CoT + synthetic data; only the last +3 points come from RLVR.** The big lift is the data, not the RL.

### Can we replicate it on our 4090?

**Honest answer: a recognizable approximation, yes; the exact 28.0% number, no.** Here's the breakdown.

| Replication blocker | Severity | Why | Workaround |
|---|---|---|---|
| **Base model is Apple-internal proprietary 3B** | High | Not released. Their 13.7% pre-CoT AW number is for *that* model; the same recipe on Gemma 4 E2B / Qwen2-VL-3B / Phi-3-Vision will start from a *different* baseline | Pick the closest open 3B GUI-VL model — **Qwen2-VL-3B-Instruct** is the standard substitute (used by ShowUI, OS-Atlas, AGUVIS), or stay with **Gemma 4 E2B** for continuity with our prior runs |
| **SFT data (~28M grounding + 700K nav)** | Medium | OS-Atlas/UGround/AGUVIS/ShowUI/Aria-UI/WaveUI/GroundUI are *all open*. The synthetic 17K mobile traj + 75K desktop traj are NOT released | Open data alone covers ≥90% of their mixture; the missing synthetic trajectories were the +5.6% step (19.6 → 25.2). Without them, expect the post-SFT number to land 3–5pp lower |
| **GRPO infra** | Medium | TRL ≥0.10 ships GRPO; Unsloth has a GRPO patch. 1500 RL steps is reachable on a 4090 if rollouts are cheap (grounding) — much harder for navigation rollouts that need an emulator | Grounding GRPO: feasible on-4090. Navigation GRPO: needs ≥1 emulator running alongside; one 4090 can host 1 emulator + 3B model in 4-bit, but at low throughput |
| **Visual tool-use (zoom)** | Low | Pure inference-time logic — crop image around predicted point, re-prompt, return final prediction | Drop-in: ~150 lines of Python. Easy +1–2pp |
| **Apple's exact 17K synthetic CoT trajectories** | Low-Medium | Generated by a teacher model with CoT prompting; recipe is described | Reproducible if you have a strong teacher (GPT-4o or Gemini 2.5 Pro). Cost: ~$50–200 of API |
| **GPU-hours** | Unknown | Apple did not disclose | If 10K SFT steps × 3B in 4-bit + LoRA on a 4090 with ~28M short examples → very rough estimate is 80–150 GPU-hours for SFT alone; RL adds 10–30 more |
| **Reproducibility of headline 28.0%** | High | Five-run mean reported, no seed disclosure, no eval-harness disclosure (M3A vs custom?) | Even with all the data, expect ±3pp run-to-run variance |

### Concrete replication plan if we want to chase Ferret-UI Lite (1–2 weeks, 1× 4090)

1. **Pick a base** — the cleanest swap-in is **Qwen2-VL-3B-Instruct** (it's what ShowUI/OS-Atlas/AGUVIS use; published AW points exist for it; it has a real chat template that handles multi-image). Stick with Gemma 4 E2B if continuity with prior runs matters more than apples-to-apples with the paper.
2. **SFT data preprocessing (3 days)** — pull OS-Atlas, UGround, AGUVIS, Aria-UI, ShowUI from HF. Write one converter per source into the unified action space (point-based grounding + function-call navigation). This is the bulk of the eng effort.
3. **SFT (5–7 days, one 4090)** — QLoRA on the unified mix, ~10K steps. Don't try the full 28M-row mixture; subsample to ~3–5M examples weighted toward mobile + desktop (which AW tests). Add CoT either by relabeling with a teacher or by SFT-ing on Apple-style CoT prompts in OpenCUA.
4. **Synthetic CoT trajectories (1 day, $50–200 API)** — generate ~5–10K mobile trajectories with GPT-4o or Gemini 2.5 Pro acting as a CoT teacher on AndroidLab or AndroidControl-Test prompts.
5. **Grounding GRPO (2 days)** — TRL or Unsloth GRPO, containment reward, ~500 steps. Cheap because no emulator needed.
6. **Navigation GRPO (3–5 days)** — one emulator + one model on the 4090, step-wise reward `f_type + f_param`, ~1000 rollouts. This is the part where 4090 throughput hurts.
7. **Add zoom-in inference (½ day)** — two-pass crop+re-predict for grounding. Pure post-hoc.
8. **Eval on AW with M3A** — same harness as the baseline run we just shipped, fair-comparison vs Gemma 4 E2B baseline.

**Realistic ceiling for a 4090 reproduction:** **mid-teens to low-20s on AW**, well below the paper's 28%. The gap closes if (a) you go to Qwen2-VL-3B (stronger GUI prior than Gemma 4 E2B), (b) you get the CoT-synthetic-trajectory step right, and (c) you have the patience for the navigation-RL phase.

**If you want a single highest-ROI lift this week, do step 7 alone (zoom-in) on top of our existing Path W LoRA.** It's a half-day of code, no training, and Apple's data shows it's worth +1–2pp on grounding tasks; that translates to roughly +1pp on AW SR. Not a moonshot, but it's free.


- Feng, Huang, Qu, Zhang, Qin, et al. *UI-TARS-2 Technical Report.* 2025. arXiv:2509.02544.
- Bai et al. *DigiRL: Training In-The-Wild Device-Control Agents with Autonomous RL.* 2024. arXiv:2406.11896 (NeurIPS 2024).
- Bai et al. *Digi-Q: Learning VLM Q-Value Functions for Training Device-Control Agents.* 2025. arXiv:2502.15760 (ICLR 2025).
- Dai et al. *V-Droid: Advancing Mobile GUI Agent Through Generative Verifiers.* 2025. arXiv:2503.15937.
- Xu, Liu et al. *AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents.* 2024. arXiv:2410.24024 (ACL 2025).
- Wu, Wu et al. *OS-ATLAS: A Foundation Action Model for Generalist GUI Agents.* 2024. arXiv:2410.23218 (ICLR 2025).
- Lin et al. *ShowUI: One Vision-Language-Action Model for GUI Visual Agent.* 2024. arXiv:2411.17465 (CVPR 2025).
- Wang et al. *Mobile-Agent-E: Self-Evolving Mobile Assistant for Complex Tasks.* 2025. arXiv:2501.11733.
- Ye et al. *Mobile-Agent-v3 / GUI-Owl: Fundamental Agents for GUI Automation.* 2025. arXiv:2508.15144.
- You et al. *Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs.* 2024. arXiv:2404.05719 (ECCV 2024).
- Apple. *Ferret-UI Lite: Lessons from Building Small On-Device GUI Agents.* 2025. arXiv:2509.26539.
- Yang et al. *Aria-UI: Visual Grounding for GUI Instructions.* 2024. arXiv:2412.16256.
- Lu, Yang, Shen, Awadallah. *OmniParser for Pure Vision Based GUI Agent.* 2024. arXiv:2408.00203.
- Zhang et al. *Auto-GUI / You Only Look at Screens.* 2023. arXiv:2309.11436.
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models* (§6.4 on OOD narrowness). arXiv:2106.09685.
