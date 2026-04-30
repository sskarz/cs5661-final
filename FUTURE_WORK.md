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

## References
- Qin et al. *UI-TARS: Pioneering Automated GUI Interaction with Native Agents.* 2025. arXiv:2501.12326.
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
