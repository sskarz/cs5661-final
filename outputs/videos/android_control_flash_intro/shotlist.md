# Android Control Flash Intro Shot List

Runtime target: 92 seconds, 1920x1080, energetic synth bed, captions baked in.

1. 0-8s: Hook with kinetic title, glowing phone silhouettes, "Can a small VLM learn to control Android apps?"
2. 8-17s: Problem setup: screenshot + UI tree + goal flow into structured action JSON.
3. 17-27s: Baseline pain: raw coordinate target, invalid/weak actions, loop warnings.
4. 27-36s: Training pivot: replace exact pixels with accessibility element ID menu.
5. 36-46s: Data cards: 73K train rows, 8,217 test rows, Gemma 4 E2B + QLoRA, RTX 4090, 0.58% trainable.
6. 46-57s: Metric reveal: zero-shot 0.3935 to Run L 0.5311, +13.76 pp, +35% relative.
7. 57-66s: Ablation chain: I 49.3, J 47.9, K 52.3, L 53.1 with animated bars.
8. 66-78s: Real AW demo split screen using extracted pickle screenshots: baseline/loop-like trace vs trained RecipeDeleteSingleRecipe trace.
9. 78-86s: AndroidWorld reality check: AW-116 live multi-step result 8.70%, progress off hard-zero floor.
10. 86-92s: Final takeaway: "Element-native grounding works. Multi-step agency needs trajectory training."
