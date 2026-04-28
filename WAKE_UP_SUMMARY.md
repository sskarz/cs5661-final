# Wake-up summary — Run I (Path W SFT) overnight

## Headline

- **Best checkpoint: `ckpt-1500` at full_match = 0.595**
- Path W zero-shot baseline: 0.515 (no training)
- Coord baseline (HF, all 8 prior LoRA runs failed to clear): 0.288
- Run H best (LoRA SFT, projector unlocked, coord regression): 0.275

- Lift over coord baseline: **+0.307** (+106.6% relative)
- Lift over Run H best: **+0.320**

## Decision tree

- If best ckpt > 0.515 (baseline) → SFT is helping; ship the best checkpoint, optionally try DPO.
- If best ckpt < baseline → SFT is hurting; ship the zero-shot baseline.
- If best ckpt is at oracle ceiling (0.858): we've saturated the projection-center scoring; need element-level metric to gain more.

## Per-checkpoint sweep

| step | full_match | parse | oracle | tap acc | scroll | type | open_app | nav_back |
|---|---|---|---|---|---|---|---|---|
| **baseline** | **0.515** | 1.000 | 0.858 | 0.683 | 0.074 | 0.625 | 0.000 | 1.000 |
| 600 | 0.555 | 0.945 | 0.858 | 0.683 | 0.222 | 0.750 | 0.167 | 1.000 |
| 900 | 0.570 | 0.955 | 0.858 | 0.692 | 0.296 | 0.750 | 0.167 | 1.000 |
| 1200 | 0.540 | 0.955 | 0.858 | 0.700 | 0.111 | 0.750 | 0.000 | 1.000 |
| **1500** | **0.595** | 0.995 | 0.858 | 0.700 | 0.444 | 0.750 | 0.167 | 1.000 |
| 1800 | 0.550 | 0.975 | 0.858 | 0.700 | 0.185 | 0.688 | 0.083 | 1.000 |
| 2100 | 0.510 | 1.000 | 0.858 | 0.675 | 0.037 | 0.625 | 0.083 | 1.000 |
| 2400 | 0.505 | 1.000 | 0.858 | 0.675 | 0.037 | 0.625 | 0.000 | 1.000 |
| 2700 | 0.530 | 0.995 | 0.858 | 0.683 | 0.185 | 0.625 | 0.000 | 1.000 |
| 3000 | 0.535 | 1.000 | 0.858 | 0.692 | 0.185 | 0.625 | 0.000 | 1.000 |
| 3300 | 0.520 | 0.995 | 0.858 | 0.700 | 0.037 | 0.625 | 0.000 | 1.000 |
| 3600 | 0.510 | 1.000 | 0.858 | 0.667 | 0.037 | 0.688 | 0.000 | 1.000 |
| 3900 | 0.480 | 0.995 | 0.858 | 0.633 | 0.000 | 0.625 | 0.000 | 1.000 |
| 4200 | 0.505 | 1.000 | 0.858 | 0.667 | 0.000 | 0.625 | 0.000 | 1.000 |
| 4500 | 0.500 | 0.995 | 0.858 | 0.658 | 0.037 | 0.625 | 0.000 | 1.000 |
| 4800 | 0.465 | 1.000 | 0.858 | 0.608 | 0.000 | 0.625 | 0.083 | 1.000 |
| 5100 | 0.480 | 0.995 | 0.858 | 0.633 | 0.000 | 0.562 | 0.083 | 1.000 |
| 5400 | 0.490 | 1.000 | 0.858 | 0.642 | 0.000 | 0.562 | 0.083 | 1.000 |
| 5700 | 0.495 | 1.000 | 0.858 | 0.650 | 0.000 | 0.625 | 0.000 | 1.000 |
| 6000 | 0.485 | 1.000 | 0.858 | 0.633 | 0.000 | 0.688 | 0.000 | 1.000 |
| 6300 | 0.500 | 1.000 | 0.858 | 0.667 | 0.000 | 0.625 | 0.000 | 1.000 |
| 6600 | 0.510 | 0.995 | 0.858 | 0.675 | 0.000 | 0.688 | 0.000 | 1.000 |
| 6900 | 0.495 | 0.990 | 0.858 | 0.658 | 0.000 | 0.688 | 0.000 | 1.000 |
| 7200 | 0.505 | 1.000 | 0.858 | 0.650 | 0.000 | 0.688 | 0.083 | 1.000 |
| 7500 | 0.505 | 1.000 | 0.858 | 0.658 | 0.000 | 0.688 | 0.000 | 1.000 |
| 7800 | 0.505 | 1.000 | 0.858 | 0.658 | 0.000 | 0.688 | 0.000 | 1.000 |
| 8100 | 0.510 | 1.000 | 0.858 | 0.667 | 0.000 | 0.688 | 0.000 | 1.000 |
| 8400 | 0.505 | 1.000 | 0.858 | 0.658 | 0.000 | 0.688 | 0.000 | 1.000 |
| 8700 | 0.495 | 1.000 | 0.858 | 0.650 | 0.000 | 0.625 | 0.000 | 1.000 |
| 9000 | 0.500 | 1.000 | 0.858 | 0.658 | 0.000 | 0.625 | 0.000 | 1.000 |
| 9132 | 0.500 | 1.000 | 0.858 | 0.650 | 0.000 | 0.688 | 0.000 | 1.000 |

## Failure analysis

**Best checkpoint: ckpt-1500, full_match = 0.595**

Confusion matrix (rows = GT action, cols = predicted):

| GT \ pred | <parse_fail> | action_not_found | launch_app | long_press | navigate_back | open_app | scroll | scroll down | scroll_up | tap | type |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **navigate_back** | 0 | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| **open_app** | 0 | 1 | 1 | 0 | 1 | 2 | 0 | 0 | 0 | 7 | 0 |
| **scroll** | 0 | 0 | 0 | 0 | 0 | 0 | 19 | 4 | 3 | 1 | 0 |
| **tap** | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 118 | 1 |
| **type** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 15 |
| **wait** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 13 | 2 |

Of wrong tap predictions: **0/36 (0.0%) were 'near misses'** (picked a different element whose bbox center is within 0.14 of GT). The rest (36) were structurally wrong picks.

Per-action-type lift over zero-shot baseline (best ckpt):

| action | n | baseline acc | best ckpt acc | Δ |
|---|---|---|---|---|
| wait | 16 | 0.000 | 0.000 | ↑ +0.000 |
| tap | 120 | 0.683 | 0.700 | ↑ +0.017 |
| scroll | 27 | 0.074 | 0.444 | ↑ +0.370 |
| type | 16 | 0.625 | 0.750 | ↑ +0.125 |
| open_app | 12 | 0.000 | 0.167 | ↑ +0.167 |
| navigate_back | 9 | 1.000 | 1.000 | ↑ +0.000 |

## Files

- Run I checkpoints: `outputs/gemma4-e2b-pathW-lora-runI/`
- Eval JSONs: `outputs/eval/runI_ckpt*.json`
- Best-ckpt full predictions (with all_predictions): `outputs/eval/runI_ckpt1500.json`
- Path W baseline: `outputs/eval/native_baseline.json`

## Suggested next step

- Path W SFT cleared zero-shot. Pick best checkpoint, run on full test set (8,217 rows, ~30 min), publish the number. Consider DPO if time permits.
