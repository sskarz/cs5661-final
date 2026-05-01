# Autoresearch Dashboard: pathZ-sft-smoke

**Runs:** 6 | **Kept:** 3 | **Discarded:** 3 | **Crashed:** 0
**Segment 1 baseline:** full_match: 20.50% (#3, max_new=384)
**Best:** full_match: 22.00% (#6, +1.50)

## Segment 0 (max_new=128 eval)

| # | commit | full_match | type_match | parse_pct | status | description |
|---|--------|-----------|------------|-----------|--------|-------------|
| 1 | ce09f38 | 20.00% | 51.50% | 92.50% | keep | baseline @ max_new=128 |
| 2 | ce09f38 | 18.50% (-1.5) | 44.00% | 78.50% | discard | 200-step QLoRA, lr=2e-4, projector on; max_new=128 truncation |

## Segment 1 (max_new=384 eval)

| # | commit | full_match | type_match | parse_pct | status | description |
|---|--------|-----------|------------|-----------|--------|-------------|
| 3 | 12e0490 | 20.50% | 54.50% | 100.00% | keep | baseline @ max_new=384 (segment floor) |
| 4 | 12e0490 | 18.50% (-2.0) | 54.50% | 98.50% | discard | 200-step QLoRA, natural dist; open_app/navigate_back regressed |
| 5 | 12e0490 | 13.50% (-7.0) | 48.50% | 99.50% | discard | 500-step QLoRA, natural dist; class-collapse to click (67% of preds) |
| 6 | e41a94c | **22.00% (+1.5)** | 50.00% | 98.50% | **KEEP** | 300-step QLoRA on **balanced** data (250/cls × 6 cls); first positive |
