# Autoresearch ideas backlog (pathZ smoke)

## High-priority recipe variations
- **Lower LR (1e-4)**: 200/500 steps damaged open_app at 2e-4. A gentler LR
  may preserve base capabilities while still learning M3A schema.
- **Higher LR (3e-4)**: pushes more strongly toward the response distribution
  per step; with cosine decay, may fix grounding faster.
- **Disable projector training**: Run L's "projector matters" came from full-
  scale training. At smoke scale, unfreezing the projector may inject too
  much variance.
- **Lower lora_r (8)**: smaller capacity may force concentration on action
  schema instead of memorization.

## Data variations
- **Substantive synthetic Reason** (build a sentence that uses goal text and
  a justification, not just "Click element 5"). Risk: model overfits to AC-
  test goals.
- **No Reason at all** (label = `Action: {...}`). Tests whether reason is
  hurting rather than helping at this scale.
- **Resample to fix open_app underrep** (currently 6.6% in train, 6% in eval
  — consistent — but boost to 15% to compensate for AC's bias).

## Eval variations
- **Stratified eval scoring** (weight per-action-type so click doesn't
  dominate the headline).
- **Loose match for input_text** (ignore index, just check action_type +
  text length similarity). Currently input_text full=0% which kills the
  headline number even when type and rough text match.

## Architectural changes (only if simple variations cap)
- **Constrained decoding** (outlines/llguidance) on the eval to prevent
  any non-schema action_type from being scored at all. Inference-time only.
- **Longer training data** (4K rows from the full 73K AC-train).
- **Add long_press / status / answer synthetic rows** to teach actions AC
  doesn't have.
