"""Collator wrapper that adds coordinate-aware loss metadata to vision batches.

For each tap-action row, parses the GT (x, y) from the assistant JSON, locates
the digit-token positions in the labels tensor, and adds metadata tensors that
the CoordAwareSFTTrainer consumes to compute a screen-distance auxiliary loss.

The "last two digit-or-dot token runs" heuristic is robust to SentencePiece
context-dependent merging (e.g., '",' fused into one token, ' "' fused, etc.).
For a tap action the assistant message ends in '... "x": V, "y": W}' so the
last two digit runs in labels are always the GT x and y values, in order.

See plan: /home/sanskar/.claude/plans/sunny-coalescing-book.md
"""

from __future__ import annotations

import json
import re

import torch


X_VALUE_RE = re.compile(r'"x":\s*(-?\d*\.?\d+)')
Y_VALUE_RE = re.compile(r'"y":\s*(-?\d*\.?\d+)')


def compute_place_values(num_str: str) -> list[float]:
    """[(digit_char, place_value)] for digit chars in num_str. Decimal point skipped.

    "0.293" -> [1.0, 0.1, 0.01, 0.001]
    "0.1625" -> [1.0, 0.1, 0.01, 0.001, 0.0001]
    """
    if "." in num_str:
        int_part, dec_part = num_str.split(".", 1)
    else:
        int_part, dec_part = num_str, ""
    pvs: list[float] = []
    Li = len(int_part)
    for k, ch in enumerate(int_part):
        if ch.isdigit():
            pvs.append(10.0 ** (Li - 1 - k))
    for k, ch in enumerate(dec_part):
        if ch.isdigit():
            pvs.append(10.0 ** -(k + 1))
    return pvs


def _digit_chars_only(num_str: str) -> list[str]:
    return [c for c in num_str if c.isdigit()]


class CoordAwareCollator:
    """Wrap a base UnslothVisionDataCollator. Add coord-loss metadata for tap rows.

    Adds these batch keys (consumed by CoordAwareSFTTrainer):
      coord_is_tap         [B] bool   — True iff row is a tap with both coords located
      coord_gt_x           [B] f32    — GT x in [0, 1]
      coord_gt_y           [B] f32    — GT y in [0, 1]
      coord_x_pos          [B, max_digits] long  — label-tensor positions of x's digit tokens; -1 padding
      coord_x_place        [B, max_digits] f32   — place value per digit (1.0, 0.1, 0.01, ...); 0 padding
      coord_y_pos          [B, max_digits] long  — same for y
      coord_y_place        [B, max_digits] f32   — same for y
      coord_action_weight  [B] f32    — per-row action-type weight; defaults to 1.0
    """

    def __init__(
        self,
        base_collator,
        tokenizer,
        max_digits: int = 8,
        action_weights: dict[str, float] | None = None,
    ):
        self.base = base_collator
        self.tok = tokenizer
        self.max_digits = max_digits
        self.action_weights = action_weights  # None = all 1.0 (back-compat)
        self._validated = False
        self.tap_count = 0
        self.skip_count = 0
        self.first_batch_reported = False

    def _validate_tokenizer(self) -> None:
        # Each digit + decimal point should be a single token, with consistent ids.
        for ch in "0123456789.":
            ids = self.tok.encode(ch, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(
                    f"[coord] Tokenizer split {ch!r} into {len(ids)} tokens (expected 1). "
                    f"Coord-aware loss requires single-token digits."
                )
        standalone = {d: self.tok.encode(d, add_special_tokens=False)[0] for d in "0123456789"}
        for frag in (', "x": 0.5,', ', "y": 0.293}', '{"x": 0.1625}'):
            ids = self.tok.encode(frag, add_special_tokens=False)
            for j, tid in enumerate(ids):
                dec = self.tok.decode([tid]).strip()
                if len(dec) == 1 and dec in "0123456789":
                    if tid != standalone[dec]:
                        raise RuntimeError(
                            f"[coord] Digit {dec!r} has inconsistent IDs across contexts: "
                            f"standalone={standalone[dec]} but in {frag!r} at pos {j} got {tid}. "
                            f"This tokenizer needs per-position digit IDs (not implemented)."
                        )
        self._validated = True

    # ------------------------------------------------------------------ collate
    def __call__(self, examples):
        if not self._validated:
            self._validate_tokenizer()

        batch = self.base(examples)
        if not isinstance(batch, dict):
            batch = dict(batch)

        labels = batch["labels"]
        B = labels.shape[0]

        is_tap = torch.zeros(B, dtype=torch.bool)
        gt_x = torch.zeros(B, dtype=torch.float32)
        gt_y = torch.zeros(B, dtype=torch.float32)
        x_pos = torch.full((B, self.max_digits), -1, dtype=torch.long)
        x_place = torch.zeros((B, self.max_digits), dtype=torch.float32)
        y_pos = torch.full((B, self.max_digits), -1, dtype=torch.long)
        y_place = torch.zeros((B, self.max_digits), dtype=torch.float32)
        action_weight = torch.ones(B, dtype=torch.float32)

        if B != len(examples):
            print(
                f"[coord] WARN: collated batch B={B} != len(examples)={len(examples)}; "
                f"disabling coord meta for this batch."
            )
        else:
            for i, ex in enumerate(examples):
                self._process_row(
                    i, ex, labels, is_tap, gt_x, gt_y, x_pos, x_place, y_pos, y_place
                )
                if self.action_weights is not None:
                    a = self._extract_action_type(ex)
                    if a is not None:
                        action_weight[i] = float(self.action_weights.get(a, 1.0))

        batch["coord_is_tap"] = is_tap
        batch["coord_gt_x"] = gt_x
        batch["coord_gt_y"] = gt_y
        batch["coord_x_pos"] = x_pos
        batch["coord_x_place"] = x_place
        batch["coord_y_pos"] = y_pos
        batch["coord_y_place"] = y_place
        batch["coord_action_weight"] = action_weight

        if not self.first_batch_reported:
            self.first_batch_reported = True
            mapped = int(is_tap.sum().item())
            print(
                f"[coord] first batch: tap_count={self.tap_count} mapped={mapped} "
                f"skipped={self.skip_count}"
            )
            if self.tap_count >= 4 and self.skip_count / max(self.tap_count, 1) > 0.20:
                raise RuntimeError(
                    f"[coord] Skip rate too high ({self.skip_count}/{self.tap_count}). "
                    f"Aborting."
                )

        return batch

    # -------------------------------------------------------- per-row processing
    def _process_row(
        self,
        i: int,
        ex: dict,
        labels: torch.Tensor,
        is_tap: torch.Tensor,
        gt_x: torch.Tensor,
        gt_y: torch.Tensor,
        x_pos: torch.Tensor,
        x_place: torch.Tensor,
        y_pos: torch.Tensor,
        y_place: torch.Tensor,
    ) -> None:
        assistant_text = self._extract_assistant_text(ex)
        if assistant_text is None:
            return
        try:
            obj = json.loads(assistant_text)
        except (json.JSONDecodeError, ValueError):
            return
        if obj.get("action") != "tap":
            return

        m_x = X_VALUE_RE.search(assistant_text)
        m_y = Y_VALUE_RE.search(assistant_text)
        if not m_x or not m_y:
            return
        x_str = m_x.group(1)
        y_str = m_y.group(1)
        try:
            x_val = float(x_str)
            y_val = float(y_str)
        except ValueError:
            return
        if not (0.0 <= x_val <= 1.0 and 0.0 <= y_val <= 1.0):
            return

        self.tap_count += 1

        # Find the last two digit-or-dot runs in the labels.
        runs = self._find_digit_runs(labels[i])
        if len(runs) < 2:
            self.skip_count += 1
            return
        x_run, y_run = runs[-2], runs[-1]

        # Verify the digit chars (skipping dots) match the parsed GT digit chars.
        x_run_digits = [c for _, c in x_run if c.isdigit()]
        y_run_digits = [c for _, c in y_run if c.isdigit()]
        if x_run_digits != _digit_chars_only(x_str) or y_run_digits != _digit_chars_only(y_str):
            self.skip_count += 1
            return

        x_pvs = compute_place_values(x_str)
        y_pvs = compute_place_values(y_str)
        x_locs = [p for p, c in x_run if c.isdigit()]
        y_locs = [p for p, c in y_run if c.isdigit()]
        if len(x_locs) != len(x_pvs) or len(y_locs) != len(y_pvs):
            self.skip_count += 1
            return

        n_x = min(len(x_locs), self.max_digits)
        n_y = min(len(y_locs), self.max_digits)
        is_tap[i] = True
        gt_x[i] = x_val
        gt_y[i] = y_val
        x_pos[i, :n_x] = torch.tensor(x_locs[:n_x], dtype=torch.long)
        x_place[i, :n_x] = torch.tensor(x_pvs[:n_x], dtype=torch.float32)
        y_pos[i, :n_y] = torch.tensor(y_locs[:n_y], dtype=torch.long)
        y_place[i, :n_y] = torch.tensor(y_pvs[:n_y], dtype=torch.float32)

    # --------------------------------------------------------------- digit runs
    def _find_digit_runs(self, labels_row: torch.Tensor) -> list[list[tuple[int, str]]]:
        """Return all maximal contiguous runs of single-char digit/dot tokens in labels.

        Each run is a list of (label_position, digit_char) tuples. Tokens with
        id < 0 (e.g., -100 from train_on_responses_only masking) break runs.
        Runs containing no digit chars (e.g., a stray '.') are filtered out.
        """
        labels_list = labels_row.tolist()
        runs: list[list[tuple[int, str]]] = []
        cur: list[tuple[int, str]] = []
        for j, tid in enumerate(labels_list):
            if tid < 0:
                if cur:
                    runs.append(cur)
                    cur = []
                continue
            decoded = self.tok.decode([tid]).strip()
            if len(decoded) == 1 and (decoded.isdigit() or decoded == "."):
                cur.append((j, decoded))
            else:
                if cur:
                    runs.append(cur)
                    cur = []
        if cur:
            runs.append(cur)
        return [r for r in runs if any(c.isdigit() for _, c in r)]

    @staticmethod
    def _extract_assistant_text(ex: dict) -> str | None:
        for msg in ex.get("messages", []):
            if msg.get("role") == "assistant":
                for c in msg.get("content", []):
                    if c.get("type") == "text":
                        return c["text"]
        return None

    @classmethod
    def _extract_action_type(cls, ex: dict) -> str | None:
        txt = cls._extract_assistant_text(ex)
        if txt is None:
            return None
        try:
            obj = json.loads(txt)
        except (json.JSONDecodeError, ValueError):
            return None
        a = obj.get("action")
        return a if isinstance(a, str) else None
