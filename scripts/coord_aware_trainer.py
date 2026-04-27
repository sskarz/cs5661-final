"""SFTTrainer subclass that adds a coordinate-aware auxiliary loss term.

For tap rows, recovers the predicted (x, y) by soft-expected-value reconstruction
over the digit-token logits and adds Huber(||pred - GT||) to the standard CE loss.

See plan: /home/sanskar/.claude/plans/sunny-coalescing-book.md
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from trl import SFTTrainer


class CoordAwareSFTTrainer(SFTTrainer):
    """SFTTrainer with auxiliary coordinate Huber loss on tap rows.

    Optionally applies per-row action-type weights to both the CE term
    (manually computed in this case) and the coord Huber term.
    """

    def __init__(
        self,
        *args,
        coord_loss_weight: float = 1.0,
        huber_delta: float = 0.05,
        use_sample_weights: bool = False,
        digit_validation: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.coord_loss_weight = coord_loss_weight
        self.huber_delta = huber_delta
        self.use_sample_weights = use_sample_weights
        # Pre-build CPU caches at init; move to device lazily on first use.
        tok = self.processing_class
        self._digit_token_ids_cpu: torch.Tensor | None = None
        self._digit_values_cpu: torch.Tensor | None = None
        if digit_validation:
            ids: list[int] = []
            for d in "0123456789":
                tids = tok.encode(d, add_special_tokens=False)
                if len(tids) != 1:
                    raise RuntimeError(
                        f"[coord] Digit {d!r} encodes to {tids}; expected single token. Aborting."
                    )
                ids.append(tids[0])
            self._digit_token_ids_cpu = torch.tensor(ids, dtype=torch.long)
            self._digit_values_cpu = torch.arange(10, dtype=torch.float32)
        self._digit_token_ids: torch.Tensor | None = None  # set on first compute_loss
        self._digit_values: torch.Tensor | None = None
        self._coord_metrics_acc = {
            "loss_sum": 0.0,
            "active_sum": 0,
            "n": 0,
            "weighted_ce_sum": 0.0,
            "mean_weight_sum": 0.0,
        }

    # ----------------------------------------------------------- digit-id cache
    def _ensure_digit_ids(self, device) -> None:
        if self._digit_token_ids_cpu is None:
            return  # validation skipped (e.g. discrete encoding)
        if self._digit_token_ids is not None and self._digit_token_ids.device == device:
            return
        self._digit_token_ids = self._digit_token_ids_cpu.to(device)
        self._digit_values = self._digit_values_cpu.to(device)

    # ------------------------------------------------------------- compute_loss
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Pop coord metadata before forward — model rejects unknown kwargs.
        coord_meta = {
            k: inputs.pop(k) for k in [k for k in inputs if k.startswith("coord_")]
        }

        # Unsloth's train() wrapper calls FastVisionModel.for_training() at the start
        # of training, which sets UNSLOTH_RETURN_LOGITS=0 (so the model's lm_head is
        # skipped during forward to save memory). We need raw logits for coord loss
        # AND for manual per-row CE under sample weighting; override on every step.
        import os
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

        # Per-row weights default to all-1.0 if collator didn't attach them.
        if self.use_sample_weights and "coord_action_weight" in coord_meta:
            ce_loss, outputs, mean_w = self._weighted_ce_loss(
                model, inputs, coord_meta["coord_action_weight"]
            )
            self._coord_metrics_acc["weighted_ce_sum"] += float(ce_loss.detach())
            self._coord_metrics_acc["mean_weight_sum"] += float(mean_w)
        else:
            ce_loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )

        if self.coord_loss_weight == 0.0 or not coord_meta:
            # Still record so weighted-CE metric logs in the non-coord path.
            self._record_coord_metric(0.0, 0)
            return (ce_loss, outputs) if return_outputs else ce_loss

        is_tap = coord_meta["coord_is_tap"]
        if not is_tap.any():
            self._record_coord_metric(0.0, 0)
            return (ce_loss, outputs) if return_outputs else ce_loss

        logits = outputs.logits
        if logits is None:
            # Some loss types skip logits; fall back to plain CE.
            return (ce_loss, outputs) if return_outputs else ce_loss
        if self._digit_token_ids_cpu is None:
            # Coord-aware aux loss requires digit cache; if validation was disabled
            # (e.g. discrete coord encoding), aux loss is unsupported — skip silently.
            return (ce_loss, outputs) if return_outputs else ce_loss
        # Get device from a normal tensor we control. (`logits.device` can return
        # a wrapped/lazy attribute under some Unsloth memory-offload paths.)
        if "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
            device = inputs["input_ids"].device
        else:
            device = next(model.parameters()).device
        self._ensure_digit_ids(device)

        is_tap = is_tap.to(device)
        gt_x = coord_meta["coord_gt_x"].to(device)
        gt_y = coord_meta["coord_gt_y"].to(device)
        x_pos = coord_meta["coord_x_pos"].to(device)
        x_place = coord_meta["coord_x_place"].to(device)
        y_pos = coord_meta["coord_y_pos"].to(device)
        y_place = coord_meta["coord_y_place"].to(device)

        pred_x = self._soft_value(logits, x_pos, x_place)
        pred_y = self._soft_value(logits, y_pos, y_place)

        dx = pred_x - gt_x
        dy = pred_y - gt_y
        dist = torch.sqrt(dx * dx + dy * dy + 1e-12)

        d = self.huber_delta
        huber = torch.where(
            dist < d,
            0.5 * dist.pow(2) / d,
            dist - 0.5 * d,
        )
        active = is_tap.float()
        # Apply per-row weights to the coord loss too, matching the CE rebalancing.
        if self.use_sample_weights and "coord_action_weight" in coord_meta:
            row_w = coord_meta["coord_action_weight"].to(device).float()
            weighted_active = active * row_w
            denom = weighted_active.sum().clamp(min=1.0)
            coord_loss = (huber * weighted_active).sum() / denom
        else:
            n_active = active.sum().clamp(min=1.0)
            coord_loss = (huber * active).sum() / n_active

        total = ce_loss + self.coord_loss_weight * coord_loss
        self._record_coord_metric(coord_loss.detach().float().item(), int(is_tap.sum().item()))
        return (total, outputs) if return_outputs else total

    # --------------------------------------------------- weighted per-row CE
    def _weighted_ce_loss(self, model, inputs, action_weight):
        """Manual per-row CE with per-sequence action-type weighting.

        Returns (loss, outputs, mean_weight). Behaves like the standard
        masked CE in SFTTrainer when all weights == 1.0, so the no-weights
        path remains unchanged structurally.
        """
        outputs = model(**inputs, return_dict=True)
        logits = outputs.logits
        labels = inputs["labels"]
        if logits is None:
            raise RuntimeError(
                "[coord] use_sample_weights=True requires logits, but model returned None. "
                "Check UNSLOTH_RETURN_LOGITS=1."
            )
        # Causal-LM shift.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        B = shift_labels.size(0)
        device = shift_labels.device
        weights = action_weight.to(device=device, dtype=torch.float32)

        # ignore_index=-100 returns 0 at masked positions; reduction='none' keeps shape.
        flat_ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)).float(),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view(B, -1)
        mask = (shift_labels != -100).float()
        per_row_ce = (flat_ce * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        # Skip rows with zero unmasked tokens (shouldn't happen with response masking
        # but defensive — they otherwise contribute 0 to the weighted mean anyway).
        valid = mask.sum(dim=1) > 0
        if not valid.any():
            return logits.new_zeros(()), outputs, 1.0
        weighted = per_row_ce[valid] * weights[valid]
        loss = weighted.mean()
        mean_w = float(weights[valid].mean().item())
        return loss, outputs, mean_w

    # ---------------------------------------------- soft expected-value reconstruction
    def _soft_value(self, logits: torch.Tensor, pos: torch.Tensor, place: torch.Tensor) -> torch.Tensor:
        """Compute predicted scalar value from digit-token logits.

        pos: [B, K] long (label positions of each digit). -1 for unused.
        place: [B, K] float (place value per digit). 0 for unused.

        Off-by-one: in causal LM, logits[..., t, :] predicts the token at t+1.
        So the prediction for the digit at label position k comes from logits[..., k-1, :].

        Returns [B] float predicted value (sum over digit positions).
        """
        B, K = pos.shape
        valid = pos >= 0
        gather_idx = (pos - 1).clamp(min=0)  # [B, K]
        idx = gather_idx.unsqueeze(-1).expand(-1, -1, logits.size(-1))  # [B, K, V]
        gathered = torch.gather(logits, 1, idx)  # [B, K, V]
        # bf16 has ~3 sig figs; 0.001 * digit math needs fp32.
        gathered = gathered.float()
        probs = gathered.softmax(dim=-1)
        digit_probs = probs.index_select(-1, self._digit_token_ids)  # [B, K, 10]
        exp_digit = (digit_probs * self._digit_values).sum(dim=-1)  # [B, K]
        contrib = exp_digit * place * valid.float()
        return contrib.sum(dim=1)

    # ---------------------------------------------------------------- log hooks
    def _record_coord_metric(self, coord_loss_val: float, active_count: int) -> None:
        self._coord_metrics_acc["loss_sum"] += coord_loss_val
        self._coord_metrics_acc["active_sum"] += active_count
        self._coord_metrics_acc["n"] += 1

    def log(self, logs, *args, **kwargs):
        n = self._coord_metrics_acc["n"]
        if n > 0:
            logs["coord_loss"] = self._coord_metrics_acc["loss_sum"] / n
            logs["coord_active"] = self._coord_metrics_acc["active_sum"] / n
            if self.use_sample_weights:
                logs["weighted_ce"] = self._coord_metrics_acc["weighted_ce_sum"] / n
                logs["mean_row_weight"] = self._coord_metrics_acc["mean_weight_sum"] / n
            self._coord_metrics_acc = {
                "loss_sum": 0.0,
                "active_sum": 0,
                "n": 0,
                "weighted_ce_sum": 0.0,
                "mean_weight_sum": 0.0,
            }
        return super().log(logs, *args, **kwargs)
