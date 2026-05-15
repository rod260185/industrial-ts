from __future__ import annotations

import torch
import torch.nn.functional as F

from . import CostFunction
from ._shared import _standard_error_stats
from ..collections import tensors_dict


class MaskBCECostFunction(CostFunction):
    """L4: binary cross-entropy for timestep observability mask."""

    heads=['miss']

    @staticmethod
    def calculate_sample_loss(
        state: torch.Tensor,
        miss_hat: torch.Tensor,
        mask_train: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask_train is None:
            mask_train = mask
        if mask_train is None:
            raise KeyError("MaskBCECostFunction expects 'mask_train' (or fallback 'mask') in batch.")

        sample_loss = F.binary_cross_entropy_with_logits(
            miss_hat,
            mask_train,
            reduction="none",
        ).sum(dim=-1)
        sample_weight = state.new_full(sample_loss.shape, float(miss_hat.size(-1)))
        return sample_loss, sample_weight

    @staticmethod
    def calculate_loss(
        state: torch.Tensor,
        miss_hat: torch.Tensor,
        mask_train: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample_loss, sample_weight = MaskBCECostFunction.calculate_sample_loss(state, miss_hat, mask_train, mask)
        return sample_loss.sum(), sample_weight.sum().clamp(min=1.0)

    def __call__(
        self,
        batch: tensors_dict,
        training: bool,
        *args,
        **kwargs
    ) -> torch.Tensor:
        if not self._check_zero_return(training):
            return 0.0
        state = batch["state"]
        mask_train = batch.get("mask_train")
        mask = batch.get("mask")
        miss_hat = batch['miss_hat']

        sample_loss, sample_weight = self.calculate_sample_loss(state, miss_hat, mask_train, mask)
        loss_num = sample_loss.sum()
        loss_div = sample_weight.sum().clamp(min=1.0)
        self.metrics["se"] = _standard_error_stats(sample_loss, sample_weight)
        loss = loss_num / loss_div.clamp(min=1.0)
        self.loss = loss_num.item()
        self.loss_div = loss_div.item()
        return loss
