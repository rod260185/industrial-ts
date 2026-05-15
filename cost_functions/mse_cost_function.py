from __future__ import annotations

import torch

from . import CostFunction
from ._shared import _feature_loss_metrics, _standard_error_stats, _weighted_metric, _zero
from ..collections import tensors_dict


class MSECostFunction(CostFunction):
    """L1: reconstruction MSE on observed (and unmasked) values."""

    heads = ['x']

    @staticmethod
    def calculate_sample_loss(
        x: torch.Tensor,
        x_hat: torch.Tensor | None,
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if x_hat is None:
            return None, None
        sse = (((x_hat - x) ** 2) * mask).sum(dim=-1)
        nobs = mask.sum(dim=-1)
        return sse, nobs

    @staticmethod
    def calculate_element_loss(
        x: torch.Tensor,
        x_hat: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if x_hat is None:
            return None
        return ((x_hat - x) ** 2) * mask

    @staticmethod
    def calculate_loss(
        x: torch.Tensor,
        x_hat: torch.Tensor | None,
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample_sse, sample_nobs = MSECostFunction.calculate_sample_loss(x, x_hat, mask)
        if sample_sse is None or sample_nobs is None:
            loss_sum, div = _zero(x.device)
            return (loss_sum), (div)
        return sample_sse.sum(), sample_nobs.clamp(min=1e-8).sum()

    def __call__(self, batch: tensors_dict, training: bool,
        *args,
        **kwargs) -> torch.Tensor:
        if not self._check_zero_return(training):
            return 0.0
        
        x = batch["x_cost"]
        mask = batch["mask_cost"]
        mask_train = batch["mask_train_cost"]
        x_hat = batch.get("x_hat")
        mask_err = mask * (1 - mask_train)

        sample_sse, sample_nobs = self.calculate_sample_loss(x, x_hat, mask_err)
        if sample_sse is None or sample_nobs is None:
            loss_num, loss_div = _zero(x.device)
        else:
            loss_num = sample_sse.sum()
            loss_div = sample_nobs.clamp(min=1e-8).sum()
            self.metrics["loss"] = _weighted_metric(loss_num, loss_div)
            self.metrics["se"] = _standard_error_stats(sample_sse, sample_nobs)
            element_loss = self.calculate_element_loss(x, x_hat, mask_err)
            if element_loss is not None:
                self.metrics["features"] = _feature_loss_metrics(element_loss, mask_err)
        loss = loss_num / loss_div.clamp(min=1.0)
        self.loss = loss_num.item()
        self.loss_div = loss_div.item()
        return loss

class MSECostFunctionDecoder(MSECostFunction):

    heads = ['x', 'decoder']

    def __call__(self, batch: tensors_dict, training: bool,
        *args,
        **kwargs) -> torch.Tensor:
        if not self._check_zero_return(training):
            return 0.0
    
        x = batch['head_cost']
        mask_err = batch['mask_head_cost']
        x_hat = batch.get('x_hat')
        sample_sse, sample_nobs = self.calculate_sample_loss(x, x_hat, mask_err)
        if sample_sse is None or sample_nobs is None:
            loss_num, loss_div = _zero(x.device)
        else:
            loss_num = sample_sse.sum()
            loss_div = sample_nobs.clamp(min=1e-8).sum()
            self.metrics["loss"] = _weighted_metric(loss_num, loss_div)
            self.metrics["se"] = _standard_error_stats(sample_sse, sample_nobs)
            element_loss = self.calculate_element_loss(x, x_hat, mask_err)
            if element_loss is not None:
                self.metrics["features"] = _feature_loss_metrics(element_loss, mask_err)
        loss = loss_num / loss_div.clamp(min=1.0)
        self.loss = loss_num.item()
        self.loss_div = loss_div.item()
        return loss
