from __future__ import annotations

import torch

from . import CostFunction
from ._shared import _feature_loss_metrics, _standard_error_stats, _weighted_metric, _zero
from ..collections import tensors_dict


class NoisePredictionCostFunction(CostFunction):
    """L2: diffusion noise prediction MSE."""

    heads=['noise']

    @staticmethod
    def calculate_sample_loss(
        state: torch.Tensor,
        noise: torch.Tensor | None,
        noise_hat: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if noise is None or noise_hat is None:
            return None, None

        sample_loss = ((noise - noise_hat) ** 2).sum(dim=-1)
        sample_weight = torch.ones_like(noise,device=noise.device).sum(dim=-1)
        return sample_loss, sample_weight

    @staticmethod
    def calculate_element_loss(
        noise: torch.Tensor | None,
        noise_hat: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if noise is None or noise_hat is None:
            return None, None
        element_loss = (noise - noise_hat) ** 2
        element_weight = torch.ones_like(element_loss, device=element_loss.device)
        return element_loss, element_weight

    @staticmethod
    def calculate_loss(
        state: torch.Tensor,
        noise: torch.Tensor | None,
        noise_hat: torch.Tensor | None,
        m_t: torch.Tensor | None = None,
        mask_train: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample_loss, sample_weight = NoisePredictionCostFunction.calculate_sample_loss(
            state,
            noise,
            noise_hat,
        )
        if sample_loss is None or sample_weight is None:
            loss_sum, div = _zero(state.device)
            return loss_sum, div

        return sample_loss.sum(), sample_weight.sum()

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
        noise = batch.get("noise")
        noise_hat = batch.get("noise_hat")

        sample_loss, sample_weight = self.calculate_sample_loss(state, noise, noise_hat)
        if sample_loss is None or sample_weight is None:
            loss_num, loss_div = _zero(state.device)
        else:
            loss_num = sample_loss.sum()
            loss_div = sample_weight.sum().clamp(min=1.0)
            self.metrics["loss"] = _weighted_metric(loss_num, loss_div)
            self.metrics["se"] = _standard_error_stats(sample_loss, sample_weight)
            element_loss, element_weight = self.calculate_element_loss(noise, noise_hat)
            if element_loss is not None and element_weight is not None:
                self.metrics["features"] = _feature_loss_metrics(element_loss, element_weight)
        loss = loss_num / loss_div.clamp(min=1.0)
        self.loss = loss_num.item()
        self.loss_div = loss_div.item()
        return loss
