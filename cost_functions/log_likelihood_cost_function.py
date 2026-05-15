from __future__ import annotations

import math

import torch


from ._shared import _feature_loss_metrics, _standard_error_stats, _weighted_metric, _zero
from .mse_cost_function import MSECostFunction
from ..collections import tensors_dict


class LogLikelihoodCostFunction(MSECostFunction):
    """L1 variant: Gaussian negative log-likelihood with learned precision."""

    heads=['x','lambda']

    @staticmethod
    def calculate_normalized_learned_precision_metric(
        lam2: torch.Tensor | None,
        mask: torch.Tensor,
        target_shape: torch.Size,
    ) -> list[float] | None:
        if lam2 is None:
            return None
        if lam2.shape != target_shape:
            try:
                lam2 = torch.broadcast_to(lam2, target_shape)
            except RuntimeError as exc:
                raise ValueError(
                    "lambda_hat must be broadcast-compatible with the target tensor. "
                    f"Got lam2.shape={tuple(lam2.shape)}, target_shape={tuple(target_shape)}."
                ) from exc

        valid = mask > 0
        if not valid.any():
            return None

        log_min = math.log(1 / (2 * math.pi))
        log_max = math.log(2 * math.pi)
        eps = torch.finfo(lam2.dtype).eps
        normalized_learned_precision = (torch.log(lam2.clamp(min=eps)) - log_min) / (log_max - log_min)
        normalized_learned_precision = normalized_learned_precision.clamp(min=0.0, max=1.0)

        weights = mask.to(dtype=lam2.dtype)
        return [
            (normalized_learned_precision * weights).sum().item(),
            weights.sum().item(),
        ]

    @staticmethod
    def calculate_normalized_learned_precision_feature_metrics(
        lam2: torch.Tensor | None,
        mask: torch.Tensor,
        target_shape: torch.Size,
    ) -> dict[str, dict[str, list[float]]]:
        if lam2 is None:
            return {}
        if lam2.shape != target_shape:
            try:
                lam2 = torch.broadcast_to(lam2, target_shape)
            except RuntimeError as exc:
                raise ValueError(
                    "lambda_hat must be broadcast-compatible with the target tensor. "
                    f"Got lam2.shape={tuple(lam2.shape)}, target_shape={tuple(target_shape)}."
                ) from exc

        log_min = math.log(1 / (2 * math.pi))
        log_max = math.log(2 * math.pi)
        eps = torch.finfo(lam2.dtype).eps
        normalized = (torch.log(lam2.clamp(min=eps)) - log_min) / (log_max - log_min)
        normalized = normalized.clamp(min=0.0, max=1.0)

        metrics: dict[str, dict[str, list[float]]] = {}
        weights = mask.to(dtype=lam2.dtype)
        for feature_idx in range(target_shape[-1]):
            feature_weight = weights[..., feature_idx]
            if not (feature_weight > 0).any():
                continue
            metrics[f"feature_{feature_idx}"] = {
                "normalized_learned_precision": _weighted_metric(
                    normalized[..., feature_idx] * feature_weight,
                    feature_weight,
                )
            }
        return metrics

    @staticmethod
    def calculate_sample_loss(
        x: torch.Tensor,
        x_hat: torch.Tensor | None,
        mask: torch.Tensor,
        lam2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if x_hat is None or lam2 is None:
            return None, None
        if lam2.shape != x.shape:
            try:
                lam2 = torch.broadcast_to(lam2, x.shape)
            except RuntimeError as exc:
                raise ValueError(
                    "lambda_hat must be broadcast-compatible with x/x_hat. "
                    f"Got lam2.shape={tuple(lam2.shape)}, x.shape={tuple(x.shape)}."
                ) from exc

        err2 = (x_hat - x) ** 2
        nll = 0.5 * (lam2 * err2 - torch.log(lam2) + math.log(2 * math.pi))
        sample_loss = (nll * mask).sum(dim=-1)
        nobs = mask.sum(dim=-1)
        return sample_loss, nobs

    @staticmethod
    def calculate_element_loss(
        x: torch.Tensor,
        x_hat: torch.Tensor | None,
        mask: torch.Tensor,
        lam2: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if x_hat is None or lam2 is None:
            return None
        if lam2.shape != x.shape:
            try:
                lam2 = torch.broadcast_to(lam2, x.shape)
            except RuntimeError as exc:
                raise ValueError(
                    "lambda_hat must be broadcast-compatible with x/x_hat. "
                    f"Got lam2.shape={tuple(lam2.shape)}, x.shape={tuple(x.shape)}."
                ) from exc
        err2 = (x_hat - x) ** 2
        nll = 0.5 * (lam2 * err2 - torch.log(lam2) + math.log(2 * math.pi))
        return nll * mask

    @staticmethod
    def calculate_loss(
        x: torch.Tensor,
        x_hat: torch.Tensor | None,
        mask: torch.Tensor,
        lam2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample_loss, sample_nobs = LogLikelihoodCostFunction.calculate_sample_loss(x, x_hat, mask, lam2)
        if sample_loss is None or sample_nobs is None:
            loss_sum, div = _zero(x.device)
            return loss_sum, div

        return sample_loss.sum(), sample_nobs.clamp(min=1e-8).sum().clamp(min=1.0)

    def __call__(
        self,
        batch: tensors_dict,
        training: bool,
        *args,
        **kwargs
    ) -> torch.Tensor:
        if not self._check_zero_return(training):
            return 0.0
        
        x = batch["x_cost"]
        mask = batch["mask_cost"]
        mask_train = batch["mask_train_cost"]
        mask_err = mask * (1 - mask_train)
            
        x_hat = batch.get('x_hat')            
        lam2 = batch.get('lambda_hat')
        sample_loss, sample_nobs = self.calculate_sample_loss(x, x_hat, mask_err, lam2)
        if sample_loss is None or sample_nobs is None:
            loss_num, loss_div = _zero(x.device)
        else:
            loss_num = sample_loss.sum()
            loss_div = sample_nobs.clamp(min=1e-8).sum().clamp(min=1.0)
            self.metrics["loss"] = _weighted_metric(loss_num, loss_div)
            self.metrics["se"] = _standard_error_stats(sample_loss, sample_nobs)
            normalized_learned_precision = self.calculate_normalized_learned_precision_metric(lam2, mask_err, x.shape)
            if normalized_learned_precision is not None:
                self.metrics["normalized_learned_precision"] = normalized_learned_precision
            element_loss = self.calculate_element_loss(x, x_hat, mask_err, lam2)
            if element_loss is not None:
                feature_metrics = _feature_loss_metrics(element_loss, mask_err)
                precision_features = self.calculate_normalized_learned_precision_feature_metrics(lam2, mask_err, x.shape)
                for feature_name, metrics in precision_features.items():
                    feature_metrics.setdefault(feature_name, {}).update(metrics)
                self.metrics["features"] = feature_metrics
        loss = loss_num / loss_div.clamp(min=1.0)
        self.loss = loss_num.item()
        self.loss_div = loss_div.item()
        return loss
    
class LogLikelihoodCostFunctionDecoder(LogLikelihoodCostFunction):

    heads = ['x','lambda','decoder']

    def __call__(
        self,
        batch: tensors_dict,
        training: bool,
        *args,
        **kwargs
    ) -> torch.Tensor:
        if not self._check_zero_return(training):
            return 0.0
        
        x = batch['head_cost']
        mask_err = batch['mask_head_cost']
            
        x_hat = batch.get('x_hat')            
        lam2 = batch.get('lambda_hat')
        sample_loss, sample_nobs = self.calculate_sample_loss(x, x_hat, mask_err, lam2)
        if sample_loss is None or sample_nobs is None:
            loss_num, loss_div = _zero(x.device)
        else:
            loss_num = sample_loss.sum()
            loss_div = sample_nobs.clamp(min=1e-8).sum().clamp(min=1.0)
            self.metrics["loss"] = _weighted_metric(loss_num, loss_div)
            self.metrics["se"] = _standard_error_stats(sample_loss, sample_nobs)
            normalized_learned_precision = self.calculate_normalized_learned_precision_metric(lam2, mask_err, x.shape)
            if normalized_learned_precision is not None:
                self.metrics["normalized_learned_precision"] = normalized_learned_precision
            element_loss = self.calculate_element_loss(x, x_hat, mask_err, lam2)
            if element_loss is not None:
                feature_metrics = _feature_loss_metrics(element_loss, mask_err)
                precision_features = self.calculate_normalized_learned_precision_feature_metrics(lam2, mask_err, x.shape)
                for feature_name, metrics in precision_features.items():
                    feature_metrics.setdefault(feature_name, {}).update(metrics)
                self.metrics["features"] = feature_metrics
        loss = loss_num / loss_div.clamp(min=1.0)
        self.loss = loss_num.item()
        self.loss_div = loss_div.item()
        return loss
