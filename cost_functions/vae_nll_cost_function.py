from __future__ import annotations

import math

import torch

from . import CostFunction
from ._shared import _feature_loss_metrics, _standard_error_stats, _weighted_metric, _zero
from ..collections import tensors_dict


_GAUSSIAN_CENTRAL_90_Z = 1.6448536269514722


class NELBOCostFunction(CostFunction):
    """L5: VAE reconstruction + KL divergence (x branch)."""

    heads = ['vae_x']

    @staticmethod
    def apply_nll_function(x: torch.Tensor, vae_x: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * (logvar + ((x - vae_x) ** 2) / torch.exp(logvar) + math.log(2 * math.pi))

    @classmethod
    def calculate_reconstruction_element_loss(
        cls,
        x: torch.Tensor,
        mask: torch.Tensor,
        vae_x: torch.Tensor | None,
        vae_logvar_obs: torch.Tensor | None,
        *,
        sigma_temp: float = 1.0,
    ) -> torch.Tensor | None:
        if vae_x is None:
            return None
        if vae_logvar_obs is not None:
            logvar_obs = (vae_logvar_obs + math.log(sigma_temp)).clamp(min=-5.0, max=5.0)
            element_loss = cls.apply_nll_function(x, vae_x, logvar_obs)
        else:
            element_loss = (x - vae_x).pow(2)
        return element_loss * mask

    @classmethod
    def calculate_loss_components(
        cls,
        x: torch.Tensor,
        mask: torch.Tensor,
        vae_x: torch.Tensor | None,
        vae_mu: torch.Tensor | None,
        vae_logvar: torch.Tensor | None,
        vae_logvar_obs: torch.Tensor | None,
        *,
        sigma_temp: float = 1.0,
        kl_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[None, None, None, None]:
        if vae_x is None or vae_mu is None or vae_logvar is None:
            return None, None, None, None

        recon_element_loss = cls.calculate_reconstruction_element_loss(
            x,
            mask,
            vae_x,
            vae_logvar_obs,
            sigma_temp=sigma_temp,
        )
        if recon_element_loss is None:
            return None, None, None, None

        recon_num = recon_element_loss.sum()
        recon_div = mask.sum().clamp(min=1.0)
        if kl_scale > 0.0:
            kl_num = -0.5 * torch.sum(1 + vae_logvar - vae_mu.pow(2) - vae_logvar.exp())
            kl_div = torch.tensor(float(vae_logvar.numel()), device=x.device).clamp(min=1.0)
        else:
            kl_num = torch.tensor(0.0, device=x.device)
            kl_div = torch.tensor(1.0, device=x.device)
        return recon_num, recon_div, kl_num, kl_div

    @classmethod
    def calculate_sample_loss(
        cls,
        x: torch.Tensor,
        mask: torch.Tensor,
        vae_x: torch.Tensor | None,
        vae_mu: torch.Tensor | None,
        vae_logvar: torch.Tensor | None,
        vae_logvar_obs: torch.Tensor | None,
        *,
        sigma_temp: float = 1.0,
        kl_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if vae_x is None or vae_mu is None or vae_logvar is None:
            return None, None

        recon_element_loss = cls.calculate_reconstruction_element_loss(
            x,
            mask,
            vae_x,
            vae_logvar_obs,
            sigma_temp=sigma_temp,
        )
        if recon_element_loss is None:
            return None, None
        recon_num = recon_element_loss.sum(dim=(1, 2))

        recon_div = mask.sum(dim=(1, 2)).clamp(min=1.0)
        reduce_dims = tuple(range(1, vae_logvar.ndim))
        kl_num = -0.5 * torch.sum(1 + vae_logvar - vae_mu.pow(2) - vae_logvar.exp(), dim=reduce_dims)
        kl_div = torch.full_like(recon_div, float(vae_logvar[0].numel()))
        sample_loss = (recon_num / recon_div) + kl_scale * (kl_num / kl_div.clamp(min=1.0))
        sample_weight = torch.ones_like(sample_loss)
        return sample_loss, sample_weight

    @classmethod
    def calculate_loss(
        cls,
        x: torch.Tensor,
        mask: torch.Tensor,
        vae_x: torch.Tensor | None,
        vae_mu: torch.Tensor | None,
        vae_logvar: torch.Tensor | None,
        vae_logvar_obs: torch.Tensor | None,
        *,
        sigma_temp: float = 1.0,
        kl_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if vae_x is None or vae_mu is None or vae_logvar is None:
            loss_sum, div = _zero(x.device)
            return loss_sum, div

        vae_recon, vae_recon_div, vae_kl, vae_kl_div = cls.calculate_loss_components(
            x,
            mask,
            vae_x,
            vae_mu,
            vae_logvar,
            vae_logvar_obs,
            sigma_temp=sigma_temp,
            kl_scale=kl_scale,
        )
        if vae_recon is None or vae_recon_div is None or vae_kl is None or vae_kl_div is None:
            loss_sum, div = _zero(x.device)
            return loss_sum, div

        loss_sum = vae_recon / vae_recon_div + kl_scale * (vae_kl / vae_kl_div)
        div = torch.tensor(1.0, device=x.device)
        return loss_sum, div

    @staticmethod
    def calculate_coverage_90(
        x: torch.Tensor,
        mask: torch.Tensor,
        vae_x: torch.Tensor | None,
        vae_logvar_obs: torch.Tensor | None,
        *,
        sigma_temp: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if vae_x is None or vae_logvar_obs is None:
            return None, None

        logvar_obs = (vae_logvar_obs + math.log(sigma_temp)).clamp(min=-5.0, max=5.0)
        std_obs = torch.exp(0.5 * logvar_obs)
        covered = ((x - vae_x).abs() <= _GAUSSIAN_CENTRAL_90_Z * std_obs).to(dtype=mask.dtype)
        return (covered * mask).sum(), mask.sum()

    @staticmethod
    def calculate_coverage_90_feature_metrics(
        x: torch.Tensor,
        mask: torch.Tensor,
        vae_x: torch.Tensor | None,
        vae_logvar_obs: torch.Tensor | None,
        *,
        sigma_temp: float = 1.0,
    ) -> dict[str, dict[str, list[float]]]:
        if vae_x is None or vae_logvar_obs is None:
            return {}

        logvar_obs = (vae_logvar_obs + math.log(sigma_temp)).clamp(min=-5.0, max=5.0)
        std_obs = torch.exp(0.5 * logvar_obs)
        covered = ((x - vae_x).abs() <= _GAUSSIAN_CENTRAL_90_Z * std_obs).to(dtype=mask.dtype)

        metrics: dict[str, dict[str, list[float]]] = {}
        for feature_idx in range(x.size(-1)):
            feature_mask = mask[..., feature_idx]
            if not (feature_mask > 0).any():
                continue
            metrics[f"feature_{feature_idx}"] = {
                "cov <90": _weighted_metric(
                    covered[..., feature_idx] * feature_mask,
                    feature_mask,
                )
            }
        return metrics

    @staticmethod
    def calculate_width_90(
        mask: torch.Tensor,
        vae_logvar_obs: torch.Tensor | None,
        *,
        sigma_temp: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if vae_logvar_obs is None:
            return None, None

        logvar_obs = (vae_logvar_obs + math.log(sigma_temp)).clamp(min=-5.0, max=5.0)
        std_obs = torch.exp(0.5 * logvar_obs)
        width = 2.0 * _GAUSSIAN_CENTRAL_90_Z * std_obs
        return (width * mask).sum(), mask.sum()

    @staticmethod
    def calculate_width_90_feature_metrics(
        mask: torch.Tensor,
        vae_logvar_obs: torch.Tensor | None,
        *,
        sigma_temp: float = 1.0,
    ) -> dict[str, dict[str, list[float]]]:
        if vae_logvar_obs is None:
            return {}

        logvar_obs = (vae_logvar_obs + math.log(sigma_temp)).clamp(min=-5.0, max=5.0)
        std_obs = torch.exp(0.5 * logvar_obs)
        width = 2.0 * _GAUSSIAN_CENTRAL_90_Z * std_obs

        metrics: dict[str, dict[str, list[float]]] = {}
        for feature_idx in range(mask.size(-1)):
            feature_mask = mask[..., feature_idx]
            if not (feature_mask > 0).any():
                continue
            metrics[f"feature_{feature_idx}"] = {
                "width 90": _weighted_metric(
                    width[..., feature_idx] * feature_mask,
                    feature_mask,
                )
            }
        return metrics

    def _add_nll_metrics(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        vae_x: torch.Tensor | None,
        vae_mu: torch.Tensor | None,
        vae_logvar: torch.Tensor | None,
        vae_logvar_obs: torch.Tensor | None,
        *,
        sigma_temp: float = 1.0,
    ) -> None:
        recon_num, recon_div, _, _ = self.calculate_loss_components(
            x,
            mask,
            vae_x,
            vae_mu,
            vae_logvar,
            vae_logvar_obs,
            sigma_temp=sigma_temp,
            kl_scale=0.0,
        )
        if recon_num is not None and recon_div is not None:
            reconstruction_loss = _weighted_metric(recon_num, recon_div)
            self.metrics["loss"] = reconstruction_loss
            self.metrics["reconstruction_loss"] = reconstruction_loss

        recon_element_loss = self.calculate_reconstruction_element_loss(
            x,
            mask,
            vae_x,
            vae_logvar_obs,
            sigma_temp=sigma_temp,
        )
        if recon_element_loss is not None:
            self.metrics["features"] = _feature_loss_metrics(recon_element_loss, mask)

        coverage_num, coverage_div = self.calculate_coverage_90(
            x,
            mask,
            vae_x,
            vae_logvar_obs,
            sigma_temp=sigma_temp,
        )
        if coverage_num is not None and coverage_div is not None:
            self.metrics["cov <90"] = [coverage_num.item(), coverage_div.item()]

        width_num, width_div = self.calculate_width_90(
            mask,
            vae_logvar_obs,
            sigma_temp=sigma_temp,
        )
        if width_num is not None and width_div is not None:
            self.metrics["width 90"] = [width_num.item(), width_div.item()]

        feature_metrics = self.metrics.setdefault("features", {})
        coverage_features = self.calculate_coverage_90_feature_metrics(
            x,
            mask,
            vae_x,
            vae_logvar_obs,
            sigma_temp=sigma_temp,
        )
        width_features = self.calculate_width_90_feature_metrics(
            mask,
            vae_logvar_obs,
            sigma_temp=sigma_temp,
        )
        for feature_name, metrics in coverage_features.items():
            feature_metrics.setdefault(feature_name, {}).update(metrics)
        for feature_name, metrics in width_features.items():
            feature_metrics.setdefault(feature_name, {}).update(metrics)
        if not feature_metrics:
            self.metrics.pop("features", None)

    def __call__(
        self,
        batch: tensors_dict,
        training: bool,
        sigma_temp: float = 1.0,
        kl_scale: float = 1.0,
        *args,
        **kwargs
    ) -> torch.Tensor:
        if not self._check_zero_return(training):
            return 0.0
        x = batch["x_cost"]
        mask = batch["mask_cost"]
        mask_train = batch["mask_train_cost"]
        mask_err = mask * (1 - mask_train)
        vae_x = batch.get("vae_x")
        vae_mu = batch.get("vae_mu")
        vae_logvar = batch.get("vae_logvar")
        vae_logvar_obs = batch.get("vae_logvar_obs")

        loss_num, loss_div = self.calculate_loss(
            x,
            mask_err,
            vae_x,
            vae_mu,
            vae_logvar,
            vae_logvar_obs,
            sigma_temp=sigma_temp,
            kl_scale=kl_scale,
        )
        sample_loss, sample_weight = self.calculate_sample_loss(
            x,
            mask_err,
            vae_x,
            vae_mu,
            vae_logvar,
            vae_logvar_obs,
            sigma_temp=sigma_temp,
            kl_scale=kl_scale,
        )
        if sample_loss is not None and sample_weight is not None:
            self.metrics["se"] = _standard_error_stats(sample_loss, sample_weight)
        loss = loss_num / loss_div.clamp(min=1.0)
        self.loss = loss_num.item()
        self.loss_div = loss_div.item()
        return loss


class NELBOCostFunctionDecoder(NELBOCostFunction):
    """Decoder NELBO over the prediction horizon."""

    heads = ['decoder', 'vae_x']

    def __call__(
        self,
        batch: tensors_dict,
        training: bool,
        sigma_temp: float = 1.0,
        kl_scale: float = 1.0,
        *args,
        **kwargs
    ) -> torch.Tensor:
        if not self._check_zero_return(training):
            return 0.0

        x = batch["head_cost"]
        mask = batch["mask_head_cost"]
        vae_x = batch.get("vae_x")
        vae_mu = batch.get("vae_mu")
        vae_logvar = batch.get("vae_logvar")
        vae_logvar_obs = batch.get("vae_logvar_obs")

        loss_num, loss_div = self.calculate_loss(
            x,
            mask,
            vae_x,
            vae_mu,
            vae_logvar,
            vae_logvar_obs,
            sigma_temp=sigma_temp,
            kl_scale=kl_scale,
        )
        sample_loss, sample_weight = self.calculate_sample_loss(
            x,
            mask,
            vae_x,
            vae_mu,
            vae_logvar,
            vae_logvar_obs,
            sigma_temp=sigma_temp,
            kl_scale=kl_scale,
        )
        if sample_loss is not None and sample_weight is not None:
            self.metrics["se"] = _standard_error_stats(sample_loss, sample_weight)
        loss = loss_num / loss_div.clamp(min=1.0)
        self.loss = loss_num.item()
        self.loss_div = loss_div.item()
        return loss

class NLLCostFunction(NELBOCostFunction):

    heads = ['vae_x']

    def __call__(
        self,
        batch: tensors_dict,
        training: bool,
        sigma_temp: float = 1.0,
        kl_scale: float = 1.0,
        *args,
        **kwargs
    ) -> torch.Tensor:
        loss = super().__call__(
            batch,
            training,
            sigma_temp=sigma_temp,
            kl_scale=0.0,  # Disable KL divergence for pure NLL
        )
        if self.valid_cost_function:
            self._add_nll_metrics(
                batch["x_cost"],
                batch["mask_cost"] * (1 - batch["mask_train_cost"]),
                batch.get("vae_x"),
                batch.get("vae_mu"),
                batch.get("vae_logvar"),
                batch.get("vae_logvar_obs"),
                sigma_temp=sigma_temp,
            )
        return loss

class NLLCostFunctionDecoder(NELBOCostFunctionDecoder):

    heads = ['decoder', 'vae_x']

    def __call__(
        self,
        batch: tensors_dict,
        training: bool,
        sigma_temp: float = 1.0,
        kl_scale: float = 1.0,
        *args,
        **kwargs
    ) -> torch.Tensor:
        loss = super().__call__(
            batch,
            training,
            sigma_temp=sigma_temp,
            kl_scale=0.0,  # Disable KL divergence for pure NLL
        )
        if self.valid_cost_function:
            self._add_nll_metrics(
                batch["head_cost"],
                batch["mask_head_cost"],
                batch.get("vae_x"),
                batch.get("vae_mu"),
                batch.get("vae_logvar"),
                batch.get("vae_logvar_obs"),
                sigma_temp=sigma_temp,
            )
        return loss
