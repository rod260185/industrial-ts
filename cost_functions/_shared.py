from __future__ import annotations

import torch

from ..collections import tensors_dict


def _ensure_batch(batch: tensors_dict) -> tensors_dict:
    if not isinstance(batch, tensors_dict):
        raise TypeError(f"batch must be tensors_dict, got {type(batch).__name__}.")
    return batch


def _zero(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)


def _standard_error_stats(
    sample_loss_sum: torch.Tensor,
    sample_weight: torch.Tensor,
) -> dict[str, float | str]:
    sample_loss_sum = sample_loss_sum.detach().reshape(-1).float()
    sample_weight = sample_weight.detach().reshape(-1).float()

    if sample_loss_sum.shape != sample_weight.shape:
        raise ValueError(
            "sample_loss_sum and sample_weight must have the same shape "
            f"(got {tuple(sample_loss_sum.shape)} and {tuple(sample_weight.shape)})."
        )

    valid = sample_weight > 0
    if not valid.any():
        return {
            "kind": "standard_error",
            "loss_sum": 0.0,
            "loss_div": 0.0,
            "wm2": 0.0,
            "count": 0.0,
        }

    sample_loss_sum = sample_loss_sum[valid]
    sample_weight = sample_weight[valid]
    sample_mean = sample_loss_sum / sample_weight

    return {
        "kind": "standard_error",
        "loss_sum": float(sample_loss_sum.sum().item()),
        "loss_div": float(sample_weight.sum().item()),
        "wm2": float((sample_weight * sample_mean.pow(2)).sum().item()),
        "count": float(valid.sum().item()),
    }


def _weighted_metric(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> list[float]:
    return [
        float(numerator.detach().sum().item()),
        float(denominator.detach().sum().item()),
    ]


def _feature_loss_metrics(
    element_loss_sum: torch.Tensor,
    element_weight: torch.Tensor,
) -> dict[str, dict[str, list[float] | dict[str, float | str]]]:
    if element_loss_sum.shape != element_weight.shape:
        try:
            element_weight = torch.broadcast_to(element_weight, element_loss_sum.shape)
        except RuntimeError as exc:
            raise ValueError(
                "element_weight must be broadcast-compatible with element_loss_sum. "
                f"Got element_weight={tuple(element_weight.shape)}, "
                f"element_loss_sum={tuple(element_loss_sum.shape)}."
            ) from exc

    if element_loss_sum.dim() == 0:
        return {}

    metrics: dict[str, dict[str, list[float] | dict[str, float | str]]] = {}
    for feature_idx in range(element_loss_sum.size(-1)):
        feature_loss = element_loss_sum[..., feature_idx]
        feature_weight = element_weight[..., feature_idx]
        if not (feature_weight > 0).any():
            continue

        metrics[f"feature_{feature_idx}"] = {
            "loss": _weighted_metric(feature_loss, feature_weight),
            "se": _standard_error_stats(feature_loss, feature_weight),
        }
    return metrics
