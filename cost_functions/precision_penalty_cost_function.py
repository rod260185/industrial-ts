from __future__ import annotations

from typing import Any
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from . import CostFunction
from ._shared import _zero
from .events_cost_function import EventsCostFunction
from ..collections import tensors_dict


class PrecisionPenaltyCostFunction(CostFunction):
    """Soft batch-level precision penalty for event heads."""

    heads = ["x", "decoder", "events"]

    def __init__(
        self,
        target_precision: float | Sequence[float] | torch.Tensor,
        training_ratio: float = 1.0,
        test_ratio: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(training_ratio=training_ratio, test_ratio=test_ratio)
        target = torch.as_tensor(target_precision, dtype=torch.float32)
        if target.numel() == 0:
            raise ValueError("target_precision must not be empty.")
        if not torch.isfinite(target).all() or (target < 0.0).any() or (target > 1.0).any():
            raise ValueError("target_precision values must be finite and in the [0, 1] interval.")
        if target.dim() == 0:
            self.target_precision = float(target.item())
        else:
            self.target_precision = [float(value) for value in target.flatten().tolist()]
        self.eps = float(eps)

    @staticmethod
    def _make_target_precision(
        target_precision: float | Sequence[float] | torch.Tensor,
        events: torch.Tensor,
        multilabel: bool = False,
    ) -> torch.Tensor:
        event_count = events.size(-1)
        target = torch.as_tensor(target_precision, device=events.device, dtype=events.dtype)
        is_vector_target = target.dim() > 0
        if target.numel() == 0:
            raise ValueError("target_precision must not be empty.")
        if not torch.isfinite(target).all() or (target < 0.0).any() or (target > 1.0).any():
            raise ValueError("target_precision values must be finite and in the [0, 1] interval.")

        if event_count == 1:
            if is_vector_target:
                raise ValueError(
                    "Binary BCE precision penalty expects scalar `target_precision`; "
                    "lists/vectors are only valid for multilabel or softmax events."
                )
            return target.reshape(1)

        if not is_vector_target:
            if multilabel:
                return target.reshape(1).expand(event_count)
            mode = "multilabel" if multilabel else "softmax"
            raise ValueError(
                f"{mode} precision penalty expects `target_precision` as a list/vector "
                f"with one target per output. Got scalar target for {event_count} outputs."
            )

        target = target.flatten()
        if target.numel() != event_count:
            mode = "multilabel" if multilabel else "softmax"
            raise ValueError(
                f"{mode} precision penalty expects {event_count} target_precision values. "
                f"Got {target.numel()}."
            )
        return target

    @staticmethod
    def _target_metric(target_precision: float | Sequence[float]) -> float | dict[str, float]:
        if isinstance(target_precision, Sequence) and not isinstance(target_precision, (str, bytes)):
            return {f"target_{idx}": float(value) for idx, value in enumerate(target_precision)}
        return float(target_precision)

    @staticmethod
    def _soft_counts(
        events_hat: torch.Tensor,
        events: torch.Tensor,
        mask: torch.Tensor | None,
        multilabel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if events_hat.shape != events.shape:
            raise ValueError(
                "events_hat and events must have the same shape. "
                f"Got events_hat={tuple(events_hat.shape)}, events={tuple(events.shape)}."
            )

        valid = EventsCostFunction._normalize_events_mask(events, mask, multilabel=multilabel)
        valid = valid.to(dtype=events_hat.dtype)
        target = events.to(dtype=events_hat.dtype).clamp(min=0.0, max=1.0)
        probs = events_hat.clamp(min=0.0, max=1.0)

        reduce_dims = tuple(range(events.dim() - 1))
        predicted_positives = (probs * valid).sum(dim=reduce_dims)
        soft_tp = (probs * target * valid).sum(dim=reduce_dims)
        return soft_tp, predicted_positives

    @staticmethod
    def calculate_loss(
        events_hat: torch.Tensor | None,
        events: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        target_precision: float | Sequence[float] | torch.Tensor = 0.9,
        multilabel: bool = False,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if events_hat is None or events is None:
            device = events_hat.device if events_hat is not None else events.device if events is not None else torch.device("cpu")
            return _zero(device)

        soft_tp, predicted_positives = PrecisionPenaltyCostFunction._soft_counts(
            events_hat,
            events,
            mask,
            multilabel=multilabel,
        )
        active = predicted_positives > 0.0
        if not active.any():
            return _zero(events_hat.device)

        precision = soft_tp / predicted_positives.clamp(min=eps)
        target = PrecisionPenaltyCostFunction._make_target_precision(
            target_precision,
            events,
            multilabel=multilabel,
        ).to(device=events_hat.device, dtype=events_hat.dtype)
        penalty = F.relu(target - precision).pow(2)
        loss_num = penalty[active].sum()
        loss_div = active.to(dtype=events_hat.dtype).sum().clamp(min=1.0)
        return loss_num, loss_div

    @staticmethod
    def calculate_metrics(
        events_hat: torch.Tensor | None,
        events: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        target_precision: float | Sequence[float] | torch.Tensor = 0.9,
        multilabel: bool = False,
        eps: float = 1e-8,
    ) -> dict[str, Any]:
        if events_hat is None or events is None:
            return {}

        with torch.no_grad():
            soft_tp, predicted_positives = PrecisionPenaltyCostFunction._soft_counts(
                events_hat.detach(),
                events.detach(),
                mask.detach() if torch.is_tensor(mask) else mask,
                multilabel=multilabel,
            )
            active = predicted_positives > 0.0
            if not active.any():
                return {}

            precision = soft_tp / predicted_positives.clamp(min=eps)
            target = PrecisionPenaltyCostFunction._make_target_precision(
                target_precision,
                events,
                multilabel=multilabel,
            ).to(device=events_hat.device, dtype=events_hat.dtype)
            penalty = F.relu(target - precision).pow(2)

            metrics: dict[str, Any] = {
                "soft_precision": [soft_tp[active].sum().item(), predicted_positives[active].sum().item()],
                "active_events": active.to(dtype=events_hat.dtype).sum().item(),
            }

            if events.size(-1) == 1:
                metrics["events"] = {
                    "event_0": {
                        "global": {
                            "soft_precision": [soft_tp[0].item(), predicted_positives[0].item()],
                            "penalty": [penalty[0].item(), 1.0],
                        }
                    }
                }
                return metrics

            if multilabel:
                event_0: dict[str, Any] = {
                    "global": {
                        "soft_precision": [soft_tp[active].sum().item(), predicted_positives[active].sum().item()],
                        "penalty": [penalty[active].sum().item(), active.to(dtype=events_hat.dtype).sum().item()],
                    }
                }

                for event_idx in range(events.size(-1)):
                    if not active[event_idx]:
                        continue
                    event_0[f"feature_{event_idx}"] = {
                        "soft_precision": [soft_tp[event_idx].item(), predicted_positives[event_idx].item()],
                        "penalty": [penalty[event_idx].item(), 1.0],
                    }
                metrics["events"] = {"event_0": event_0}
                return metrics

            events_metrics: dict[str, Any] = {}
            for event_idx in range(events.size(-1)):
                if not active[event_idx]:
                    continue
                events_metrics[f"event_{event_idx}"] = {
                    "soft_precision": [soft_tp[event_idx].item(), predicted_positives[event_idx].item()],
                    "penalty": [penalty[event_idx].item(), 1.0],
                }
            metrics["events"] = events_metrics
            return metrics

    def __call__(
        self,
        batch: tensors_dict,
        training: bool,
        *args,
        **kwargs
    ) -> torch.Tensor:
        if not self._check_zero_return(training):
            return 0.0

        events_hat = batch.get("events_hat")
        events = batch.get("events")
        events_mask = batch.get("events_mask")
        events_multilabel = bool(batch.get("events_multilabel", False))

        loss_num, loss_div = self.calculate_loss(
            events_hat,
            events,
            events_mask,
            target_precision=self.target_precision,
            multilabel=events_multilabel,
            eps=self.eps,
        )
        self.metrics.update(
            self.calculate_metrics(
                events_hat,
                events,
                events_mask,
                target_precision=self.target_precision,
                multilabel=events_multilabel,
                eps=self.eps,
            )
        )
        self.metrics["target_precision"] = self._target_metric(self.target_precision)

        loss = loss_num / loss_div.clamp(min=1.0)
        self.loss = loss_num.item()
        self.loss_div = loss_div.item()
        return loss
