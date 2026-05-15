from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from . import CostFunction
from ._shared import _standard_error_stats, _zero
from ..collections import tensors_dict


class EventsCostFunction(CostFunction):
    """Event classification loss over the prediction horizon."""

    heads = ["x", "decoder", "events"]

    @staticmethod
    def _normalize_events_mask(
        events: torch.Tensor,
        mask: torch.Tensor | None,
        multilabel: bool = False,
    ) -> torch.Tensor:
        if mask is None:
            if events.size(-1) == 1 or multilabel:
                mask = torch.ones_like(events)
            else:
                mask = events.sum(dim=-1, keepdim=True).clamp(max=1.0)
        elif mask.dim() == events.dim() - 1:
            mask = mask.unsqueeze(-1)

        if mask.shape != events.shape and mask.shape != events.shape[:-1] + (1,):
            raise ValueError(
                "events_mask must be broadcast-compatible with events. "
                f"Got mask={tuple(mask.shape)}, events={tuple(events.shape)}."
            )

        return mask.expand_as(events) if mask.shape != events.shape else mask

    @staticmethod
    def _make_event_weight(
        event_weight: float | Sequence[float] | torch.Tensor | None,
        events: torch.Tensor,
        multilabel: bool = False,
    ) -> torch.Tensor | None:
        if event_weight is None:
            return None

        weight = torch.as_tensor(event_weight, device=events.device, dtype=events.dtype)
        if events.size(-1) == 1:
            if weight.numel() != 1:
                raise ValueError("Binary event weighting expects a single float for positive events.")
            return weight.reshape(1)
        if multilabel and weight.numel() == 1:
            return weight.reshape(1)

        if weight.numel() != events.size(-1):
            raise ValueError(
                "Event weighting expects either a single positive-event weight for multilabel "
                "events or one weight per event type. "
                f"Got {weight.numel()} weights for {events.size(-1)} event outputs."
            )
        return weight.reshape(*([1] * (events.dim() - 1)), events.size(-1))

    @staticmethod
    def calculate_sample_loss(
        events_hat: torch.Tensor | None,
        events: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        event_weight: float | Sequence[float] | torch.Tensor | None = None,
        multilabel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if events_hat is None or events is None:
            return None, None

        if events_hat.shape != events.shape:
            raise ValueError(
                "events_hat and events must have the same shape. "
                f"Got events_hat={tuple(events_hat.shape)}, events={tuple(events.shape)}."
            )

        mask = EventsCostFunction._normalize_events_mask(events, mask, multilabel=multilabel)

        eps = torch.finfo(events_hat.dtype).eps
        probs = events_hat.clamp(min=eps, max=1.0 - eps)
        weight = EventsCostFunction._make_event_weight(event_weight, events, multilabel=multilabel)

        if events.size(-1) == 1 or multilabel:
            loss = F.binary_cross_entropy(probs, events, reduction="none")
            if weight is not None:
                event_weight_tensor = torch.where(events > 0.5, weight, torch.ones_like(events))
                mask = mask * event_weight_tensor
            sample_loss = (loss * mask).sum(dim=-1)
            sample_weight = mask.sum(dim=-1)
        else:
            class_weight = 1.0 if weight is None else weight
            ce = -(events * class_weight * probs.log()).sum(dim=-1)
            mask_step = mask.any(dim=-1).to(dtype=events.dtype)
            sample_loss = ce * mask_step
            if weight is None:
                sample_weight = mask_step
            else:
                sample_weight = (events * weight).sum(dim=-1) * mask_step

        return sample_loss, sample_weight

    @staticmethod
    def _auc_metric(scores: torch.Tensor, target: torch.Tensor) -> list[float] | None:
        target = target > 0.5
        n_pos = target.sum()
        n_neg = (~target).sum()
        pair_count = n_pos * n_neg
        if pair_count.item() == 0:
            return None

        order = torch.argsort(scores)
        sorted_scores = scores[order]
        sorted_target = target[order]
        ranks = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=scores.dtype)

        _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
        if counts.numel() == scores.numel():
            avg_ranks = ranks
        else:
            ends = counts.cumsum(dim=0).to(dtype=scores.dtype)
            starts = ends - counts.to(dtype=scores.dtype) + 1.0
            avg_by_group = (starts + ends) * 0.5
            avg_ranks = torch.repeat_interleave(avg_by_group, counts)

        rank_sum_pos = avg_ranks[sorted_target].sum()
        n_pos_f = n_pos.to(dtype=scores.dtype)
        pair_count_f = pair_count.to(dtype=scores.dtype)
        auc = (rank_sum_pos - n_pos_f * (n_pos_f + 1.0) * 0.5) / pair_count_f
        return [(auc * pair_count_f).item(), pair_count_f.item()]

    @staticmethod
    def calculate_accuracy_metric(
        events_hat: torch.Tensor | None,
        events: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        multilabel: bool = False,
    ) -> list[float] | None:
        if events_hat is None or events is None:
            return None

        if events_hat.shape != events.shape:
            raise ValueError(
                "events_hat and events must have the same shape. "
                f"Got events_hat={tuple(events_hat.shape)}, events={tuple(events.shape)}."
            )

        with torch.no_grad():
            valid = EventsCostFunction._normalize_events_mask(events, mask, multilabel=multilabel) > 0.5
            if events.size(-1) == 1 or multilabel:
                if not valid.any():
                    return None
                pred = events_hat.detach() >= 0.5
                target = events.detach() > 0.5
                correct = ((pred == target) & valid).sum()
                total = valid.sum()
            else:
                step_valid = valid.any(dim=-1)
                if not step_valid.any():
                    return None
                pred_idx = events_hat.detach().argmax(dim=-1)
                target_idx = events.detach().argmax(dim=-1)
                correct = ((pred_idx == target_idx) & step_valid).sum()
                total = step_valid.sum()

            return [
                correct.to(dtype=events_hat.dtype).item(),
                total.to(dtype=events_hat.dtype).item(),
            ]

    @staticmethod
    def calculate_auc_metric(
        events_hat: torch.Tensor | None,
        events: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        multilabel: bool = False,
    ) -> list[float] | None:
        if events_hat is None or events is None:
            return None

        if events_hat.shape != events.shape:
            raise ValueError(
                "events_hat and events must have the same shape. "
                f"Got events_hat={tuple(events_hat.shape)}, events={tuple(events.shape)}."
            )

        if events.size(-1) != 1 and not multilabel:
            return None

        with torch.no_grad():
            valid = EventsCostFunction._normalize_events_mask(events, mask, multilabel=multilabel) > 0.5
            if not valid.any():
                return None
            return EventsCostFunction._auc_metric(
                events_hat.detach()[valid].reshape(-1),
                events.detach()[valid].reshape(-1),
            )

    @staticmethod
    def _binary_class_metrics(
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        pred = pred & valid
        target = target & valid

        tp = (pred & target).sum()
        fp = (pred & ~target & valid).sum()
        tn = (~pred & ~target & valid).sum()
        fn = (~pred & target & valid).sum()

        metrics: dict[str, Any] = EventsCostFunction._confusion_counts(tp, fp, tn, fn, dtype)
        metrics.update({
            "true": EventsCostFunction._class_metrics_from_counts(tp, fp, tn, fn, dtype),
            "false": EventsCostFunction._class_metrics_from_counts(tn, fn, tp, fp, dtype),
        })
        return metrics

    @staticmethod
    def _sum_metric(value: torch.Tensor) -> list[float | None]:
        return [value.item(), None]

    @staticmethod
    def _confusion_counts(
        tp: torch.Tensor,
        fp: torch.Tensor,
        tn: torch.Tensor,
        fn: torch.Tensor,
        dtype: torch.dtype,
    ) -> dict[str, list[float | None]]:
        tp_f = tp.to(dtype=dtype)
        fp_f = fp.to(dtype=dtype)
        tn_f = tn.to(dtype=dtype)
        fn_f = fn.to(dtype=dtype)
        return {
            "tp": EventsCostFunction._sum_metric(tp_f),
            "fp": EventsCostFunction._sum_metric(fp_f),
            "tn": EventsCostFunction._sum_metric(tn_f),
            "fn": EventsCostFunction._sum_metric(fn_f),
        }

    @staticmethod
    def _class_metrics_from_counts(
        tp: torch.Tensor,
        fp: torch.Tensor,
        tn: torch.Tensor,
        fn: torch.Tensor,
        dtype: torch.dtype,
    ) -> dict[str, list[float]]:
        tp_f = tp.to(dtype=dtype)
        fp_f = fp.to(dtype=dtype)
        fn_f = fn.to(dtype=dtype)

        return {
            "precision": [tp_f.item(), (tp_f + fp_f).item()],
            "recall": [tp_f.item(), (tp_f + fn_f).item()],
            "f1": [(2.0 * tp_f).item(), (2.0 * tp_f + fp_f + fn_f).item()],
        }

    @staticmethod
    def _any_over_observation_dims(tensor: torch.Tensor) -> torch.Tensor:
        """Reduce all observation/horizon dimensions with any, preserving batch and event dims."""
        if tensor.dim() <= 2:
            return tensor
        return tensor.any(dim=tuple(range(1, tensor.dim() - 1)))

    @staticmethod
    def _any_over_non_batch_dims(tensor: torch.Tensor) -> torch.Tensor:
        """Reduce all non-batch dimensions with any, preserving only the batch dimension."""
        if tensor.dim() <= 1:
            return tensor
        return tensor.any(dim=tuple(range(1, tensor.dim())))

    @staticmethod
    def _global_binary_metrics(
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        dtype: torch.dtype,
        preserve_event_dim: bool = False,
    ) -> dict[str, dict[str, list[float]]]:
        reducer = (
            EventsCostFunction._any_over_observation_dims
            if preserve_event_dim
            else EventsCostFunction._any_over_non_batch_dims
        )
        pred_global = reducer(pred & valid)
        target_global = reducer(target & valid)
        valid_global = reducer(valid)
        return EventsCostFunction._binary_class_metrics(pred_global, target_global, valid_global, dtype)

    @staticmethod
    def calculate_event_metrics(
        events_hat: torch.Tensor | None,
        events: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        multilabel: bool = False,
    ) -> dict[str, Any]:
        if events_hat is None or events is None:
            return {}

        if events_hat.shape != events.shape:
            raise ValueError(
                "events_hat and events must have the same shape. "
                f"Got events_hat={tuple(events_hat.shape)}, events={tuple(events.shape)}."
            )

        with torch.no_grad():
            valid = EventsCostFunction._normalize_events_mask(events, mask, multilabel=multilabel) > 0.5
            probs = events_hat.detach()
            target = events.detach() > 0.5
            if events.size(-1) == 1:
                pred = probs >= 0.5

                event_valid = valid[..., 0]
                if not event_valid.any():
                    return {}

                event_metrics: dict[str, Any] = {
                    "timestep": EventsCostFunction._binary_class_metrics(
                        pred[..., 0],
                        target[..., 0],
                        event_valid,
                        probs.dtype,
                    ),
                    "global": EventsCostFunction._global_binary_metrics(
                        pred[..., 0],
                        target[..., 0],
                        event_valid,
                        probs.dtype,
                    ),
                }
                return {"event_0": event_metrics}

            if multilabel:
                pred = probs >= 0.5
                event_valid = valid.any(dim=-1)
                if not event_valid.any():
                    return {}

                event_0: dict[str, Any] = {
                    "timestep": EventsCostFunction._binary_class_metrics(
                        pred.reshape(-1),
                        target.reshape(-1),
                        valid.reshape(-1),
                        probs.dtype,
                    ),
                    "global": EventsCostFunction._global_binary_metrics(
                        pred,
                        target,
                        valid,
                        probs.dtype,
                    ),
                }

                for feature_idx in range(events.size(-1)):
                    feature_valid = valid[..., feature_idx]
                    if not feature_valid.any():
                        continue
                    event_0[f"feature_{feature_idx}"] = {
                        "timestep": EventsCostFunction._binary_class_metrics(
                            pred[..., feature_idx],
                            target[..., feature_idx],
                            feature_valid,
                            probs.dtype,
                        ),
                        "global": EventsCostFunction._global_binary_metrics(
                            pred[..., feature_idx],
                            target[..., feature_idx],
                            feature_valid,
                            probs.dtype,
                        ),
                    }

                return {"event_0": event_0}

            pred_idx = probs.argmax(dim=-1)
            pred = F.one_hot(pred_idx, num_classes=events.size(-1)).to(dtype=torch.bool)

            valid_any = EventsCostFunction._any_over_observation_dims(valid)

            metrics: dict[str, Any] = {}
            if valid_any.any():
                metrics["global"] = EventsCostFunction._global_binary_metrics(
                    pred,
                    target,
                    valid,
                    probs.dtype,
                    preserve_event_dim=True,
                )

            for event_idx in range(events.size(-1)):
                event_valid = valid[..., event_idx]
                if not event_valid.any():
                    continue

                event_metrics: dict[str, Any] = {
                    "timestep": EventsCostFunction._binary_class_metrics(
                        pred[..., event_idx],
                        target[..., event_idx],
                        event_valid,
                        probs.dtype,
                    ),
                }

                event_any_valid = valid_any[..., event_idx]
                if event_any_valid.any():
                    event_metrics["global"] = EventsCostFunction._global_binary_metrics(
                        pred[..., event_idx],
                        target[..., event_idx],
                        event_valid,
                        probs.dtype,
                    )

                metrics[f"event_{event_idx}"] = event_metrics

        return metrics

    @staticmethod
    def calculate_loss(
        events_hat: torch.Tensor | None,
        events: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        event_weight: float | Sequence[float] | torch.Tensor | None = None,
        multilabel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample_loss, sample_weight = EventsCostFunction.calculate_sample_loss(
            events_hat,
            events,
            mask,
            event_weight,
            multilabel=multilabel,
        )
        if sample_loss is None or sample_weight is None:
            device = events_hat.device if events_hat is not None else events.device if events is not None else torch.device("cpu")
            return _zero(device)

        return sample_loss.sum(), sample_weight.sum().clamp(min=1.0)

    def __call__(
        self,
        batch: tensors_dict,
        training: bool,
        event_weight: float | Sequence[float] | torch.Tensor | None = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        if not self._check_zero_return(training):
            return 0.0

        events_hat = batch.get("events_hat")
        events = batch.get("events")
        events_mask = batch.get("events_mask")
        events_multilabel = bool(batch.get("events_multilabel", False))

        sample_loss, sample_weight = self.calculate_sample_loss(
            events_hat,
            events,
            events_mask,
            event_weight,
            multilabel=events_multilabel,
        )
        if sample_loss is None or sample_weight is None:
            device = events_hat.device if events_hat is not None else events.device if events is not None else torch.device("cpu")
            loss_num, loss_div = _zero(device)
        else:
            loss_num = sample_loss.sum()
            loss_div = sample_weight.sum().clamp(min=1.0)
            self.metrics["se"] = _standard_error_stats(sample_loss, sample_weight)
            accuracy = self.calculate_accuracy_metric(events_hat, events, events_mask, multilabel=events_multilabel)
            if accuracy is not None:
                self.metrics["accuracy"] = accuracy
            auc = self.calculate_auc_metric(events_hat, events, events_mask, multilabel=events_multilabel)
            if auc is not None:
                self.metrics["auc"] = auc
            self.metrics["events"] = self.calculate_event_metrics(
                events_hat,
                events,
                events_mask,
                multilabel=events_multilabel,
            )

        loss = loss_num / loss_div.clamp(min=1.0)
        self.loss = loss_num.item()
        self.loss_div = loss_div.item()
        return loss
