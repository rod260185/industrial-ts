from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from . import CostFunction
from ._shared import _zero
from .events_cost_function import EventsCostFunction
from ..collections import tensors_dict


class _NEventMetricCostFunction(CostFunction):
    """Negative event metric cost with soft training and optional hard evaluation."""

    heads = ["x", "decoder", "events"]
    average = "micro"
    metric_name = "metric"
    negative_metric_name = "nmetric"
    denominator = "actual"

    def __init__(
        self,
        training_ratio: float = 1.0,
        test_ratio: float = 1.0,
        hard_eval: bool = True,
        global_metric: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(training_ratio=training_ratio, test_ratio=test_ratio)
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        self.hard_eval = bool(hard_eval)
        self.global_metric = bool(global_metric)
        self.eps = float(eps)

    @staticmethod
    def _sum_observations(tensor: torch.Tensor, events_ndim: int) -> torch.Tensor:
        reduce_dims = tuple(range(events_ndim - 1))
        counts = tensor.sum(dim=reduce_dims) if reduce_dims else tensor
        return counts.reshape(-1)

    @staticmethod
    def _counts(
        events_hat: torch.Tensor,
        events: torch.Tensor,
        mask: torch.Tensor | None,
        multilabel: bool = False,
        hard: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if events_hat.shape != events.shape:
            raise ValueError(
                "events_hat and events must have the same shape. "
                f"Got events_hat={tuple(events_hat.shape)}, events={tuple(events.shape)}."
            )

        valid = EventsCostFunction._normalize_events_mask(events, mask, multilabel=multilabel)
        if hard:
            if events.size(-1) == 1 or multilabel:
                valid_bool = valid > 0.5
                pred_bool = events_hat >= 0.5
                target_bool = events > 0.5
            else:
                valid_bool = valid > 0.5
                target_bool = events > 0.5
                pred_idx = events_hat.argmax(dim=-1)
                pred_bool = F.one_hot(pred_idx, num_classes=events.size(-1)).to(dtype=torch.bool)

            tp = (pred_bool & target_bool & valid_bool).to(dtype=events_hat.dtype)
            fp = (pred_bool & ~target_bool & valid_bool).to(dtype=events_hat.dtype)
            fn = (~pred_bool & target_bool & valid_bool).to(dtype=events_hat.dtype)
            tn = (~pred_bool & ~target_bool & valid_bool).to(dtype=events_hat.dtype)
        else:
            valid_float = valid.to(dtype=events_hat.dtype)
            target = events.to(dtype=events_hat.dtype).clamp(min=0.0, max=1.0)
            probs = events_hat.clamp(min=0.0, max=1.0)

            tp = probs * target * valid_float
            fp = probs * (1.0 - target) * valid_float
            fn = (1.0 - probs) * target * valid_float
            tn = (1.0 - probs) * (1.0 - target) * valid_float

        return (
            _NEventMetricCostFunction._sum_observations(tp, events.dim()),
            _NEventMetricCostFunction._sum_observations(fp, events.dim()),
            _NEventMetricCostFunction._sum_observations(fn, events.dim()),
            _NEventMetricCostFunction._sum_observations(tn, events.dim()),
        )

    @staticmethod
    def _observation_dims(events: torch.Tensor) -> tuple[int, ...]:
        return tuple(range(1, events.dim() - 1))

    @staticmethod
    def _non_batch_dims(events: torch.Tensor) -> tuple[int, ...]:
        return tuple(range(1, events.dim()))

    @staticmethod
    def _any_observed(tensor: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
        for dim in sorted(dims, reverse=True):
            tensor = tensor.any(dim=dim)
        return tensor

    @staticmethod
    def _soft_any_observed(
        probs: torch.Tensor,
        valid: torch.Tensor,
        dims: tuple[int, ...],
    ) -> torch.Tensor:
        probs = (probs * valid).clamp(min=0.0, max=1.0)
        if not dims:
            return probs
        complement = (1.0 - probs).clamp(min=0.0, max=1.0)
        for dim in sorted(dims, reverse=True):
            complement = complement.prod(dim=dim)
        return 1.0 - complement

    @staticmethod
    def _counts_global(
        events_hat: torch.Tensor,
        events: torch.Tensor,
        mask: torch.Tensor | None,
        multilabel: bool = False,
        hard: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if events_hat.shape != events.shape:
            raise ValueError(
                "events_hat and events must have the same shape. "
                f"Got events_hat={tuple(events_hat.shape)}, events={tuple(events.shape)}."
            )

        valid = EventsCostFunction._normalize_events_mask(events, mask, multilabel=multilabel)
        valid_bool = valid > 0.5

        if events.size(-1) == 1 or multilabel:
            dims = _NEventMetricCostFunction._non_batch_dims(events)
            valid_global = _NEventMetricCostFunction._any_observed(valid_bool, dims)
            target = _NEventMetricCostFunction._any_observed((events > 0.5) & valid_bool, dims).to(dtype=events_hat.dtype)

            if hard:
                pred = _NEventMetricCostFunction._any_observed((events_hat >= 0.5) & valid_bool, dims).to(dtype=events_hat.dtype)
            else:
                pred = _NEventMetricCostFunction._soft_any_observed(
                    events_hat.clamp(min=0.0, max=1.0),
                    valid.to(dtype=events_hat.dtype),
                    dims,
                )

            valid_float = valid_global.to(dtype=events_hat.dtype)
            tp = pred * target * valid_float
            fp = pred * (1.0 - target) * valid_float
            fn = (1.0 - pred) * target * valid_float
            tn = (1.0 - pred) * (1.0 - target) * valid_float
            return (
                tp.sum().reshape(1),
                fp.sum().reshape(1),
                fn.sum().reshape(1),
                tn.sum().reshape(1),
            )

        dims = _NEventMetricCostFunction._observation_dims(events)
        valid_global = _NEventMetricCostFunction._any_observed(valid_bool, dims)
        target = _NEventMetricCostFunction._any_observed((events > 0.5) & valid_bool, dims).to(dtype=events_hat.dtype)

        if hard:
            pred_idx = events_hat.argmax(dim=-1)
            pred_step = F.one_hot(pred_idx, num_classes=events.size(-1)).to(dtype=torch.bool)
            pred = _NEventMetricCostFunction._any_observed(pred_step & valid_bool, dims).to(dtype=events_hat.dtype)
        else:
            pred = _NEventMetricCostFunction._soft_any_observed(
                events_hat.clamp(min=0.0, max=1.0),
                valid.to(dtype=events_hat.dtype),
                dims,
            )

        valid_float = valid_global.to(dtype=events_hat.dtype)
        tp = pred * target * valid_float
        fp = pred * (1.0 - target) * valid_float
        fn = (1.0 - pred) * target * valid_float
        tn = (1.0 - pred) * (1.0 - target) * valid_float
        return (
            _NEventMetricCostFunction._sum_observations(tp, target.dim()),
            _NEventMetricCostFunction._sum_observations(fp, target.dim()),
            _NEventMetricCostFunction._sum_observations(fn, target.dim()),
            _NEventMetricCostFunction._sum_observations(tn, target.dim()),
        )

    @classmethod
    def _denominator_from_counts(
        cls,
        tp: torch.Tensor,
        fp: torch.Tensor,
        fn: torch.Tensor,
        tn: torch.Tensor,
    ) -> torch.Tensor:
        if cls.denominator == "actual":
            return tp + fn
        if cls.denominator == "predicted":
            return tp + fp
        raise ValueError(f"Unknown denominator={cls.denominator!r}.")

    @classmethod
    def _metric_numerator_denominator_from_counts(
        cls,
        tp: torch.Tensor,
        fp: torch.Tensor,
        fn: torch.Tensor,
        tn: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return tp, cls._denominator_from_counts(tp, fp, fn, tn)

    @classmethod
    def _loss_from_counts(
        cls,
        tp: torch.Tensor,
        fp: torch.Tensor,
        fn: torch.Tensor,
        tn: torch.Tensor,
        average: str,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        numerator, denom = cls._metric_numerator_denominator_from_counts(tp, fp, fn, tn)
        if average == "micro":
            total_num = numerator.sum()
            total_den = denom.sum()
            if total_den <= 0.0:
                return total_num * 0.0, total_den * 0.0
            return -total_num, total_den.clamp(min=eps)

        if average == "macro":
            if denom.numel() == 0:
                return _zero(tp.device)
            metric = torch.where(
                denom > 0.0,
                numerator / denom.clamp(min=eps),
                torch.zeros_like(denom),
            )
            loss_num = -metric.sum()
            loss_div = tp.new_tensor(float(denom.numel())).clamp(min=1.0)
            return loss_num, loss_div

        raise ValueError(f"Unknown average={average!r}. Expected 'micro' or 'macro'.")

    @classmethod
    def calculate_loss(
        cls,
        events_hat: torch.Tensor | None,
        events: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        multilabel: bool = False,
        hard: bool = False,
        global_metric: bool = False,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if events_hat is None or events is None:
            device = (
                events_hat.device
                if events_hat is not None
                else events.device
                if events is not None
                else torch.device("cpu")
            )
            return _zero(device)

        counter = cls._counts_global if global_metric else cls._counts
        tp, fp, fn, tn = counter(
            events_hat,
            events,
            mask,
            multilabel=multilabel,
            hard=hard,
        )
        return cls._loss_from_counts(tp, fp, fn, tn, average=cls.average, eps=eps)

    @classmethod
    def _metrics_from_counts(
        cls,
        tp: torch.Tensor,
        fp: torch.Tensor,
        fn: torch.Tensor,
        tn: torch.Tensor,
        hard: bool,
        global_metric: bool = False,
        eps: float = 1e-8,
    ) -> dict[str, Any]:
        mode = "hard" if hard else "soft"
        scope = "global_" if global_metric else ""
        numerator, denom = cls._metric_numerator_denominator_from_counts(tp, fp, fn, tn)
        micro_num = numerator.sum()
        micro_den = denom.sum()

        metrics: dict[str, Any] = {}
        if micro_den > 0.0:
            metrics[f"{mode}_{scope}{cls.metric_name}_micro"] = [micro_num.item(), micro_den.item()]
            metrics[f"{mode}_{scope}{cls.negative_metric_name}_micro"] = [(-micro_num).item(), micro_den.item()]

        if denom.numel() > 0:
            active = denom > 0.0
            metric = torch.where(
                active,
                numerator / denom.clamp(min=eps),
                torch.zeros_like(denom),
            )
            class_count = tp.new_tensor(float(denom.numel()))
            metrics[f"{mode}_{scope}{cls.metric_name}_macro"] = [metric.sum().item(), class_count.item()]
            metrics[f"{mode}_{scope}{cls.negative_metric_name}_macro"] = [(-metric.sum()).item(), class_count.item()]
            metrics["active_classes"] = active.to(dtype=tp.dtype).sum().item()
            metrics["class_count"] = class_count.item()

        return metrics

    @classmethod
    def calculate_metrics(
        cls,
        events_hat: torch.Tensor | None,
        events: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        multilabel: bool = False,
        hard: bool = False,
        global_metric: bool = False,
        eps: float = 1e-8,
    ) -> dict[str, Any]:
        if events_hat is None or events is None:
            return {}

        with torch.no_grad():
            counter = cls._counts_global if global_metric else cls._counts
            tp, fp, fn, tn = counter(
                events_hat.detach(),
                events.detach(),
                mask.detach() if torch.is_tensor(mask) else mask,
                multilabel=multilabel,
                hard=hard,
            )
            metrics = cls._metrics_from_counts(
                tp,
                fp,
                fn,
                tn,
                hard=hard,
                global_metric=global_metric,
                eps=eps,
            )
            metrics["events"] = EventsCostFunction.calculate_event_metrics(
                events_hat,
                events,
                mask,
                multilabel=multilabel,
            )
            return metrics

    def __call__(
        self,
        batch: tensors_dict,
        training: bool,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if not self._check_zero_return(training):
            return 0.0

        events_hat = batch.get("events_hat")
        events = batch.get("events")
        events_mask = batch.get("events_mask")
        events_multilabel = bool(batch.get("events_multilabel", False))
        hard = self.hard_eval and not training

        loss_num, loss_div = self.calculate_loss(
            events_hat,
            events,
            events_mask,
            multilabel=events_multilabel,
            hard=hard,
            global_metric=self.global_metric,
            eps=self.eps,
        )
        self.metrics.update(
            self.calculate_metrics(
                events_hat,
                events,
                events_mask,
                multilabel=events_multilabel,
                hard=hard,
                global_metric=self.global_metric,
                eps=self.eps,
            )
        )

        loss = loss_num / loss_div.clamp(min=1.0)
        self.loss = loss_num.item()
        self.loss_div = loss_div.item()
        return loss


class _NRecallCostFunction(_NEventMetricCostFunction):
    metric_name = "recall"
    negative_metric_name = "nrecall"
    denominator = "actual"


class _NSpecificityCostFunction(_NEventMetricCostFunction):
    metric_name = "specificity"
    negative_metric_name = "nspecificity"

    @classmethod
    def _metric_numerator_denominator_from_counts(
        cls,
        tp: torch.Tensor,
        fp: torch.Tensor,
        fn: torch.Tensor,
        tn: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return tn, tn + fp


class _NPrecisionCostFunction(_NEventMetricCostFunction):
    metric_name = "precision"
    negative_metric_name = "nprecision"
    denominator = "predicted"


class _NF1CostFunction(_NEventMetricCostFunction):
    metric_name = "f1"
    negative_metric_name = "nf1"

    @classmethod
    def _metric_numerator_denominator_from_counts(
        cls,
        tp: torch.Tensor,
        fp: torch.Tensor,
        fn: torch.Tensor,
        tn: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return 2.0 * tp, 2.0 * tp + fp + fn

    @classmethod
    def _counts(
        cls,
        events_hat: torch.Tensor,
        events: torch.Tensor,
        mask: torch.Tensor | None,
        multilabel: bool = False,
        hard: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if events_hat.shape != events.shape:
            raise ValueError(
                "events_hat and events must have the same shape. "
                f"Got events_hat={tuple(events_hat.shape)}, events={tuple(events.shape)}."
            )

        valid = EventsCostFunction._normalize_events_mask(events, mask, multilabel=multilabel)
        if events.size(-1) == 1 or multilabel:
            if hard:
                valid_bool = (valid > 0.5).unsqueeze(-1)
                pred_pos = events_hat >= 0.5
                target_pos = events > 0.5
                pred = torch.stack((~pred_pos, pred_pos), dim=-1)
                target = torch.stack((~target_pos, target_pos), dim=-1)
                valid_mask = valid_bool.expand_as(pred)

                tp = (pred & target & valid_mask).to(dtype=events_hat.dtype)
                fp = (pred & ~target & valid_mask).to(dtype=events_hat.dtype)
                fn = (~pred & target & valid_mask).to(dtype=events_hat.dtype)
                tn = (~pred & ~target & valid_mask).to(dtype=events_hat.dtype)
            else:
                valid_float = valid.to(dtype=events_hat.dtype).unsqueeze(-1)
                target_pos = events.to(dtype=events_hat.dtype).clamp(min=0.0, max=1.0)
                probs_pos = events_hat.clamp(min=0.0, max=1.0)
                target = torch.stack((1.0 - target_pos, target_pos), dim=-1)
                probs = torch.stack((1.0 - probs_pos, probs_pos), dim=-1)

                tp = probs * target * valid_float
                fp = probs * (1.0 - target) * valid_float
                fn = (1.0 - probs) * target * valid_float
                tn = (1.0 - probs) * (1.0 - target) * valid_float
        else:
            if hard:
                valid_bool = valid > 0.5
                target_bool = events > 0.5
                pred_idx = events_hat.argmax(dim=-1)
                pred_bool = F.one_hot(pred_idx, num_classes=events.size(-1)).to(dtype=torch.bool)

                tp = (pred_bool & target_bool & valid_bool).to(dtype=events_hat.dtype)
                fp = (pred_bool & ~target_bool & valid_bool).to(dtype=events_hat.dtype)
                fn = (~pred_bool & target_bool & valid_bool).to(dtype=events_hat.dtype)
                tn = (~pred_bool & ~target_bool & valid_bool).to(dtype=events_hat.dtype)
            else:
                valid_float = valid.to(dtype=events_hat.dtype)
                target_float = events.to(dtype=events_hat.dtype).clamp(min=0.0, max=1.0)
                probs = events_hat.clamp(min=0.0, max=1.0)

                tp = probs * target_float * valid_float
                fp = probs * (1.0 - target_float) * valid_float
                fn = (1.0 - probs) * target_float * valid_float
                tn = (1.0 - probs) * (1.0 - target_float) * valid_float

        return (
            cls._sum_observations(tp, events.dim()),
            cls._sum_observations(fp, events.dim()),
            cls._sum_observations(fn, events.dim()),
            cls._sum_observations(tn, events.dim()),
        )

    @classmethod
    def _counts_global(
        cls,
        events_hat: torch.Tensor,
        events: torch.Tensor,
        mask: torch.Tensor | None,
        multilabel: bool = False,
        hard: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if events_hat.shape != events.shape:
            raise ValueError(
                "events_hat and events must have the same shape. "
                f"Got events_hat={tuple(events_hat.shape)}, events={tuple(events.shape)}."
            )

        valid = EventsCostFunction._normalize_events_mask(events, mask, multilabel=multilabel)
        valid_bool = valid > 0.5

        if events.size(-1) == 1 or multilabel:
            dims = cls._non_batch_dims(events)
            target_pos = cls._any_observed((events > 0.5) & valid_bool, dims)
            valid_global = cls._any_observed(valid_bool, dims)

            if hard:
                pred_pos = cls._any_observed((events_hat >= 0.5) & valid_bool, dims)
                pred = torch.stack((~pred_pos, pred_pos), dim=-1)
            else:
                valid_float = valid.to(dtype=events_hat.dtype)
                probs_pos = cls._soft_any_observed(events_hat.clamp(min=0.0, max=1.0), valid_float, dims)
                pred = torch.stack((1.0 - probs_pos, probs_pos), dim=-1)
            target = torch.stack((~target_pos, target_pos), dim=-1)
            valid_mask = valid_global.unsqueeze(-1).expand_as(pred)

            pred_float = pred.to(dtype=events_hat.dtype)
            target_float = target.to(dtype=events_hat.dtype)
            valid_float = valid_mask.to(dtype=events_hat.dtype)
            tp = pred_float * target_float * valid_float
            fp = pred_float * (1.0 - target_float) * valid_float
            fn = (1.0 - pred_float) * target_float * valid_float
            tn = (1.0 - pred_float) * (1.0 - target_float) * valid_float
            return (
                cls._sum_observations(tp, target.dim()),
                cls._sum_observations(fp, target.dim()),
                cls._sum_observations(fn, target.dim()),
                cls._sum_observations(tn, target.dim()),
            )

        dims = cls._observation_dims(events)
        target_pos = cls._any_observed((events > 0.5) & valid_bool, dims)
        valid_global = cls._any_observed(valid_bool, dims)

        if hard:
            pred_idx = events_hat.argmax(dim=-1)
            pred_step = F.one_hot(pred_idx, num_classes=events.size(-1)).to(dtype=torch.bool)
            pred_global = cls._any_observed(pred_step & valid_bool, dims)
            probs = pred_global.to(dtype=events_hat.dtype)
        else:
            valid_float = valid.to(dtype=events_hat.dtype)
            probs = cls._soft_any_observed(events_hat.clamp(min=0.0, max=1.0), valid_float, dims)

        target_float = target_pos.to(dtype=events_hat.dtype)
        valid_float = valid_global.to(dtype=events_hat.dtype)
        tp = probs * target_float * valid_float
        fp = probs * (1.0 - target_float) * valid_float
        fn = (1.0 - probs) * target_float * valid_float
        tn = (1.0 - probs) * (1.0 - target_float) * valid_float
        return (
            cls._sum_observations(tp, target_float.dim()),
            cls._sum_observations(fp, target_float.dim()),
            cls._sum_observations(fn, target_float.dim()),
            cls._sum_observations(tn, target_float.dim()),
        )


class NF1MicroCostFunction(_NF1CostFunction):
    """Negative micro-F1 cost for event heads."""

    average = "micro"


class NF1MacroCostFunction(_NF1CostFunction):
    """Negative macro-F1 cost for event heads."""

    average = "macro"


class NF1CostFunction(NF1MicroCostFunction):
    """Negative F1 cost using micro averaging by default."""


class NRecallMicroCostFunction(_NRecallCostFunction):
    """Negative micro-recall cost for event heads."""

    average = "micro"


class NRecallMacroCostFunction(_NRecallCostFunction):
    """Negative macro-recall cost for event heads."""

    average = "macro"


class NSpecificityMicroCostFunction(_NSpecificityCostFunction):
    """Negative micro-specificity cost for event heads."""

    average = "micro"


class NSpecificityMacroCostFunction(_NSpecificityCostFunction):
    """Negative macro-specificity cost for event heads."""

    average = "macro"


class NPrecisionMicroCostFunction(_NPrecisionCostFunction):
    """Negative micro-precision cost for event heads."""

    average = "micro"


class NPrecisionMacroCostFunction(_NPrecisionCostFunction):
    """Negative macro-precision cost for event heads."""

    average = "macro"


class NRecallCostFunction(NRecallMicroCostFunction):
    """Negative recall cost using micro averaging by default."""


class NSpecificityCostFunction(NSpecificityMicroCostFunction):
    """Negative specificity cost using micro averaging by default."""


class NPrecisionCostFunction(NPrecisionMicroCostFunction):
    """Negative precision cost using micro averaging by default."""


__all__ = [
    "NF1CostFunction",
    "NF1MicroCostFunction",
    "NF1MacroCostFunction",
    "NRecallCostFunction",
    "NRecallMicroCostFunction",
    "NRecallMacroCostFunction",
    "NSpecificityCostFunction",
    "NSpecificityMicroCostFunction",
    "NSpecificityMacroCostFunction",
    "NPrecisionCostFunction",
    "NPrecisionMicroCostFunction",
    "NPrecisionMacroCostFunction",
]
