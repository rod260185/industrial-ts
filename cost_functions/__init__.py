from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from typing import Any

import torch

from ..collections import tensors_dict

class CostFunction(ABC):

    def __init__(self, training_ratio: float=1.0, test_ratio: float=1.0) -> None:
        super().__init__()
        self.training_ratio = training_ratio
        self.test_ratio = test_ratio
        self.valid_cost_function: bool = True
        self.loss: float | None = None
        self.loss_div: float | None = None
        self.metrics: dict[str, Any] = {}

    def get_ratio(self, training: bool) -> float:
        return self.training_ratio if training else self.test_ratio

    def _check_zero_return(self, training: bool) -> bool:
        self.metrics = {}
        if training and self.training_ratio > 0 or not training and self.test_ratio > 0:
            self.valid_cost_function = True
            return True
        else:
            self.valid_cost_function = False
            self.loss = 0.0
            self.loss_div = 1.0
            return False
        
    
    @staticmethod
    @abstractmethod
    def calculate_loss(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the raw loss sum and divisor for a batch.

        Implementations should return a tuple of (loss_sum, loss_div) where:
        - loss_sum is the aggregated loss across the batch (e.g. total MSE)
        - loss_div is the aggregated divisor for normalization (e.g. total count of observations)
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, batch: tensors_dict, training: bool, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Compute normalized scalar cost for a specific batch.

        Implementations should store raw aggregated values in
        `self.loss` and `self.loss_div` for logging/aggregation.
        """
        raise NotImplementedError


from .log_likelihood_cost_function import LogLikelihoodCostFunction,LogLikelihoodCostFunctionDecoder
from .mask_bce_cost_function import MaskBCECostFunction
from .mse_cost_function import MSECostFunction,MSECostFunctionDecoder
from .noise_prediction_cost_function import NoisePredictionCostFunction
from .events_cost_function import EventsCostFunction
from .recall_penalty_cost_function import RecallPenaltyCostFunction
from .precision_penalty_cost_function import PrecisionPenaltyCostFunction
from .nmetric_cost_function import (
    NF1CostFunction,
    NF1MacroCostFunction,
    NF1MicroCostFunction,
    NPrecisionCostFunction,
    NPrecisionMacroCostFunction,
    NPrecisionMicroCostFunction,
    NRecallCostFunction,
    NRecallMacroCostFunction,
    NRecallMicroCostFunction,
    NSpecificityCostFunction,
    NSpecificityMacroCostFunction,
    NSpecificityMicroCostFunction,
)
from .vae_nll_cost_function import (
    NELBOCostFunction, 
    NELBOCostFunctionDecoder,
    NLLCostFunction,
    NLLCostFunctionDecoder
)

__all__ = [
    "CostFunction",
    "MSECostFunction",
    "MSECostFunctionDecoder",
    "LogLikelihoodCostFunction",
    "LogLikelihoodCostFunctionDecoder",
    "NoisePredictionCostFunction",
    "EventsCostFunction",
    "RecallPenaltyCostFunction",
    "PrecisionPenaltyCostFunction",
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
    "MaskBCECostFunction",
    "NELBOCostFunction",
    "NELBOCostFunctionDecoder",
    "NLLCostFunction",
    "NLLCostFunctionDecoder"
]
