from __future__ import annotations

import torch.nn as nn
import torch
from abc import ABC, abstractmethod

class BaseDecoder(nn.Module,ABC):
    def __init__(self,hidden_dim: int) -> None:
        super().__init__()
        self.feedback_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.GELU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        head_window_size: int,
        params: dict,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor,...]]:
        pass