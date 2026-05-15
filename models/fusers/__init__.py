from __future__ import annotations

import torch
from torch import nn


class Fuser(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.sigma = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma(states)
        a, b = states.chunk(2, -1)
        return a * sigma + (1 - sigma) * b

class _BaseRNNFuser(nn.Module):

    rnn_cls = None

    def __init__(self, hidden_size: int):
        super().__init__()
        self.fuser = self.rnn_cls(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        H, _ = self.fuser(states)
        return H

class GRUFuser(_BaseRNNFuser):
    
    rnn_cls = nn.GRU


    
class LSTMFuser(_BaseRNNFuser):
    
    rnn_cls = nn.LSTM


__all__ = ["Fuser", "GRUFuser", "LSTMFuser"]
