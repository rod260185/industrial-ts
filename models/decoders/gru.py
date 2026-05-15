from __future__ import annotations

import torch
import torch.nn as nn

from ._common import process_decoder_out
from .._common import stack_tensor_list
from .base import BaseDecoder

class GRUDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.gru = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.decoder = decoder

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        head_window_size: int,
        params: dict,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor,...]]:
        if head_window_size < 1:
            raise ValueError("head_window_size must be at least 1")
        h0 = state[:, -1, :] if initial_state is None else initial_state
        h = h0
        x_last = x[:, -1, :]
        x_hat = []
        z_hat = []
        decoder_out_list = []
        for _ in range(head_window_size):
            h = self.gru(x_last, h)
            z_hat.append(h)
            x_last = process_decoder_out(h,self.decoder,decoder_out_list)
            x_hat.append(x_last)
        x_hat = torch.stack(x_hat, dim=1)
        z_hat = torch.stack(z_hat, dim=1)
        return x_hat, z_hat, stack_tensor_list(decoder_out_list)

class FutureGRUDecoder(BaseDecoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        decoder: nn.Module,
    ) -> None:
        super().__init__(hidden_size)
        self.gru = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)
        self.decoder = decoder

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        head_window_size: int,
        params: dict,
        initial_state: torch.Tensor | None = None,
        initial_z: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor,...]]:
        if head_window_size < 1:
            raise ValueError("head_window_size must be at least 1")
        h = state[:, -1, :] if initial_state is None else initial_state
        z = initial_z if initial_z is not None else self.feedback_proj(h)
        H = []
        Z = []
        decoder_out_list = []
        for _ in range(head_window_size):
            h = self.gru(z, h)
            H.append(h)
            z = self.feedback_proj(h)
            Z.append(z)
        H = torch.stack(H, dim=1)
        Z = torch.stack(Z, dim=1)
        x_hat = process_decoder_out(Z,self.decoder,decoder_out_list)
        return x_hat, H, stack_tensor_list(decoder_out_list)
