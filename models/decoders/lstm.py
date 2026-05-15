from __future__ import annotations

import torch
import torch.nn as nn

from ._common import process_decoder_out
from .._common import stack_tensor_list

class LSTMDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.decoder = decoder

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        head_window_size: int,
        params: dict,
        initial_state: torch.Tensor | None = None,
        cell_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor,...]]:
        if head_window_size < 1:
            raise ValueError("head_window_size must be at least 1")
        h = state[:, -1, :] if initial_state is None else initial_state
        c = torch.zeros_like(h) if cell_state is None else cell_state
        x_last = x[:, -1, :]
        x_hat = []
        z_hat = []
        decoder_out_list = []
        for _ in range(head_window_size):
            h, c = self.lstm(x_last, (h, c))
            z_hat.append(h)
            x_last = process_decoder_out(h,self.decoder,decoder_out_list)
            x_hat.append(x_last)
        x_hat = torch.stack(x_hat, dim=1)
        z_hat = torch.stack(z_hat, dim=1)
        return x_hat, z_hat, stack_tensor_list(decoder_out_list)
