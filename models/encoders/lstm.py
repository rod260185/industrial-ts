from __future__ import annotations

import torch
import torch.nn as nn

from ...confs import MergeMethod
from ._common import build_merge_module, merge_bidirectional


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bidirectional: bool,
        num_layers: int,
        dropout: float,
        merge_method: MergeMethod,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = max(1, int(num_layers))
        self.merge_method = merge_method
        self.last_decoder_state: torch.Tensor | None = None
        self.last_cell_state: torch.Tensor | None = None
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        if bidirectional:
            self.merger = build_merge_module(hidden_size, merge_method)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        states, (h_n, c_n) = self.encoder(h)
        if self.bidirectional:
            H = merge_bidirectional(self.merger, states, self.merge_method, is_sequence=True)
            h_last = h_n.view(self.num_layers, 2, h.size(0), self.hidden_size)[-1]
            h_last = h_last.permute(1, 0, 2).reshape(h.size(0), self.hidden_size * 2)
            c_last = c_n.view(self.num_layers, 2, h.size(0), self.hidden_size)[-1]
            c_last = c_last.permute(1, 0, 2).reshape(h.size(0), self.hidden_size * 2)
            self.last_decoder_state = merge_bidirectional(
                self.merger,
                h_last,
                self.merge_method,
                is_sequence=False,
            )
            self.last_cell_state = merge_bidirectional(
                self.merger,
                c_last,
                self.merge_method,
                is_sequence=False,
            )
        else:
            H = states
            self.last_decoder_state = h_n[-1]
            self.last_cell_state = c_n[-1]
        return H
