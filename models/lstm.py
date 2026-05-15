from __future__ import annotations

import torch

from .base import BaseIndustrialTSModel
from .encoders.lstm import LSTMEncoder as _LSTMEncoder
from ..confs import *


class ITS_LSTM(BaseIndustrialTSModel):
    """
    LSTM model implemented on top of `BaseIndustrialTSModel`.

    Design principles:
    - Fully compatible with `base_torch_model.py` training loop.
    - Uses `tensors_dict` batches.
    - Cost terms are fully pluggable through `cost_functions`.
    """

    def __init__(self, *args, ts_decoder_type: TSDecoderType = TSDecoderType.LSTM, **kwargs):
        super().__init__(*args, ts_decoder_type=ts_decoder_type, **kwargs)

    def _make_encoder(self):
        super()._make_encoder()
        self.ts_encoder = _LSTMEncoder(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            bidirectional=self.kwargs.get('bidirectional', False),
            num_layers=self.kwargs.get('ts_encoder_layers', 1),
            dropout=self.kwargs.get('dropout', 0.0) if self.kwargs.get('ts_encoder_layers', 1) > 1 else 0.0,
            merge_method=self.kwargs.get('merge_method', MergeMethod.LINEAR),
        )

    def _forward_ts_encoder(self, h: torch.Tensor, batch, res) -> torch.Tensor:
        state = self.ts_encoder(h)
        if self.ts_encoder.last_decoder_state is not None:
            res["decoder_state"] = self.ts_encoder.last_decoder_state
        if self.ts_encoder.last_cell_state is not None:
            res["cell_state"] = self.ts_encoder.last_cell_state
        return state


__all__ = ["ITS_LSTM"]
