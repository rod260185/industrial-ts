from __future__ import annotations

from .base import BaseIndustrialTSModel
from .encoders.gru import GRUEncoder as _GRUEncoder
from ..confs import *


class ITS_GRU(BaseIndustrialTSModel):
    """
    GRU model implemented on top of `BaseIndustrialTSModel`.

    Design principles:
    - Fully compatible with `base_torch_model.py` training loop.
    - Uses `tensors_dict` batches.
    - Cost terms are fully pluggable through `cost_functions`.
    """

    def _make_encoder(self):
        super()._make_encoder()
        self.ts_encoder = _GRUEncoder(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            bidirectional=self.kwargs.get('bidirectional', False),
            num_layers=self.kwargs.get('ts_encoder_layers', 1),
            dropout=self.kwargs.get('dropout', 0.) if self.kwargs.get('ts_encoder_layers', 1) > 1 else 0.,
            merge_method=self.kwargs.get('merge_method', MergeMethod.LINEAR)
        )



__all__ = ["ITS_GRU"]
