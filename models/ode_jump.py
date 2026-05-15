from __future__ import annotations

import torch

from .base import BaseIndustrialTSModel
from .encoders.ode_jump import ODEJumpEncoder as _ODEJumpEncoder
from .encoders._common import resolve_merge_method
from ..confs import *
from ..collections import tensors_dict


class ITS_ODEJump(BaseIndustrialTSModel):
    """
    ODE-Jump model implemented on top of `BaseIndustrialTSModel`.

    Encoder:
    - RK4 integration between events
    - GRU jump at each observation

    Decoder:
    - GRU decoder by default
    - optional Transformer decoder with causal target attention
      and scheduled teacher forcing during training
    """

    def _make_encoder(self):
        super()._make_encoder()
        merge_method = resolve_merge_method(self.kwargs.get("merge_method", MergeMethod.LINEAR))
        if (
            self.kwargs.get("bidirectional", False)
            and "decoder" in self.heads
            and merge_method not in {MergeMethod.LINEAR, MergeMethod.FUSER}
        ):
            raise ValueError(
                "ITS_ODEJump with decoder heads is incompatible with "
                f"merge_method={merge_method.value!r}. The ODEJump decoder is initialized from "
                "an explicit final-time state adapter, which supports linear or gated "
                "initialization. Use MergeMethod.LINEAR or MergeMethod.FUSER for bidirectional "
                "ODEJump models with decoder heads."
            )
        self.ts_encoder = _ODEJumpEncoder(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            bidirectional=self.kwargs.get("bidirectional", False),
            num_layers=self.kwargs.get("ts_encoder_layers", 1),
            dropout=self.kwargs.get("dropout", 0.0) if self.kwargs.get("ts_encoder_layers", 1) > 1 else 0.0,
            merge_method=merge_method,
            layernorm_mode=self.kwargs.get("layernorm_mode", "gru"),
            decoder_init_method=self.kwargs.get("decoder_init_method"),
        )

    def _forward_ts_encoder(self, h: torch.Tensor, batch: tensors_dict, res: tensors_dict) -> torch.Tensor:
        timestamps = batch["timestamps"]
        mask = res["mask_train"] if "mask_train" in res else batch["mask"]
        state = self.ts_encoder(
            h,
            timestamps,
            mask,
            final_timestamps=batch.get("encoder_final_timestamp"),
        )
        if self.ts_encoder.last_decoder_state is not None:
            res["decoder_state"] = self.ts_encoder.last_decoder_state
        return state
    


__all__ = [
    "ITS_ODEJump",
]
