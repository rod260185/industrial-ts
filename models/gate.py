from __future__ import annotations

from torch import nn
import torch

from .base import BaseIndustrialTSModel
from ..confs import *


class ITS_Gate(BaseIndustrialTSModel):

    def __init__(
            self,
            *args,
            encoders_types: list[EncoderType] | None = None, 
            ts_encoders_types:list | None = None, 
            decoders_types: list[TSDecoderType] | None = None, 
            **kwargs
            ) -> None:
        raise NotImplementedError("ITS_Gate is a placeholder for a future gated architecture. It is not implemented yet.")
        super().__init__(*args, **kwargs)
        self.encoders_types = encoders_types
        self.ts_encoders_types = ts_encoders_types
        self.decoders_types = decoders_types
        self.encoder_gate = self._make_gates(encoders_types)
        self.ts_encoder_gate = self._make_gates(ts_encoders_types)
        self.decoder_gate = self._make_gates(decoders_types)

    def _make_gates(self,types: list | list[EncoderType] | list[TSDecoderType]) -> nn.Module | None:
        if types is not None:
            if len(types) >= 2:
                gate = nn.Sequential(
                    nn.Linear(self.hidden_dim * len(types), self.hidden_dim * len(types)),
                    nn.Softmax(dim=-1)
                )        
            else:
                raise ValueError("Gated architecture requires at least 2 encoders/decoders to be effective.")
            return gate

    def _forward_gate_encoder(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        ):
        h = []
        for encoder in self.encoders:
            h.append(encoder(x, timestamps))
        h = torch.cat(h, dim=-1)
        gate_weights = self.encoder_gate(h)
        h = h * gate_weights
        h = h.view(x.size(0), x.size(1), self.hidden_dim, len(self.encoders))
        h = h.sum(dim=-1)
        return h

    def _make_encoder(self) -> None:
        self.encoders = []
        for enc in self.encoders_types:
            self.encoder_type = enc
            super()._make_encoder()
            self.encoders.append(self.encoder)
        self.encoder = self._forward_gate_encoder
        


    

__all__ = ["ITS_Gate"]
