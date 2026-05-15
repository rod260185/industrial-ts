from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseIndustrialTSModel
from .encoders.patchtst import PatchTSTEncoder as _PatchTSTEncoder
from ..collections import tensors_dict
from ..confs import *


class ITS_PatchTST(BaseIndustrialTSModel):
    """
    PatchTST-style model implemented on top of BaseIndustrialTSModel.

    This class adds a patch-based Transformer temporal encoder while keeping
    the existing IndustrialTS heads, losses and decoder options unchanged.
    It does not require any change to base.py.

    Main kwargs:
    - patch_len: length of each temporal patch. Default: 16.
    - patch_stride: stride between consecutive patches. Default: patch_len // 2.
    - ts_encoder_layers: number of Transformer encoder layers. Default: 2.
    - encoder_nhead: requested number of attention heads. Default: 4.
    - patchtst_max_patches: maximum number of patches per window. Default: 512.
    - patchtst_dim_feedforward: Transformer feedforward width. Default: 4 * hidden_dim.
    - patchtst_activation: Transformer activation. Default: gelu.
    - patchtst_norm_first: pre-norm Transformer encoder. Default: True.
    """
    def __init__(self,*args,encoder_type: EncoderType = EncoderType.PATCHTST,**kwargs) -> None:
        if encoder_type != EncoderType.PATCHTST:
            raise ValueError('This model only supports argument encoder_type = EncoderType.PATCHTST.')
        super().__init__(*args,encoder_type=encoder_type,**kwargs)

    def _make_encoder(self) -> None:
        super()._make_encoder()  # sets self.in_channels and self.hidden_dim
        self.ts_encoder = nn.Identity()  # passthrough; no separate TS encoder layers

    def _forward_ts_encoder(
        self,
        h: torch.Tensor,
        batch: tensors_dict,
        res: tensors_dict,
    ) -> torch.Tensor:
        self.ts_encoder.last_decoder_state = self.encoder.last_decoder_state
        return super()._forward_ts_encoder(h, batch, res)

__all__ = ["ITS_PatchTST"]
