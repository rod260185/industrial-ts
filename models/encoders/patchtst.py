from __future__ import annotations

import math

import torch
import torch.nn as nn


def resolve_patchtst_nhead(hidden_size: int, requested_nhead: int) -> int:
    """Return the largest valid attention head count not exceeding requested_nhead."""
    requested_nhead = max(1, int(requested_nhead))
    if hidden_size % requested_nhead == 0:
        return requested_nhead
    for nhead in range(requested_nhead - 1, 0, -1):
        if hidden_size % nhead == 0:
            return nhead
    return 1


class PatchTSTEncoder(nn.Module):
    """
    Patch-based Transformer encoder for the IndustrialTS encoder contract.

    This module is intentionally shaped as an encoder, not as a full PatchTST
    forecasting model. It receives a dense temporal representation with shape
    [B, T, input_size], treats each input channel as an independent univariate
    series, encodes overlapping time patches with a shared Transformer encoder,
    then folds decoded patch representations back into a sequence with shape
    [B, T, hidden_size].

    Design goals:
    - compatible with BaseIndustrialTSModel without changing base.py;
    - returns per-timestep states expected by the existing heads;
    - exposes last_decoder_state expected by decoder heads;
    - keeps PatchTST's main inductive biases: channel independence and
      patch-level temporal attention.
    - uses patch-level time embeddings when timestamps are provided, with
      positional embeddings kept as a fallback for regular/legacy calls.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        patch_len: int = 16,
        stride: int = 8,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.0,
        max_patches: int = 512,
        dim_feedforward: int | None = None,
        activation: str = "gelu",
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.patch_len = max(1, int(patch_len))
        self.stride = max(1, int(stride))
        self.max_patches = max(1, int(max_patches))
        self.last_decoder_state: torch.Tensor | None = None

        # PatchTST embeds each univariate channel with shared weights; there is
        # no C -> hidden projection before patching.
        self.patch_proj = nn.Linear(self.patch_len, self.hidden_size)
        self.pos_embedding = nn.Embedding(self.max_patches, self.hidden_size)
        self.time_embedding = nn.Sequential(
            nn.Linear(3, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        nhead = resolve_patchtst_nhead(self.hidden_size, nhead)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward or self.hidden_size * 4,
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(1, int(num_layers)),
        )
        self.norm = nn.LayerNorm(self.hidden_size)
        self.patch_decoder = nn.Linear(self.hidden_size, self.patch_len * self.hidden_size)

    def _right_pad_to_cover_last_timestep(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Pad the sequence so patch extraction covers the final original step."""
        T = x.size(1)
        if T <= self.patch_len:
            num_patches = 1
            padded_len = self.patch_len
        else:
            num_patches = math.ceil((T - self.patch_len) / self.stride) + 1
            padded_len = (num_patches - 1) * self.stride + self.patch_len

        pad_len = padded_len - T
        if pad_len <= 0:
            return x, T, num_patches

        # Replicate the final representation instead of padding with zeros to
        # avoid creating an artificial drop at the end of the sequence.
        pad = x[:, -1:, :].expand(-1, pad_len, -1)
        return torch.cat([x, pad], dim=1), T, num_patches

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        # torch.unfold over dimension 1 returns [B, N, C, patch_len].
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 2, 1, 3).contiguous()
        return patches.view(x.size(0) * x.size(2), patches.size(2), self.patch_len)

    def _prepare_timestamps(self, timestamps: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor | None:
        if timestamps is None:
            return None
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(0).expand(x.size(0), -1)
        if timestamps.dim() != 2 or timestamps.size(0) != x.size(0) or timestamps.size(1) != x.size(1):
            raise ValueError(
                "PatchTSTEncoder timestamps must have shape [B, T] matching x. "
                f"Got timestamps={tuple(timestamps.shape)}, x={tuple(x.shape)}."
            )
        return timestamps.to(device=x.device, dtype=x.dtype)

    @staticmethod
    def _right_pad_timestamps(timestamps: torch.Tensor, padded_len: int) -> torch.Tensor:
        pad_len = padded_len - timestamps.size(1)
        if pad_len <= 0:
            return timestamps
        pad = timestamps[:, -1:].expand(-1, pad_len)
        return torch.cat([timestamps, pad], dim=1)

    def _make_patch_position_embedding(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor | None,
        padded_len: int,
        num_patches: int,
    ) -> torch.Tensor:
        if timestamps is None:
            pos_ids = torch.arange(num_patches, device=x.device)
            return self.pos_embedding(pos_ids).unsqueeze(0).expand(x.size(0) * x.size(2), -1, -1)

        timestamps = self._right_pad_timestamps(timestamps, padded_len)
        patch_times = timestamps.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patch_start = patch_times[..., 0]
        patch_end = patch_times[..., -1]
        patch_center = patch_times.mean(dim=-1)
        origin = timestamps[:, x.size(1) - 1 : x.size(1)].expand_as(patch_center)
        previous_center = torch.cat([patch_center[:, :1], patch_center[:, :-1]], dim=1)

        time_features = torch.stack(
            [
                patch_center - origin,
                patch_center - previous_center,
                (patch_end - patch_start).clamp_min(0.0),
            ],
            dim=-1,
        )
        time_embedding = self.time_embedding(time_features.to(dtype=x.dtype))
        return time_embedding.repeat_interleave(x.size(2), dim=0)

    def _fold_patches(
        self,
        patch_states: torch.Tensor,
        original_len: int,
        padded_len: int,
    ) -> torch.Tensor:
        batch_channels = patch_states.size(0)
        patch_values = self.patch_decoder(patch_states)
        patch_values = patch_values.view(batch_channels, -1, self.patch_len, self.hidden_size)

        states = patch_values.new_zeros(batch_channels, padded_len, self.hidden_size)
        counts = patch_values.new_zeros(batch_channels, padded_len, 1)
        for patch_idx in range(patch_values.size(1)):
            start = patch_idx * self.stride
            end = start + self.patch_len
            states[:, start:end, :] = states[:, start:end, :] + patch_values[:, patch_idx, :, :]
            counts[:, start:end, :] = counts[:, start:end, :] + 1.0

        states = states / counts.clamp_min(1.0)
        return states[:, :original_len, :]

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"PatchTSTEncoder expects [B, T, C], got shape={tuple(x.shape)}")
        if x.size(1) < 1:
            raise ValueError("PatchTSTEncoder requires at least one timestep.")

        timestamps = self._prepare_timestamps(timestamps, x)
        x_padded, original_len, num_patches = self._right_pad_to_cover_last_timestep(x)
        if num_patches > self.max_patches:
            raise ValueError(
                f"Number of patches ({num_patches}) exceeds max_patches={self.max_patches}. "
                "Increase patchtst_max_patches or use a larger stride/patch_len."
            )

        patches = self._extract_patches(x_padded)
        patch_tokens = self.patch_proj(patches)
        position_embedding = self._make_patch_position_embedding(
            x,
            timestamps,
            padded_len=x_padded.size(1),
            num_patches=num_patches,
        )
        patch_tokens = self.dropout(patch_tokens + position_embedding)
        patch_states = self.norm(self.transformer(patch_tokens))

        channel_states = self._fold_patches(
            patch_states,
            original_len=original_len,
            padded_len=x_padded.size(1),
        )
        states = channel_states.reshape(x.size(0), x.size(2), original_len, self.hidden_size).mean(dim=1)
        self.last_decoder_state = states[:, -1, :]
        return states


__all__ = ["PatchTSTEncoder", "resolve_patchtst_nhead"]
