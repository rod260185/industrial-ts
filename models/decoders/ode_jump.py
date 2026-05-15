from __future__ import annotations

import torch
import torch.nn as nn

from ..encoders.ode_jump import JumpODEBlock

from ._common import process_decoder_out
from .._common import stack_tensor_list

class ODEJumpDecoder(JumpODEBlock):
    """
    Autoregressive ODE-Jump decoder.

    The decoder follows the same high-level contract as the GRU/LSTM decoders:
    it receives the observed input window, encoder states, and a forecast
    horizon, then returns predicted values and decoder hidden states.

    Timestamp arguments are optional to keep the decoder usable in the same
    pattern as the existing decoders. When provided, `head_timestamps` is used
    to project each future step with the ODE block before the GRU jump.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        decoder: nn.Module,
        layernorm_mode: str = "gru",
    ) -> None:
        self.input_size = int(input_size)
        super().__init__(hidden_size=int(hidden_size), layernorm_mode=layernorm_mode)
        self.input_proj = nn.Linear(self.input_size, self.hidden_size) if self.input_size != self.hidden_size else nn.Identity()
        self.decoder = decoder

    @staticmethod
    def _ensure_batched_timestamps(
        timestamps: torch.Tensor | None,
        batch_size: int,
        *,
        name: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if timestamps is None:
            return None
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(0).expand(batch_size, -1)
        if timestamps.dim() != 2 or timestamps.size(0) != batch_size:
            raise ValueError(
                f"{name} must have shape [batch, time] or [time]. "
                f"Got shape {tuple(timestamps.shape)} for batch_size={batch_size}."
            )
        return timestamps.to(device=device, dtype=dtype)

    def _make_step_deltas(
        self,
        head_window_size: int,
        input_timestamps: torch.Tensor | None,
        head_timestamps: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        eps = torch.finfo(dtype).eps
        if head_timestamps is None:
            return torch.ones(batch_size, head_window_size, 1, device=device, dtype=dtype)
        if head_timestamps.size(1) != head_window_size:
            raise ValueError(
                "head_timestamps must match head_window_size. "
                f"Got shape={tuple(head_timestamps.shape)}, head_window_size={head_window_size}."
            )

        first_previous = head_timestamps[:, :1] - 1.0 if input_timestamps is None else input_timestamps[:, -1:]
        previous_ts = torch.cat([first_previous, head_timestamps[:, :-1]], dim=1)
        return (head_timestamps - previous_ts).abs().unsqueeze(-1).clamp(min=eps).to(device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        head_window_size: int,
        params: dict,
        initial_state: torch.Tensor | None = None,
        input_timestamps: torch.Tensor | None = None,
        head_timestamps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor,...]]:
        if head_window_size < 1:
            raise ValueError("head_window_size must be at least 1")
        if x.dim() != 3 or x.shape[-1] != self.input_size:
            raise ValueError(
                "ODEJumpDecoder expects x with shape [batch, seq, input_size]. "
                f"Got shape {tuple(x.shape)} and input_size={self.input_size}."
            )
        if state.dim() != 3 or state.shape[-1] != self.hidden_size:
            raise ValueError(
                "ODEJumpDecoder expects state with shape [batch, seq, hidden_size]. "
                f"Got shape {tuple(state.shape)} and hidden_size={self.hidden_size}."
            )

        batch_size = x.size(0)
        input_timestamps = self._ensure_batched_timestamps(
            input_timestamps,
            batch_size,
            name="input_timestamps",
            device=x.device,
            dtype=x.dtype,
        )
        head_timestamps = self._ensure_batched_timestamps(
            head_timestamps,
            batch_size,
            name="head_timestamps",
            device=x.device,
            dtype=x.dtype,
        )

        h = state[:, -1, :] if initial_state is None else initial_state
        x_ref = self.input_proj(x[:, -1, :])
        step_deltas = self._make_step_deltas(
            head_window_size,
            input_timestamps,
            head_timestamps,
            batch_size,
            x.device,
            x.dtype,
        )
        x_hat = []
        z_hat = []
        decoder_out_list = []

        for step in range(head_window_size):
            h = self._ode_step(h, step_deltas[:, step], x_ref)
            h = self.norm_gru(self.gru(x_ref, h))
            x_next = process_decoder_out(h,self.decoder,decoder_out_list)
            x_ref = self.input_proj(x_next)
            x_hat.append(x_next)
            z_hat.append(h)

        return torch.stack(x_hat, dim=1), torch.stack(z_hat, dim=1), stack_tensor_list(decoder_out_list)


__all__ = ["ODEJumpDecoder"]
