from __future__ import annotations

import torch
import torch.nn as nn

from ...confs import MergeMethod
from ._common import build_merge_module, merge_bidirectional, resolve_merge_method


def resolve_layernorm_mode(layernorm_mode: str) -> tuple[bool, bool]:
    mode = str(layernorm_mode).lower()
    if mode == "both":
        return True, True
    if mode == "ode":
        return True, False
    if mode == "gru":
        return False, True
    if mode == "none":
        return False, False
    raise ValueError(f"Unsupported layernorm_mode: {layernorm_mode}")


class RK4ODEFunc(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(
        self,
        h: torch.Tensor,
        x_last: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([h, x_last], dim=-1))


class JumpODEBlock(nn.Module):
    def __init__(self, hidden_size: int, layernorm_mode: str = "gru") -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.odefunc = RK4ODEFunc(hidden_size)
        self.last_decoder_state: torch.Tensor | None = None
        use_norm_ode, use_norm_gru = resolve_layernorm_mode(layernorm_mode)
        self.norm_ode = nn.LayerNorm(hidden_size) if use_norm_ode else nn.Identity()
        self.norm_gru = nn.LayerNorm(hidden_size) if use_norm_gru else nn.Identity()

    def _ode_step(
        self,
        h: torch.Tensor,
        dt: torch.Tensor,
        x_ref: torch.Tensor,
    ) -> torch.Tensor:
        k1 = self.odefunc(h, x_ref)
        k2 = self.odefunc(h + 0.5 * dt * k1, x_ref)
        k3 = self.odefunc(h + 0.5 * dt * k2, x_ref)
        k4 = self.odefunc(h + dt * k3, x_ref)
        h = h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.norm_ode(h)

    @staticmethod
    def _prepare_final_timestamps(
        final_timestamps: torch.Tensor | None,
        timestamps: torch.Tensor,
    ) -> torch.Tensor | None:
        if final_timestamps is None:
            return None
        if final_timestamps.dim() == 0:
            final_timestamps = final_timestamps.expand(timestamps.size(0))
        elif final_timestamps.dim() == 2 and final_timestamps.size(1) == 1:
            final_timestamps = final_timestamps.squeeze(1)
        if final_timestamps.dim() != 1 or final_timestamps.size(0) != timestamps.size(0):
            raise ValueError(
                "final_timestamps must have shape [batch] or [batch, 1]. "
                f"Got final_timestamps={tuple(final_timestamps.shape)}, batch={timestamps.size(0)}."
            )
        return final_timestamps.to(device=timestamps.device, dtype=timestamps.dtype)

    def _project_decoder_state_to_final_timestamp(
        self,
        h: torch.Tensor,
        x_ref: torch.Tensor,
        timestamps: torch.Tensor,
        final_timestamps: torch.Tensor | None,
    ) -> torch.Tensor:
        final_timestamps = self._prepare_final_timestamps(final_timestamps, timestamps)
        if final_timestamps is None:
            return h

        last_timestamps = timestamps[:, -1].to(dtype=h.dtype)
        final_timestamps = final_timestamps.to(dtype=h.dtype)
        needs_projection = final_timestamps > last_timestamps
        if not needs_projection.any():
            return h

        projected = h.clone()
        project_idx = needs_projection.nonzero(as_tuple=False).squeeze(-1)
        eps = torch.finfo(h.dtype).eps
        dt = (final_timestamps[project_idx] - last_timestamps[project_idx]).unsqueeze(-1).clamp(min=eps)
        h_projected = self._ode_step(h[project_idx], dt, x_ref[project_idx])
        return projected.index_copy(0, project_idx, h_projected)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor,
        final_timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        states: list[torch.Tensor] = []
        eps = torch.finfo(x.dtype).eps
        x_ref = torch.zeros_like(x[:, 0])
        self.last_decoder_state = None

        for i in range(T):
            if i > 0:
                dt = (timestamps[:, i] - timestamps[:, i - 1]).abs().to(dtype=x.dtype).unsqueeze(-1)
                dt = dt.clamp(min=eps)
                h = self._ode_step(h, dt, x_ref)
            obs_t = mask[:, i].any(dim=-1, keepdim=True)
            obs_idx = obs_t.squeeze(-1).nonzero(as_tuple=False).squeeze(-1)
            if obs_idx.numel() > 0:
                h_jump = self.norm_gru(self.gru(x[obs_idx, i], h[obs_idx]))
                h = h.index_copy(0, obs_idx, h_jump)
                x_ref = x_ref.index_copy(0, obs_idx, x[obs_idx, i])
            states.append(h)

        self.last_decoder_state = self._project_decoder_state_to_final_timestamp(
            h,
            x_ref,
            timestamps,
            final_timestamps,
        )
        return torch.stack(states, dim=1)


class ODEJumpEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bidirectional: bool,
        num_layers: int,
        dropout: float,
        merge_method: MergeMethod,
        layernorm_mode: str = "gru",
        decoder_init_method: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = max(1, int(num_layers))
        self.merge_method = resolve_merge_method(merge_method)
        self.decoder_init_method = self._resolve_decoder_init_method(decoder_init_method)
        self.last_decoder_state: torch.Tensor | None = None
        self.input_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
        self.fw_layers = nn.ModuleList(
            JumpODEBlock(hidden_size, layernorm_mode=layernorm_mode) for _ in range(self.num_layers)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        if bidirectional:
            self.bw_layers = nn.ModuleList(
                JumpODEBlock(hidden_size, layernorm_mode=layernorm_mode) for _ in range(self.num_layers)
            )
            self.mergers = nn.ModuleList(
                build_merge_module(hidden_size, self.merge_method) for _ in range(self.num_layers)
            )
            if self.decoder_init_method == "linear":
                self.decoder_init_linear = nn.Linear(hidden_size * 2, hidden_size)
                self.decoder_init_gate = None
                self.decoder_init_delta = None
            else:
                self.decoder_init_linear = None
                self.decoder_init_gate = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.Sigmoid(),
                )
                self.decoder_init_delta = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.Tanh(),
                )
        else:
            self.bw_layers = None
            self.mergers = None
            self.decoder_init_linear = None
            self.decoder_init_gate = None
            self.decoder_init_delta = None

    def _resolve_decoder_init_method(self, decoder_init_method: str | None) -> str:
        if decoder_init_method is None:
            return "linear" if self.merge_method == MergeMethod.LINEAR else "gate"
        method = str(decoder_init_method).strip().lower()
        if method in {"gate", "gated"}:
            return "gate"
        if method == "linear":
            return "linear"
        raise ValueError(
            "Unsupported decoder_init_method for ODEJumpEncoder. "
            "Use 'gate' or 'linear'."
        )

    def _make_decoder_initial_state(
        self,
        fw_decoder_state: torch.Tensor,
        bw_summary: torch.Tensor,
    ) -> torch.Tensor:
        decoder_input = torch.cat([fw_decoder_state, bw_summary], dim=-1)
        if self.decoder_init_method == "linear":
            return self.decoder_init_linear(decoder_input)

        gate = self.decoder_init_gate(decoder_input)
        delta = self.decoder_init_delta(decoder_input)
        return fw_decoder_state + gate * delta

    def _run_layer(
        self,
        layer: JumpODEBlock,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor,
        final_timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return layer(x, timestamps, mask, final_timestamps=final_timestamps)

    def _run_backward_layer(
        self,
        layer: JumpODEBlock,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        states = layer(x.flip(1), timestamps.flip(1), mask.flip(1))
        return states.flip(1)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor,
        final_timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if timestamps is None:
            raise ValueError("timestamps must be provided to ODEJumpEncoder")

        h = self.input_proj(x) if self.input_proj is not None else x
        self.last_decoder_state = None
        for layer_idx, fw_layer in enumerate(self.fw_layers):
            h_fw = self._run_layer(fw_layer, h, timestamps, mask, final_timestamps=final_timestamps)
            fw_decoder_state = fw_layer.last_decoder_state
            if fw_decoder_state is None:
                fw_decoder_state = h_fw[:, -1, :]
            if self.bidirectional:
                h_bw = self._run_backward_layer(self.bw_layers[layer_idx], h, timestamps, mask)
                merged_states = torch.cat([h_fw, h_bw], dim=-1)
                h = merge_bidirectional(
                    self.mergers[layer_idx],
                    merged_states,
                    self.merge_method,
                    is_sequence=True,
                )
                self.last_decoder_state = self._make_decoder_initial_state(
                    fw_decoder_state,
                    h_bw[:, 0, :],
                )
            else:
                h = h_fw
                self.last_decoder_state = fw_decoder_state
            if self.dropout is not None and layer_idx < self.num_layers - 1:
                h = self.dropout(h)
        return h
