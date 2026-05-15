from __future__ import annotations

import torch
import torch.nn as nn

from ._common import process_decoder_out
from .._common import stack_tensor_list


def resolve_transformer_nhead(hidden_size: int, requested_nhead: int) -> int:
    requested_nhead = max(1, int(requested_nhead))
    if hidden_size % requested_nhead == 0:
        return requested_nhead
    for nhead in range(requested_nhead - 1, 0, -1):
        if hidden_size % nhead == 0:
            return nhead
    return 1


class TransformerTimeDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        input_size: int,
        decoder: nn.Module,
        nhead: int,
        num_layers: int,
        dropout: float,
        max_length: int,
        autoregressive: bool = True,
        causal: bool = False
    ) -> None:
        super().__init__()
        self.autoregressive = autoregressive
        self.causal = autoregressive or causal
        self.max_length = max(1, int(max_length))
        self.current_epoch: int | None = None

        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.target_proj = nn.Linear(input_size, hidden_size)
        self.pos_embedding = nn.Embedding(self.max_length, hidden_size)
        self.memory_time_embedding = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.target_time_embedding = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.memory_input_norm = nn.LayerNorm(hidden_size)
        self.target_input_norm = nn.LayerNorm(hidden_size)
        self.register_buffer(
            "_tgt_causal_mask",
            torch.triu(torch.ones(self.max_length, self.max_length, dtype=torch.bool), diagonal=1),
            persistent=False,
        )
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=max(1, int(num_layers)))
        self.norm = nn.LayerNorm(hidden_size)
        self.decoder = decoder

    def _teacher_forcing_ratio(self) -> float:
        if not self.training or not self.teacher_forcing:
            return 0.0
        if self.current_epoch is None or self.teacher_forcing_decay_epochs == 0:
            return self.teacher_forcing_start
        progress = min(max((int(self.current_epoch) - 1) / self.teacher_forcing_decay_epochs, 0.0), 1.0)
        return self.teacher_forcing_start + progress * (self.teacher_forcing_end - self.teacher_forcing_start)

    @staticmethod
    def _ensure_batched_timestamps(
        timestamps: torch.Tensor | None,
        batch_size: int,
        expected_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        name: str,
    ) -> torch.Tensor | None:
        if timestamps is None:
            return None
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(0).expand(batch_size, -1)
        if timestamps.dim() != 2 or timestamps.size(0) != batch_size or timestamps.size(1) != expected_size:
            raise ValueError(
                f"{name} must have shape [B, {expected_size}]. "
                f"Got shape={tuple(timestamps.shape)} for batch_size={batch_size}."
            )
        return timestamps.to(device=device, dtype=dtype)

    @staticmethod
    def _time_features(
        timestamps: torch.Tensor,
        origin_timestamps: torch.Tensor,
        previous_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        delta_origin = timestamps - origin_timestamps
        delta_step = timestamps - previous_timestamps
        return torch.stack([delta_origin, delta_step], dim=-1)

    def _make_memory(self, state: torch.Tensor, input_timestamps: torch.Tensor | None) -> torch.Tensor:
        if input_timestamps is None:
            return state

        origin_timestamps = input_timestamps[:, -1:].expand_as(input_timestamps)
        previous_timestamps = torch.cat([input_timestamps[:, :1], input_timestamps[:, :-1]], dim=1)
        time_features = self._time_features(
            input_timestamps,
            origin_timestamps,
            previous_timestamps,
        ).to(dtype=state.dtype)
        return self.memory_input_norm(state + self.memory_time_embedding(time_features))

    def _make_target_time_embedding(
        self,
        head_timestamps: torch.Tensor | None,
        input_timestamps: torch.Tensor | None,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if head_timestamps is None:
            return None

        if input_timestamps is None:
            origin_timestamps = head_timestamps[:, :1] - 1.0
        else:
            origin_timestamps = input_timestamps[:, -1:]
        origin_timestamps = origin_timestamps.expand_as(head_timestamps)
        previous_timestamps = torch.cat([origin_timestamps[:, :1], head_timestamps[:, :-1]], dim=1)
        time_features = self._time_features(
            head_timestamps,
            origin_timestamps,
            previous_timestamps,
        ).to(dtype=dtype)
        return self.target_time_embedding(time_features)

    def _decode_tokens(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        target_time_embedding: torch.Tensor | None,
        *,
        causal: bool,
        add_time_embedding: bool = True,
    ) -> torch.Tensor:
        if add_time_embedding and target_time_embedding is not None:
            tgt = tgt_tokens + target_time_embedding[:, : tgt_tokens.size(1), :]
        elif add_time_embedding:
            pos_ids = torch.arange(tgt_tokens.size(1), device=tgt_tokens.device)
            tgt = tgt_tokens + self.pos_embedding(pos_ids).unsqueeze(0)
        else:
            tgt = tgt_tokens
        tgt = self.target_input_norm(tgt)
        tgt_mask = self._tgt_causal_mask[:tgt.size(1), :tgt.size(1)] if causal else None
        decoded = self.transformer(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        return self.norm(decoded)

    def _non_autoregressive_tokens(
        self,
        start_token: torch.Tensor,
        target_time_embedding: torch.Tensor | None,
        head_window_size: int,
    ) -> torch.Tensor:
        start_tokens = start_token.expand(-1, head_window_size, -1)
        if target_time_embedding is not None:
            return start_tokens + target_time_embedding[:, :head_window_size, :]

        pos_ids = torch.arange(head_window_size, device=start_token.device)
        return start_tokens + self.pos_embedding(pos_ids).unsqueeze(0)

    def _forward_non_autoregressive(
        self,
        start_token: torch.Tensor,
        memory: torch.Tensor,
        target_time_embedding: torch.Tensor | None,
        head_targets: torch.Tensor | None,
        head_window_size: int,
        teacher_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        decoder_out_list: list[list[torch.Tensor]] = []
        tgt_tokens = self._non_autoregressive_tokens(
            start_token,
            target_time_embedding,
            head_window_size,
        )
        causal = False

        if self.training and teacher_ratio > 0.0 and head_targets is not None:
            teacher_tokens = tgt_tokens
            if head_window_size > 1:
                teacher_tokens = tgt_tokens.clone()
                teacher_tokens[:, 1:, :] = (
                    teacher_tokens[:, 1:, :] + self.target_proj(head_targets[:, :-1, :])
                )
            if teacher_ratio >= 1.0:
                tgt_tokens = teacher_tokens
            else:
                teacher_mask = torch.rand(
                    tgt_tokens.size(0),
                    tgt_tokens.size(1),
                    1,
                    device=tgt_tokens.device,
                ) < teacher_ratio
                teacher_mask[:, :1, :] = False
                tgt_tokens = torch.where(teacher_mask, teacher_tokens, tgt_tokens)
            causal = True
        elif self.causal or not self.training and (
            (self.current_epoch is not None and
            self.teacher_forcing_decay_epochs >= self.current_epoch)
            ):
            causal = True

        decoded = self._decode_tokens(
            tgt_tokens,
            memory,
            None,
            causal=causal,
            add_time_embedding=False,
        )
        x_hat = process_decoder_out(decoded, self.decoder, decoder_out_list)
        return x_hat, decoded, stack_tensor_list(decoder_out_list)

    def _forward_teacher_forced_autoregressive(
        self,
        start_token: torch.Tensor,
        memory: torch.Tensor,
        target_time_embedding: torch.Tensor | None,
        head_targets: torch.Tensor,
        head_window_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        decoder_out_list: list[list[torch.Tensor]] = []
        if head_window_size == 1:
            tgt_tokens = start_token
        else:
            teacher_tokens = self.target_proj(head_targets[:, :-1, :])
            tgt_tokens = torch.cat([start_token, teacher_tokens], dim=1)
        decoded = self._decode_tokens(
            tgt_tokens,
            memory,
            target_time_embedding,
            causal=True,
        )
        x_hat = process_decoder_out(decoded, self.decoder, decoder_out_list)
        return x_hat, decoded, stack_tensor_list(decoder_out_list)

    def _forward_autoregressive(
        self,
        start_token: torch.Tensor,
        memory: torch.Tensor,
        target_time_embedding: torch.Tensor | None,
        head_targets: torch.Tensor | None,
        head_window_size: int,
        teacher_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        if self.training and teacher_ratio >= 1.0 and head_targets is not None:
            return self._forward_teacher_forced_autoregressive(
                start_token,
                memory,
                target_time_embedding,
                head_targets,
                head_window_size,
            )

        decoder_out_list: list[list[torch.Tensor]] = []
        outputs: list[torch.Tensor] = []
        z_hats: list[torch.Tensor] = []

        tgt_tokens = start_token
        for step in range(head_window_size):
            decoded = self._decode_tokens(
                tgt_tokens,
                memory,
                target_time_embedding,
                causal=True,
            )
            next_hidden = decoded[:, -1:, :]
            next_output = process_decoder_out(next_hidden, self.decoder, decoder_out_list)
            outputs.append(next_output)
            z_hats.append(next_hidden)

            if step + 1 >= head_window_size:
                continue

            next_token = self.target_proj(next_output)
            if self.training and head_targets is not None and teacher_ratio > 0.0:
                teacher_token = self.target_proj(head_targets[:, step : step + 1, :])
                teacher_mask = torch.rand(next_output.size(0), 1, 1, device=next_output.device) < teacher_ratio
                next_token = torch.where(teacher_mask, teacher_token, next_token)
            tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)

        return torch.cat(outputs, dim=1), torch.cat(z_hats, dim=1), stack_tensor_list(decoder_out_list)

    def forward(
        self,
        state: torch.Tensor,
        head_window_size: int,
        params: dict,
        initial_state: torch.Tensor | None = None,
        head_targets: torch.Tensor | None = None,
        input_timestamps: torch.Tensor | None = None,
        head_timestamps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor,...]]:
        
        if not hasattr(self, "teacher_forcing") or (self.training and self.current_epoch == 1):
            self.teacher_forcing=bool(params.get("decoder_teacher_forcing", self.autoregressive))
            self.teacher_forcing_start=float(params.get("decoder_teacher_forcing_start", 1.0))
            self.teacher_forcing_end=float(params.get("decoder_teacher_forcing_end", 0.25))
            self.teacher_forcing_decay_epochs=max(int(params.get("decoder_teacher_forcing_decay_epochs", 20)),0)
            if not 0.0 <= self.teacher_forcing_end <= self.teacher_forcing_start <= 1.0:
                raise ValueError(
                    "Teacher forcing schedule must satisfy "
                    "0 <= teacher_forcing_end <= teacher_forcing_start <= 1."
                )
            if not self.causal and self.teacher_forcing and self.teacher_forcing_end>0.:
                raise ValueError(
                    "Non-causal decoder cannot use teacher forcing with end value > 0. "
                )
        if head_window_size < 1:
            raise ValueError("head_window_size must be at least 1")
        if head_window_size > self.max_length:
            raise ValueError(
                f"head_window_size={head_window_size} exceeds decoder_max_length={self.max_length}"
            )
        if head_targets is not None and head_targets.shape[1] != head_window_size:
            raise ValueError(
                "head_targets must match head_window_size. "
                f"Got head_targets.shape={tuple(head_targets.shape)}, "
                f"head_window_size={head_window_size}."
            )

        device = state.device
        input_timestamps = self._ensure_batched_timestamps(
            input_timestamps,
            state.size(0),
            state.size(1),
            device=device,
            dtype=state.dtype,
            name="input_timestamps",
        )
        head_timestamps = self._ensure_batched_timestamps(
            head_timestamps,
            state.size(0),
            head_window_size,
            device=device,
            dtype=state.dtype,
            name="head_timestamps",
        )
        memory = self._make_memory(state, input_timestamps)
        target_time_embedding = self._make_target_time_embedding(
            head_timestamps,
            input_timestamps,
            dtype=state.dtype,
        )
        seed_state = state[:, -1, :] if initial_state is None else initial_state
        start_token = self.query_proj(seed_state).unsqueeze(1)
        teacher_ratio = self._teacher_forcing_ratio()

        if not self.autoregressive:
            return self._forward_non_autoregressive(
                start_token,
                memory,
                target_time_embedding,
                head_targets,
                head_window_size,
                teacher_ratio,
            )

        return self._forward_autoregressive(
            start_token,
            memory,
            target_time_embedding,
            head_targets,
            head_window_size,
            teacher_ratio,
        )
