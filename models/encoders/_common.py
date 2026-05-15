from __future__ import annotations

import torch
import torch.nn as nn

from ...confs import MergeMethod
from ..fusers import Fuser, GRUFuser, LSTMFuser


def resolve_merge_method(merge_method: MergeMethod | str) -> MergeMethod:
    if isinstance(merge_method, MergeMethod):
        return merge_method

    if isinstance(merge_method, str):
        value = merge_method.strip()
        try:
            return MergeMethod(value.lower())
        except ValueError:
            try:
                return MergeMethod[value.upper()]
            except KeyError as exc:
                valid = ", ".join(method.value for method in MergeMethod)
                raise ValueError(f"Unsupported merge_method={merge_method!r}. Valid values: {valid}.") from exc

    # Handles enum instances kept alive across notebook reloads.
    value = getattr(merge_method, "value", None)
    if isinstance(value, str):
        return resolve_merge_method(value)
    name = getattr(merge_method, "name", None)
    if isinstance(name, str):
        return resolve_merge_method(name)

    raise TypeError(
        "merge_method must be a MergeMethod, string value, or enum-like object; "
        f"got {type(merge_method).__name__}."
    )


def build_merge_module(hidden_size: int, merge_method: MergeMethod | str) -> nn.Module:
    merge_method = resolve_merge_method(merge_method)
    if merge_method == MergeMethod.LINEAR:
        return nn.Linear(hidden_size * 2, hidden_size)
    if merge_method == MergeMethod.FUSER:
        return Fuser(hidden_size)
    if merge_method == MergeMethod.GRU_FUSER:
        return GRUFuser(hidden_size)
    if merge_method == MergeMethod.LSTM_FUSER:
        return LSTMFuser(hidden_size)
    raise ValueError(f"Unsupported merge_method: {merge_method}")


def merge_bidirectional(
    merger: nn.Module,
    states: torch.Tensor,
    merge_method: MergeMethod | str,
    is_sequence: bool,
) -> torch.Tensor:
    merge_method = resolve_merge_method(merge_method)
    if merge_method in [MergeMethod.GRU_FUSER, MergeMethod.LSTM_FUSER] and not is_sequence:
        return merger(states.unsqueeze(1)).squeeze(1)
    return merger(states)
