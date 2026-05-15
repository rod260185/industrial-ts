from __future__ import annotations

from typing import Callable, Iterable, Mapping

import torch


class tensors_dict(dict):
    def __init__(
        self,
        tensors_dict: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]] | None = None,
    ) -> None:
        super().__init__({} if tensors_dict is None else tensors_dict)

    def move_to_device(self, device: torch.device) -> tensors_dict:
        return tensors_dict({k: v.to(device) for k, v in self.items()})

    def __call__(self, func: Callable) -> tensors_dict:
        return tensors_dict({k: func(v) for k, v in self.items()})

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        return super().__setitem__(key, value)


__all__ = ["tensors_dict"]
