from __future__ import annotations
from typing import Callable

import torch

def process_decoder_out(state:torch.Tensor,decoder: Callable,out: list[list[torch.Tensor]]) -> torch.Tensor:

    decoder_out = decoder(state)

    if isinstance(decoder_out,tuple):
        for i in range(1,len(decoder_out)):
            try:
                out[i-1] += [decoder_out[i]]
            except IndexError:
                out.append([decoder_out[i]])
        return decoder_out[0]
    else:
        return decoder_out
    
