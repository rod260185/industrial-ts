import torch
import torch.nn as nn

def stack_tensor_list(tensor_list: list[list[torch.Tensor]]) -> tuple[torch.Tensor,...]:
    stacked: list[torch.Tensor] = []
    for tensors in tensor_list:
        if len(tensors) == 1 and tensors[0].dim() >= 3:
            stacked.append(tensors[0])
        elif tensors and tensors[0].dim() >= 3 and tensors[0].size(1) == 1:
            stacked.append(torch.cat(tensors, dim=1))
        else:
            stacked.append(torch.stack(tensors, dim=1))
    return tuple(stacked)
