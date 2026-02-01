import torch
import torch.nn.functional as F

from typing import Tuple


def group(tensor: torch.Tensor,
          group_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # activation: dim0 = batch * token, dim1 = channel
    # weight: dim0 = output_channel, dim1 = input_channel
    if group_size == -1:
        group_size = tensor.shape[1]

    dim0, dim1 = tensor.shape

    if dim1 < group_size:
        return tensor, torch.ones_like(tensor, dtype=torch.bool)
    elif dim1 % group_size == 0:
        group_tensor = tensor.reshape(-1, group_size)
        return group_tensor, torch.ones_like(tensor, dtype=torch.bool).reshape(-1, group_size)
    else:
        mask = torch.ones_like(tensor, dtype=torch.bool)
        residual_len = dim1 % group_size
        pad_len = group_size - residual_len
        group_tensor = F.pad(tensor, (0, pad_len), mode="constant", value=0.0).reshape(-1, group_size)
        mask = F.pad(mask, (0, pad_len), mode="constant", value=False).reshape(-1, group_size)
        return group_tensor, mask


def degroup(group_tensor: torch.Tensor,
            group_size: int,
            shape: tuple) -> torch.Tensor:
    # activation: dim0 = batch * token, dim1 = channel
    # weight: dim0 = output_channel, dim1 = input_channel
    if group_size == -1:
        group_size = shape[1]

    dim0, dim1 = shape

    if dim1 < group_size:
        return group_tensor
    elif dim1 % group_size == 0:
        return group_tensor.reshape(shape)
    else:
        restore_group_tensor = group_tensor.reshape(dim0, -1)
        tensor = restore_group_tensor[:, :dim1]

        return tensor
