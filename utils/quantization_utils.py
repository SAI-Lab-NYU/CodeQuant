import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict, Callable, Union

from utils.tensor_utils import group, degroup


def int_activation_quantization_pre_hook(module_name: str,
                                         quantization_bit: int,
                                         input_group_size: int,
                                         output_dict: Dict[str, torch.Tensor]
                                         ) -> Callable[[nn.Module, Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]:
    def hook(module: nn.Module, input_args: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        quant_input = activation_quantizer(
            input_args=input_args,
            quantization_bit=quantization_bit,
            group_size=input_group_size
        )

        return (quant_input,)

    return hook


def int_quantizer(tensor: torch.Tensor,
                  quantization_bit: int,
                  dim: int,
                  min_val: float,
                  max_val: float,
                  mask: torch.Tensor,
                  method: str) -> torch.Tensor:
    if method == "minmax":
        if min_val is None and max_val is None:
            min_max_tensor = tensor.clone().masked_fill(~mask, float("inf"))
            min_val, _ = min_max_tensor.min(dim=dim, keepdim=True)
            min_max_tensor = min_max_tensor.masked_fill(~mask, float("-inf"))
            max_val, _ = min_max_tensor.max(dim=dim, keepdim=True)

        scale = (max_val - min_val) / (2**quantization_bit - 1)
        scale.clamp_(min=1e-5)

        q = torch.round((tensor - min_val) / scale).clamp(0, 2**quantization_bit - 1)
        quant_tensor = q * scale + min_val
    elif method == "absmax":
        scale = tensor.clone().masked_fill(~mask, float("-inf")).abs().max(dim=dim, keepdim=True)[0]
        q_max = 2**(quantization_bit - 1) - 1
        scale.clamp_(min=1e-5).div_(q_max)
        quant_tensor = tensor.div(scale).round_().mul_(scale)

    return quant_tensor


def activation_quantizer(input_args: Tuple[torch.Tensor, ...],
                         group_size: int = 1024,
                         quantization_bit: int = 4) -> torch.Tensor:
    org_shape = input_args[0].shape

    input_tensor = input_args[0].reshape(-1, org_shape[-1])
    group_input_tensor, mask = group(input_tensor, group_size=group_size)
    quant_input_tensor = int_quantizer(group_input_tensor, quantization_bit, -1, None, None, mask, method="minmax")
    quant_input_tensor = degroup(quant_input_tensor, group_size, input_tensor.shape)
    return quant_input_tensor.reshape(org_shape)


def weight_quantizer(weight: torch.Tensor,
                     group_size: int = 1024,
                     quantization_bit: int = 4) -> torch.Tensor:
    group_weight, mask = group(weight, group_size)
    quant_group_weight = int_quantizer(group_weight, quantization_bit, -1, None, None, mask, method="absmax")
    quant_weight = degroup(quant_group_weight, group_size, weight.shape)

    return quant_weight
