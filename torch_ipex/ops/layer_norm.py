import torch
import torch_ipex._C as core
from typing import Optional

torch_layer_norm = torch.layer_norm

def _layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enabled):
    if input.device.type != "xpu":
        return torch_layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enabled)
    else:
        return torch.ops.torch_ipex.layer_norm(input, normalized_shape, weight, bias, eps)

torch.layer_norm = _layer_norm
