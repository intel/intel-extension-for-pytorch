import torch
from torch import nn
from torch.autograd import Function
import intel_pytorch_extension as ipex
import torch_ipex._C as core
from typing import Callable, List, Optional, Tuple

# # extension for BF16 fast path only
Tensor = torch.Tensor
torch_embedding_bag = torch.embedding_bag

def ipex_embedding_bag(
    weight: Tensor,
    input: Tensor,
    offsets: Optional[Tensor] = None,
    scale_grad_by_freq: bool = False,
    mode: int = 0,
    sparse: bool = False,
    per_sample_weights: Optional[Tensor] = None,
    include_last_offset: bool = False,
    padding_idx: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if weight.device.type in ipex.DEVICE:
        assert padding_idx == None
        ret = torch.ops.torch_ipex.embedding_bag(weight, input, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset)
        return ret[0], torch.rand(0), torch.rand(0), torch.rand(0)
    else:
        return torch_embedding_bag(weight, input, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx)

torch.embedding_bag = ipex_embedding_bag
