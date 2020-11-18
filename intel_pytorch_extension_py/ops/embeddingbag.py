import torch
from torch import nn
from torch.autograd import Function
import _torch_ipex as core

# # extension for BF16 fast path only


def embeddingbag(weights, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset):
    ret = torch.ops.torch_ipex.embedding_bag(weights, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset)
    if len(ret)==1:
        ret += [torch.Tensor(), torch.Tensor(), torch.Tensor()]
    return ret
torch.embedding_bag = embeddingbag
