import torch
from torch import nn
from torch.autograd import Function
import _torch_ipex as core

'''
# extension for BF16 fast path only
torch_embedding_bag = torch.embedding_bag
def embeddingbag(weights, inputs, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset):
    if weights.dtype == torch.float:
        ret = torch_embedding_bag(weights, inputs, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset)
    elif sparse and mode == 0 and per_sample_weights is None and scale_grad_by_freq == False:
        ret = EmbeddingBagFunction.apply(weights, inputs.contiguous(), offsets.contiguous())
        ret = (ret, None, None, None)
    else:
        assert(0, "unimplement embeddingbag path in extension")
'''
def embeddingbag(weights, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset):
    ret = EmbeddingBagFunction.apply(weights, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset)
    return ret


class EmbeddingBagFunction(Function):
    '''
    @staticmethod
    def forward(ctx, weights, inputs, offsets):
        ctx.save_for_backward(weights, inputs, offsets)
        output = core.embedding_bag_forward(weights, inputs, offsets)
        return output
    '''
    @staticmethod
    def forward(ctx, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset):
        ctx.scale_grad_by_freq = scale_grad_by_freq
        ctx.mode = mode
        ctx.sparse = sparse
        ctx.num_weight = weight.size(0)
        ctx.save_for_backward(indices, offsets, per_sample_weights)
        ret = core.embedding_bag_forward(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset)
        return ret

    '''
    @staticmethod
    def backward(ctx, grad_out):
        weights, inputs, offsets = ctx.saved_tensors
        grad_weight = core.embedding_bag_backward(grad_out, weights, inputs, offsets)
        return (grad_weight, None, None)
    '''
    @staticmethod
    def backward(ctx, grad, offset2bag, bag_size, maximum_indices):
        indices, offsets, per_sample_weights = ctx.saved_tensors
        grad_weight = core.embedding_bag_backward(grad, indices, offsets, offset2bag, bag_size, maximum_indices, ctx.num_weight, ctx.scale_grad_by_freq, ctx.mode, ctx.sparse, per_sample_weights)
        return grad_weight, None, None, None, None, None, None, None

torch.embedding_bag = embeddingbag
