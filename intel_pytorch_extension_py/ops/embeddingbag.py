import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
import _torch_ipex as core
import itertools


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

def normal_(tensor, mean=0., std=1.):
    with torch.no_grad():
        return tensor.normal_(mean, std)

class _EmbeddingBag(nn.Module):
    __constants__ = ['num_embeddings', 'embedding_dim', 'max_norm', 'norm_type',
                     'scale_grad_by_freq', 'mode', 'sparse', 'include_last_offset']

    def __init__(self, num_embeddings, embedding_dim,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 mode='mean', sparse=False, _weight=None, include_last_offset=False, eval_mode=False):
        super(_EmbeddingBag, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.eval_mode = eval_mode
        if _weight is None:
            if self.eval_mode:
                self.weight = nn.Parameter()
            else:
                self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
                self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = Parameter(_weight)
        self.mode = mode
        self.sparse = sparse
        self.include_last_offset = include_last_offset

    def reset_parameters(self):
        normal_(self.weight)

    def forward(self, input, offsets=None, per_sample_weights=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        return F.embedding_bag(input, self.weight, offsets,
                               self.max_norm, self.norm_type,
                               self.scale_grad_by_freq, self.mode, self.sparse,
                               per_sample_weights, self.include_last_offset)

    def extra_repr(self):
        s = '{num_embeddings}, {embedding_dim}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        s += ', mode={mode}'
        return s.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        if self.eval_mode:
            local_name_params = itertools.chain(self._parameters.items())
            local_state = {k: v for k, v in local_name_params}
 
            for name, param in local_state.items():
                key = prefix + name
                if key in state_dict:
                    input_param = state_dict[key]
 
                    if input_param.shape != (self.num_embeddings, self.embedding_dim):
                        # local shape should match the one in checkpoint
                        error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                          'the shape in current model is {}.'
                                          .format(key, input_param.shape, param.shape))
                        continue
 
                    self._parameters[name] = input_param

                elif strict:
                    missing_keys.append(key)
 
            if strict:
                for key in state_dict.keys():
                    if key.startswith(prefix):
                        input_name = key[len(prefix):]
                        input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                        if input_name not in self._modules and input_name not in local_state:
                            unexpected_keys.append(key)
        else:
            super(_EmbeddingBag, self)._load_from_state_dict(
                state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs)

torch.nn.EmbeddingBag = _EmbeddingBag
