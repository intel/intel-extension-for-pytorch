import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
import _torch_ipex as core

# # extension for BF16 fast path only


def embeddingbag(weights, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset):
    ret = torch.ops.torch_ipex.embedding_bag(weights, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset)
    if len(ret)==1:
        ret += [torch.Tensor(), torch.Tensor(), torch.Tensor()]
    return ret
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
            for name, param in self._parameters.items():
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
                        if input_name not in self._modules and input_name not in self._parameters:
                            unexpected_keys.append(key)
        else:
            super(_EmbeddingBag, self)._load_from_state_dict(
                state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs)

torch.nn.EmbeddingBag = _EmbeddingBag
