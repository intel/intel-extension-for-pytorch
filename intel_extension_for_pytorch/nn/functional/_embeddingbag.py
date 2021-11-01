import torch
import intel_extension_for_pytorch._C as core
import warnings

torch_embedding_bag = torch.embedding_bag

def _embeddingbag(weights, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx):
    if core.embedding_bag_fast_path_sum(weights, per_sample_weights, mode, padding_idx):
        ret = torch.ops.torch_ipex.embedding_bag(weights, indices, offsets, sparse, include_last_offset)
        # torch.embedding_bag expected 4 Tensor returned
        # here we only return 1 tensor since the other three tensors are not needed in our fast path
        ret = [ret, torch.Tensor(), torch.Tensor(), torch.Tensor()]
    else:
        warnings.warn('Fallback to torch.embedding bag')
        ret = torch_embedding_bag(weights, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx)
    return ret

torch.embedding_bag = _embeddingbag
