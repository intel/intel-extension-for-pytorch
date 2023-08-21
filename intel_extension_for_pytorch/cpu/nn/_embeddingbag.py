import torch
import warnings
import intel_extension_for_pytorch._C as core
from typing import Optional, Tuple

Tensor = torch.Tensor


# TODO: remove indices and offsets because cpu 2.0 release code does not
# contain these code, will add back after release 2.0
def _embedding_bag_fast_path_sum(
    weights: Tensor,
    mode: int = 0,
    scale_grad_by_freq: bool = False,
    per_sample_weights: Optional[Tensor] = None,
    padding_idx: Optional[int] = None,
) -> bool:
    if mode != 0 or scale_grad_by_freq:
        return False
    if weights.stride(1) != 1 or weights.dtype not in (torch.float, torch.bfloat16):
        return False
    if per_sample_weights is not None or padding_idx is not None:
        return False
    return True


torch_embedding_bag = torch.embedding_bag


def patch_emb_bag_cpu_only(func):
    def wrapper(weights: Tensor,
                indices: Tensor,
                offsets: Tensor,
                scale_grad_by_freq: bool = False,
                mode: int = 0,
                sparse: bool = False,
                per_sample_weights: Optional[Tensor] = None,
                include_last_offset: bool = False,
                padding_idx: Optional[int] = None):
        all_cpu = weights.device.type == 'cpu' \
            and indices.device.type == 'cpu' \
            and offsets.device.type == 'cpu'\
            and (True if per_sample_weights is None else per_sample_weights.device.type == 'cpu')
        if all_cpu:
            return func(weights,
                        indices,
                        offsets,
                        scale_grad_by_freq,
                        mode,
                        sparse,
                        per_sample_weights,
                        include_last_offset,
                        padding_idx)
        else:
            return torch_embedding_bag(weights,
                                       indices,
                                       offsets,
                                       scale_grad_by_freq,
                                       mode,
                                       sparse,
                                       per_sample_weights,
                                       include_last_offset,
                                       padding_idx)
    return wrapper


@patch_emb_bag_cpu_only
def _embeddingbag(
    weights: Tensor,
    indices: Tensor,
    offsets: Tensor,
    scale_grad_by_freq: bool = False,
    mode: int = 0,
    sparse: bool = False,
    per_sample_weights: Optional[Tensor] = None,
    include_last_offset: bool = False,
    padding_idx: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # TODO: remove indices and offsets because cpu 2.0 release code does not
    # contain these code, will add back after release 2.0
    if _embedding_bag_fast_path_sum(
        weights, mode, scale_grad_by_freq, per_sample_weights, padding_idx
    ):
        ret = torch.ops.torch_ipex.embedding_bag(weights, indices, offsets, sparse, include_last_offset)
        # torch.embedding_bag expected 4 Tensor returned
        # here we only return 1 tensor since the other three tensors are not needed in our fast path
        ret = (ret, torch.empty(0), torch.empty(0), torch.empty(0))
    else:
        warnings.warn('Fallback to torch.embedding bag')
        ret = torch_embedding_bag(weights,
                                  indices,
                                  offsets,
                                  scale_grad_by_freq,
                                  mode,
                                  sparse,
                                  per_sample_weights,
                                  include_last_offset,
                                  padding_idx)
    return ret


if core._has_cpu():
    torch.embedding_bag = _embeddingbag
