import torch
import intel_extension_for_pytorch._C as core
from typing import Optional, Tuple
import warnings

Tensor = torch.Tensor


def _embedding_bag_fast_path_sum(
    weights: Tensor,
    indices: Tensor,
    offsets: Tensor,
    mode: int = 0,
    scale_grad_by_freq: bool = False,
    per_sample_weights: Optional[Tensor] = None,
    padding_idx: Optional[int] = None,
) -> Tuple[bool, str]:
    if indices.dtype != torch.int64 or offsets.dtype != torch.int64:
        return False, "IPEX embeddingbag only support int32 offsets/indices."
    if mode != 0 or scale_grad_by_freq:
        return (
            False,
            "IPEX embeddingbag only support mode='sum' and scale_grad_by_freq=False.",
        )
    if weights.stride(1) != 1 or weights.dtype not in (torch.float, torch.bfloat16):
        return False, "IPEX embeddingbag only support fp32/bf16 weights."
    if per_sample_weights is not None or padding_idx is not None:
        return (
            False,
            "IPEX embeddingbag only support per_sample_weights/padding_idx = None.",
        )
    return True, "supported"


torch_embedding_bag = torch.embedding_bag


def patch_emb_bag_cpu_only(func):
    def wrapper(
        weights: Tensor,
        indices: Tensor,
        offsets: Tensor,
        scale_grad_by_freq: bool = False,
        mode: int = 0,
        sparse: bool = False,
        per_sample_weights: Optional[Tensor] = None,
        include_last_offset: bool = False,
        padding_idx: Optional[int] = None,
    ):
        all_cpu = (
            weights.device.type == "cpu"
            and indices.device.type == "cpu"
            and offsets.device.type == "cpu"
            and (
                True
                if per_sample_weights is None
                else per_sample_weights.device.type == "cpu"
            )
        )
        if all_cpu:
            return func(
                weights,
                indices,
                offsets,
                scale_grad_by_freq,
                mode,
                sparse,
                per_sample_weights,
                include_last_offset,
                padding_idx,
            )
        else:
            return torch_embedding_bag(
                weights,
                indices,
                offsets,
                scale_grad_by_freq,
                mode,
                sparse,
                per_sample_weights,
                include_last_offset,
                padding_idx,
            )

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
    supported, msg = _embedding_bag_fast_path_sum(
        weights,
        indices,
        offsets,
        mode,
        scale_grad_by_freq,
        per_sample_weights,
        padding_idx,
    )
    if supported:
        ret = torch.ops.torch_ipex.embedding_bag(
            weights, indices, offsets, sparse, include_last_offset
        )
        # torch.embedding_bag expected 4 Tensor returned
        # here we only return 1 tensor since the other three tensors are not needed in our fast path
        ret = (ret, torch.empty(0), torch.empty(0), torch.empty(0))
    else:
        r"""
        Cannot use logging.logger here
        File "torch/jit/_script.py", line 1395, in script
            fn = torch._C._jit_script_compile(
        RuntimeError:
        attribute lookup is not defined on python value of type 'Logger':
        File "intel_extension_for_pytorch/cpu/nn/_embeddingbag.py", line 116
                ret = (ret, torch.empty(0), torch.empty(0), torch.empty(0))
            else:
                logger.warning(
                ~~~~~~~~~~~~~~ <--- HERE
                    msg + " Fallback to torch.embedding bag.", _type=WarningType.NotSupported
                )
        '_embeddingbag' is being compiled since it was called from 'wrapper'
        """
        warnings.warn("[NotSupported]" + msg + " Fallback to torch.embedding bag.")
        ret = torch_embedding_bag(
            weights,
            indices,
            offsets,
            scale_grad_by_freq,
            mode,
            sparse,
            per_sample_weights,
            include_last_offset,
            padding_idx,
        )
    return ret


if core._has_cpu():
    torch.embedding_bag = _embeddingbag
