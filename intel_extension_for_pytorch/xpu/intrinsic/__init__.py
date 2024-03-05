import torch
from torch.nn.modules.utils import _pair
from torch import nn, Tensor
from torch.jit.annotations import BroadcastingList2
from typing import List, Union, Dict
from .modules import EMA
from .modules import TransducerLoss, clip_grad_norm_, clip_grad_norm
import intel_extension_for_pytorch

__all__ = [
    "TransducerLoss",
    "nms",
    "locations_to_boxes",
    "roi_align",
    "IpexSDP",
    "IpexSDP_Index",
    "EMA",
    "clip_grad_norm_",
    "clip_grad_norm",
    "varlen_fwd",
    "reshape_and_cache",
    "paged_attention_v1",
    "paged_attention_v2",
    "copy_blocks",
    "swap_blocks",
    "IpexPaged_attention",
]


def MulAdd(input, other, accumu, alpha=1.0):
    return torch.ops.torch_ipex.mul_add(input, other, accumu, alpha)


def nms(dets, scores, iou_threshold):
    return torch.ops.torch_ipex.nms(dets, scores, iou_threshold)


def locations_to_boxes(locations, priors, center_variance, size_variance):
    return torch.ops.torch_ipex.locations_to_boxes(
        locations, priors, center_variance, size_variance
    )


def check_roi_boxes_shape(boxes: Union[Tensor, List[Tensor]]):
    if isinstance(boxes, (list, tuple)):
        for _tensor in boxes:
            torch._assert(
                _tensor.size(1) == 4,
                "The shape of the tensor in the boxes list is not correct as List[Tensor[L, 4]]",
            )
    elif isinstance(boxes, torch.Tensor):
        torch._assert(
            boxes.size(1) == 5, "The boxes tensor shape is not correct as Tensor[K, 5]"
        )
    else:
        torch._assert(
            False, "boxes is expected to be a Tensor[L, 5] or a List[Tensor[K, 4]]"
        )
    return


def convert_boxes_to_roi_format(boxes: List[Tensor]) -> Tensor:
    concat_boxes = _cat(list(boxes), dim=0)
    temp = []
    for i, b in enumerate(boxes):
        temp.append(torch.full_like(b[:, :1], i))
    ids = _cat(temp, dim=0)
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


def roi_align(
    input: Tensor,
    boxes: Union[Tensor, List[Tensor]],
    output_size: BroadcastingList2[int],
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> Tensor:
    check_roi_boxes_shape(boxes)
    rois = boxes
    output_size = _pair(output_size)
    if not isinstance(rois, torch.Tensor):
        rois = convert_boxes_to_roi_format(rois)
    return torch.ops.torch_ipex.roi_align(
        input,
        rois,
        spatial_scale,
        output_size[0],
        output_size[1],
        sampling_ratio,
        aligned,
    )


def IpexSDP(
    query,
    key,
    value,
    alibi=None,
    bias=None,
    head_mask=None,
    alpha=1.0,
    beta=1.0,
    dropout_p=0.0,
    is_causal=False,
    seq_last=False,
) -> Tensor:
    return torch.ops.torch_ipex.xetla_fsdp_forward_atten_mask_alibi_strided(
        query,
        key,
        value,
        alibi,
        bias,
        head_mask,
        alpha,
        beta,
        dropout_p,
        is_causal,
        seq_last,
    )


def IpexSDP_Index(
    query,
    key,
    value,
    key_cache,
    value_cache,
    index,
    alibi,
    attn_mask,
    head_mask,
    timestep,
    alpha,
    beta,
    dropout_p=0.0,
    is_causal=False,
) -> Tensor:
    return torch.ops.torch_ipex.xetla_fsdp_index_forward(
        query,
        key,
        value,
        key_cache,
        value_cache,
        index,
        alibi,
        attn_mask,
        head_mask,
        timestep,
        alpha,
        beta,
        dropout_p,
        is_causal,
    )


def varlen_fwd(
    query,  # [total_q, num_head, head_size]
    key,  # [total_k, num_head_k, head_size]
    value,  # [total_k, num_head_k, head_size]
    out,  # [total_q, num_head, head_size]
    seqlen_q,  # [batch_size + 1]
    seqlen_k,  # [batch_size + 1]
    max_seqlen_q,
    max_seqlen_k,
    pdropout,
    softmax_scale,
    zero_tensors,
    is_causal,
    return_softmax,
    gen_,
):
    assert return_softmax is False, "ipex do not support return_softmax option"
    assert gen_ is None, "ipex do not support custom random generator"
    assert zero_tensors is False, "ipex varlen_fwd do not support zero tensors"
    total_q, num_head, head_size = query.size()
    total_k, num_head_k, _ = key.size()
    batch_size = seqlen_q.size(0) - 1
    seqlen_q_ = seqlen_q.clone()
    seqlen_q_[:batch_size] = seqlen_q[1:]
    seqlen_q = (seqlen_q_ - seqlen_q)[:batch_size]
    seqlen_k_ = seqlen_k.clone()
    seqlen_k_[:batch_size] = seqlen_k[1:]
    seqlen_k = (seqlen_k_ - seqlen_k)[:batch_size]

    pad_q = torch.zeros(
        [batch_size, max_seqlen_q, num_head, head_size],
        dtype=query.dtype,
        device=query.device,
    )
    pad_k = torch.zeros(
        [batch_size, max_seqlen_k, num_head_k, head_size],
        dtype=key.dtype,
        device=key.device,
    )
    pad_v = torch.zeros(
        [batch_size, max_seqlen_k, num_head_k, head_size],
        dtype=value.dtype,
        device=value.device,
    )
    q_mask = torch.arange(0, max_seqlen_q, device=query.device)[None, :].repeat(
        batch_size, 1
    )
    q_mask = q_mask < seqlen_q[:, None].repeat(1, q_mask.size(-1))
    k_mask = torch.arange(0, max_seqlen_k, device=key.device)[None, :].repeat(
        batch_size, 1
    )
    k_mask = k_mask < seqlen_k[:, None].repeat(1, k_mask.size(-1))
    align_mask_seqlen = (max_seqlen_k + 63) // 64 * 64
    attn_mask = torch.empty(
        [batch_size, 1, 1, align_mask_seqlen], dtype=torch.float16, device=query.device
    ).fill_(float("-inf"))
    attn_mask[:, :, :, :max_seqlen_k].masked_fill_(k_mask[:, None, None, :], 0)

    pad_q[q_mask] = query
    pad_k[k_mask] = key
    pad_v[k_mask] = value

    pad_q = pad_q.permute(0, 2, 1, 3)
    pad_k = pad_k.permute(0, 2, 1, 3)
    pad_v = pad_v.permute(0, 2, 1, 3)

    out_ = torch.ops.torch_ipex.xetla_fsdp_forward_atten_mask_alibi_strided(
        pad_q,
        pad_k,
        pad_v,
        None,
        attn_mask,
        None,
        softmax_scale,
        1.0,
        pdropout,
        is_causal,
        False,
    )
    out_ = out_.permute(0, 2, 1, 3)
    if out is None:
        return out_[q_mask]
    else:
        out.copy_(out_[q_mask])


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots: torch.Tensor,
):
    return torch.ops.torch_ipex.reshape_and_cache(
        key, value, key_cache, value_cache, slots
    )


def IpexPaged_attention(
    out,
    query,
    key_cache,
    value_cache,
    head_mapping,
    block_tables,
    context_lens,
    head_scale,
    block_size,
    max_context_len,
    alibi_scopes,
):
    return torch.ops.torch_ipex.xetla_paged_attention_v1(
        out,
        query,
        key_cache,
        value_cache,
        head_mapping,
        block_tables,
        context_lens,
        head_scale,
        block_size,
        max_context_len,
        alibi_scopes,
    )


def paged_attention_v2(
    out,
    exp_sums,
    max_logits,
    tmp_out,
    query,
    key_cache,
    value_cache,
    head_mapping,
    block_tables,
    context_lens,
    head_scale,
    block_size,
    max_context_len,
    alibi_scopes,
):
    return torch.ops.torch_ipex.xetla_paged_attention_v2(
        max_logits,
        exp_sums,
        tmp_out,
        out,
        query,
        key_cache,
        value_cache,
        head_mapping,
        block_tables,
        context_lens,
        head_scale,
        block_size,
        max_context_len,
        alibi_scopes,
    )


def paged_attention_v1(
    out,
    query,
    key_cache,
    value_cache,
    head_mapping,
    head_scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_scopes,
):
    return torch.ops.torch_ipex.xetla_paged_attention_v1(
        out,
        query,
        key_cache,
        value_cache,
        head_mapping,
        block_tables,
        context_lens,
        head_scale,
        block_size,
        max_context_len,
        alibi_scopes,
    )


def copy_blocks(key_caches, value_caches, block_mapping):
    assert isinstance(block_mapping, Dict) or isinstance(
        block_mapping, torch.Tensor
    ), "We only support block_mapping as dict or torch tensor"
    if isinstance(block_mapping, Dict):
        block_mapping_tensor = []
        for key, values in block_mapping.items():
            if hasattr(values, "__iter__"):
                for value in values:
                    block_mapping_tensor.append([key, value])
            else:
                block_mapping_tensor.append([key, value])
        block_mapping = torch.tensor(
            block_mapping_tensor, device="xpu", dtype=torch.int64
        )
    return torch.ops.torch_ipex.copy_blocks(key_caches, value_caches, block_mapping)


def swap_blocks(src, dst, block_mapping):
    assert isinstance(block_mapping, Dict) or isinstance(
        block_mapping, torch.Tensor
    ), "We only support block_mapping as dict or torch tensor"
    if isinstance(block_mapping, Dict):
        block_mapping_tensor = []
        for key, values in block_mapping.items():
            if hasattr(values, "__iter__"):
                for value in values:
                    block_mapping_tensor.append([key, value])
            else:
                block_mapping_tensor.append([key, value])
        block_mapping = torch.tensor(
            block_mapping_tensor, device="xpu", dtype=torch.int64
        )
    return torch.ops.torch_ipex.swap_blocks(src, dst, block_mapping)
