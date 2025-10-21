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
    "IpexSDP_dropout",
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
    "IpexRmsNorm",
    "IpexSDP_forward",
    "IpexSDP_backward",
    "moe_gemm",
]


def MulAdd(input, other, accumu, alpha=1.0):
    return torch.ops.torch_ipex.mul_add(input, other, accumu, alpha)


def nms(dets, scores, iou_threshold):
    import torchvision

    return torchvision.ops.nms(dets, scores, iou_threshold)


def locations_to_boxes(locations, priors, center_variance, size_variance):
    return torch.ops.torch_ipex.locations_to_boxes(
        locations, priors, center_variance, size_variance
    )


def roi_align(
    input: Tensor,
    boxes: Union[Tensor, List[Tensor]],
    output_size: BroadcastingList2[int],
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> Tensor:
    import torchvision

    return torchvision.ops.roi_align(
        input,
        boxes,
        output_size,
        spatial_scale,
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


def IpexSDP_dropout(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) -> Tensor:
    return torch.ops.torch_ipex.xetla_sdp_dropout(
        query, key, value, attn_mask, dropout_p, is_causal, scale
    )


def IpexSDP_forward(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) -> Tensor:
    return torch.ops.torch_ipex.xetla_sdp_forward(
        query, key, value, attn_mask, dropout_p, is_causal, scale
    )


def IpexSDP_backward(
    out,
    out_grad,
    query,
    key,
    value,
    attn_mask,
    logsumexp,
    seed,
    offset,
    dropout_p,
    is_mask_grad,
    is_causal,
    scale,
) -> Tensor:
    return torch.ops.torch_ipex.xetla_sdp_backward(
        out_grad,
        query,
        key,
        value,
        attn_mask,
        out,
        logsumexp,
        seed,
        offset,
        dropout_p,
        is_mask_grad,
        is_causal,
        scale,
    )


def IpexRmsNorm(input, normalized_shape, weight, epsilon) -> Tensor:
    return torch.ops.torch_ipex.rms_norm_impl(input, normalized_shape, weight, epsilon)


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
    torch.ops.torch_ipex.reshape_and_cache(key, value, key_cache, value_cache, slots)


def IpexPaged_attention(
    out,
    query,
    key_cache,
    value_cache,
    num_queries_per_tokens,
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
        block_tables,
        context_lens,
        num_queries_per_tokens,
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
    block_tables,
    context_lens,
    num_queries_per_tokens,
    head_scale,
    block_size,
    max_context_len,
    alibi_scopes,
    softcap=-1,
):
    return torch.ops.torch_ipex.xetla_paged_attention_v2(
        max_logits,
        exp_sums,
        tmp_out,
        out,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        num_queries_per_tokens,
        head_scale,
        block_size,
        max_context_len,
        alibi_scopes,
        softcap,
    )


def paged_attention_v1(
    out,
    query,
    key_cache,
    value_cache,
    num_queries_per_tokens,
    head_scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_scopes,
    softcap=-1,
):
    return torch.ops.torch_ipex.xetla_paged_attention_v1(
        out,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        num_queries_per_tokens,
        head_scale,
        block_size,
        max_context_len,
        alibi_scopes,
        softcap,
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


def moe_gemm(
    matrix_a,
    matrix_b,
    rows_for_experts,
    n_experts,
    matrix_a_scale_inv=None,
    matrix_b_scale_inv=None,
    bias=None,
    is_mxfp4=False,
    is_fp8=False,
    is_int4=False,
    use_native=False,
):
    """
    Performs MoE (Mixture of Experts) GEMM operation.

    @param matrix_a: 2D Tensor of shape [batch_size, hidden_dim], representing input data.
    @param matrix_b: 3D Tensor of shape [n_experts, hidden_dim, output_dim], representing weights for each expert.
    @param rows_for_experts: 2D Tensor of shape [n_experts], indicating rows belong to each expert.
    @param n_experts: Integer, the number of experts used in the model.
    @param matrix_a_scale_inv: 1D Tensor of shape [n_experts], input scale.
    @param matrix_b_scale_inv: 1D Tensor of shape [n_experts], weights scale.

    @note Fused moe_gemm is expected to work with n_experts < 1024, but to reduce binary size,
          we currently only instantiate templates for n_experts = 8 and n_experts = 16.
          Additional instances can be added in the future as needed.
    """
    use_fused_kernel = (
        torch.xpu.has_2d_block_array() and torch.xpu.has_xmx() and use_native is False
    )
    assert matrix_a_scale_inv is None, "matrix_a_scale_inv is not supported now"
    if use_fused_kernel:
        if is_mxfp4:
            total_m = matrix_a.shape[0]
            gemm_k = matrix_a.shape[1]
            gemm_n = matrix_b.shape[2]
            group_size = gemm_k // matrix_b_scale_inv.shape[1]
            assert group_size == 32, "mxfp4 only support group size 32"
            group_marlin_output = torch.empty(
                total_m, gemm_n, dtype=matrix_a.dtype, device=matrix_a.device
            )
            torch.ops.torch_ipex.group_mm_mxfp4_out_marlin(
                group_marlin_output,
                matrix_a,
                matrix_b,
                matrix_b_scale_inv,
                bias,
                rows_for_experts,
                group_size,
            )
            return group_marlin_output
        elif is_int4:
            total_m = matrix_a.shape[0]
            gemm_k = matrix_a.shape[1]
            gemm_n = matrix_b.shape[2]
            group_size = gemm_k // matrix_b_scale_inv.shape[1]
            group_marlin_output = torch.empty(
                total_m, gemm_n, dtype=matrix_a.dtype, device=matrix_a.device
            )
            torch.ops.torch_ipex.group_mm_int4_out_marlin(
                group_marlin_output,
                matrix_a,
                matrix_b,
                matrix_b_scale_inv,
                bias,
                rows_for_experts,
                None,
                group_size,
            )
            return group_marlin_output
        return torch.ops.torch_ipex.fused_moe_gemm_persistent(
            matrix_a,
            matrix_b,
            matrix_b_scale_inv,
            rows_for_experts,
            n_experts,
        )
    else:
        total_m = matrix_a.shape[0]
        gemm_k = matrix_a.shape[1]
        gemm_n = matrix_b.shape[2]
        output = torch.empty(
            total_m, gemm_n, device=matrix_a.device, dtype=matrix_a.dtype
        )
        rows_for_experts_cpu = rows_for_experts.to("cpu")
        if is_mxfp4:
            group_size = gemm_k // matrix_b_scale_inv.shape[1]
            assert group_size == 32, "mxfp4 only support group size 32"
        start = 0
        for i in range(n_experts):
            end = start + rows_for_experts_cpu[i].item()
            if start == end:
                continue
            if not is_fp8 and not is_mxfp4:
                output[start:end] = torch.mm(matrix_a[start:end], matrix_b[i])
            elif is_fp8:
                assert matrix_a_scale_inv is None
                output[start:end] = torch.ops.torch_ipex.fp8_gemm(
                    matrix_a[start:end],
                    False,
                    matrix_b[i],
                    False,
                    None,
                    matrix_a.dtype,
                    torch.ones(1, device="xpu"),
                    matrix_b_scale_inv[i],
                    None,
                    False,
                )
            else:  # is_mxfp4
                torch.ops.torch_ipex.mm_mxfp4_out_marlin(
                    output[start:end],
                    matrix_a[start:end],
                    matrix_b[i],
                    matrix_b_scale_inv[i],
                    group_size,
                )
            if bias is not None:
                output[start:end] += bias[i]
            start = end

        return output
