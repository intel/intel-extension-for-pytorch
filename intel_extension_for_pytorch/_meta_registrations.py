import functools
from typing import List, Optional

import torch
import torch.library
from torch._prims_common import IntLike
from .utils.channels_last_1d import to_channels_last_1d


@functools.lru_cache(None)
def get_meta_lib():
    return torch.library.Library("torch_ipex", "IMPL", "Meta")


def register_meta(op_name, overload_name="default"):
    def wrapper(fn):
        get_meta_lib().impl(
            getattr(getattr(torch.ops.torch_ipex, op_name), overload_name), fn
        )
        return fn

    return wrapper


def calc_conv_nd_return_shape(
    input_tensor,
    weight_size,
    stride,
    padding,
    dilation,
    is_transposed,
    groups,
    output_padding,
):
    def _formula(ln: int, p: int, d: int, k: int, s: int) -> int:
        """
        Formula to apply to calculate the length of some dimension of the output

        See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        Args:
            ln: length of the dimension
            p: padding in that dim
            d: dilation in that dim
            k: kernel size in that dim
            s: stride in that dim
        Returns:
            The output length
        """
        return (ln + 2 * p - d * (k - 1) - 1) // s + 1

    def _formula_transposed(ln: int, p: int, d: int, k: int, s: int, op: int) -> int:
        """
        Formula to apply to calculate the length of some dimension of the output
        if transposed convolution is used.
        See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

        Args:
            ln: length of the dimension
            p: padding in that dim
            d: dilation in that dim
            k: kernel size in that dim
            s: stride in that dim
            op: output padding in that dim

        Returns:
            The output length
        """
        return (ln - 1) * s - 2 * p + d * (k - 1) + op + 1

    kernel_size = weight_size[2:]
    dims = input_tensor.shape[2:]
    if is_transposed:
        out_channels = groups * weight_size[1]
    else:
        out_channels = weight_size[0]

    ret_shape = [input_tensor.shape[0], out_channels]
    if isinstance(stride, IntLike):
        stride = [stride] * len(dims)
    elif len(stride) == 1:
        stride = [stride[0]] * len(dims)

    if isinstance(padding, IntLike):
        padding = [padding] * len(dims)
    elif len(padding) == 1:
        padding = [padding[0]] * len(dims)

    if isinstance(dilation, IntLike):
        dilation = [dilation] * len(dims)
    elif len(dilation) == 1:
        dilation = [dilation[0]] * len(dims)

    output_padding_list: Optional[List[int]] = None
    if output_padding:
        if isinstance(output_padding, IntLike):
            output_padding_list = [output_padding] * len(dims)
        elif len(output_padding) == 1:
            output_padding_list = [output_padding[0]] * len(dims)
        else:
            output_padding_list = output_padding

    for i in range(len(dims)):
        # If output_padding is present, we are dealing with a transposed convolution
        if output_padding_list:
            ret_shape.append(
                _formula_transposed(
                    dims[i],
                    padding[i],
                    dilation[i],
                    kernel_size[i],
                    stride[i],
                    output_padding_list[i],
                )
            )
        else:
            ret_shape.append(
                _formula(dims[i], padding[i], dilation[i], kernel_size[i], stride[i])
            )

    return ret_shape


def is_channels_last(ten):
    return torch._prims_common.suggest_memory_format(ten) == torch.channels_last


def is_channels_last_3d(ten):
    return torch._prims_common.suggest_memory_format(ten) == torch.channels_last_3d


@register_meta("reshape_and_cache")
def meta_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping):
    return None


@register_meta("single_query_cached_kv_attention")
def meta_single_query_cached_kv_attention(
    output,
    query,
    key_cache,
    value_cache,
    head_mapping,
    scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_slopes,
):
    return None


@register_meta("convolution_forward")
def meta_convolution_forward(
    input,
    weight,
    bias,
    W_prepack,
    kernel_size,
    padding,
    stride,
    dilation,
    weight_channels_last,
):
    shape_out = calc_conv_nd_return_shape(
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        False,
        None,
        None,
    )

    use_channels_last = (
        is_channels_last(input) or is_channels_last_3d(input) or weight_channels_last
    )
    memory_format = torch.contiguous_format
    if use_channels_last:
        if input.dim() == 4:
            memory_format = torch.channels_last
        elif input.dim() == 5:
            memory_format = torch.channels_last_3d

    out = input.new_empty(shape_out)
    out = out.to(memory_format=memory_format)  # type: ignore[call-overload]

    if input.dim() == 3:
        out = to_channels_last_1d(out)

    return out


@register_meta("convolution_backward")
def meta_convolution_backward(
    input,
    weight,
    bias,
    grad_output,
    out_mask,
    W_prepack,
    weight_channels_last,
):
    use_channels_last = (
        is_channels_last(input) or is_channels_last_3d(input) or weight_channels_last
    )
    memory_format = torch.contiguous_format
    if use_channels_last:
        if input.dim() == 4:
            memory_format = torch.channels_last
        elif input.dim() == 5:
            memory_format = torch.channels_last_3d

    backend_grad_input = None
    backend_grad_weight = None
    backend_grad_bias = None

    if out_mask[0]:
        backend_grad_input = grad_output.new_empty(input.size()).to(
            memory_format=memory_format
        )
    if out_mask[1]:
        backend_grad_weight = grad_output.new_empty(weight.size())
    if out_mask[2]:
        backend_grad_bias = grad_output.new_empty(bias.size())

    return (backend_grad_input, backend_grad_weight, backend_grad_bias)


@register_meta("conv_transpose")
def meta_conv_transpose(
    input,
    weight,
    bias_opt,
    W_prepack,
    weight_size,
    padding,
    output_padding,
    stride,
    dilation,
    groups,
    weight_channels_last,
):
    shape_out = calc_conv_nd_return_shape(
        input,
        weight_size,
        stride,
        padding,
        dilation,
        True,
        groups,
        output_padding,
    )

    use_channels_last = (
        is_channels_last(input) or is_channels_last_3d(input) or weight_channels_last
    )
    memory_format = torch.contiguous_format
    if use_channels_last:
        if input.dim() == 4:
            memory_format = torch.channels_last
        elif input.dim() == 5:
            memory_format = torch.channels_last_3d

    out = input.new_empty(shape_out)
    out = out.to(memory_format=memory_format)  # type: ignore[call-overload]
    return out


@register_meta("conv_transpose_backward")
def meta_conv_transpose_backward(
    input,
    weight,
    bias,
    grad_output,
    out_mask,
    W_prepack,
    weight_channels_last,
):
    use_channels_last = (
        is_channels_last(input) or is_channels_last_3d(input) or weight_channels_last
    )
    memory_format = torch.contiguous_format
    if use_channels_last:
        if input.dim() == 4:
            memory_format = torch.channels_last
        elif input.dim() == 5:
            memory_format = torch.channels_last_3d

    backend_grad_input = None
    backend_grad_weight = None
    backend_grad_bias = None

    if out_mask[0]:
        backend_grad_input = grad_output.new_empty(input.size()).to(
            memory_format=memory_format
        )
    if out_mask[1]:
        backend_grad_weight = grad_output.new_empty(weight.size())
    if out_mask[2]:
        backend_grad_bias = grad_output.new_empty(bias.size())

    return (backend_grad_input, backend_grad_weight, backend_grad_bias)


@register_meta("ipex_linear")
def meta_ipex_linear(
    input,
    weight,
    bias,
    W_prepack,
    out_features,
):
    return input.new_empty((*input.shape[:-1], out_features))


@register_meta("linear_backward")
def meta_linear_backward(
    input,
    weight,
    bias,
    grad_output,
    out_mask,
    W_prepack,
):
    backend_grad_input = None
    backend_grad_weight = None
    backend_grad_bias = None

    if out_mask[0]:
        backend_grad_input = grad_output.new_empty(input.size())
    if out_mask[1]:
        backend_grad_weight = grad_output.new_empty(weight.size())
    if out_mask[2]:
        backend_grad_bias = grad_output.new_empty(bias.size())

    return (backend_grad_input, backend_grad_weight, backend_grad_bias)


@register_meta("ipex_MKLSGEMM")
def meta_ipex_MKLSGEMM(
    input,
    weight,
    bias,
    W_prepack,
    out_features,
):
    return input.new_empty((*input.shape[:-1], out_features))


@register_meta("ipex_linear_eltwise")
def meta_ipex_linear_eltwise(
    input,
    weight,
    bias,
    eltwise,
    W_prepack,
    out_features,
):
    return input.new_empty((*input.shape[:-1], out_features))


@register_meta("linear_eltwise_backward")
def meta_linear_eltwise_backward(
    input,
    weight,
    bias,
    output,
    eltwise,
    grad_output,
    out_mask,
    W_prepack,
):
    backend_grad_input = None
    backend_grad_weight = None
    backend_grad_bias = None

    if out_mask[0]:
        backend_grad_input = grad_output.new_empty(input.size())
    if out_mask[1]:
        backend_grad_weight = grad_output.new_empty(weight.size())
    if out_mask[2]:
        backend_grad_bias = grad_output.new_empty(bias.size())

    return (backend_grad_input, backend_grad_weight, backend_grad_bias)


@register_meta("embedding_bag")
def meta_embedding_bag(
    weight,
    indices,
    offsets,
    sparse,
    include_last_offset,
):
    num_bags = offsets.shape[0]
    if indices.dim() == 2:
        num_bags = indices.shape[0]
    shape_out = [num_bags, weight.shape[1]]
    return weight.new_empty(shape_out)


@register_meta("ipex_lstm")
def meta_ipex_lstm(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout_p,
    train,
    bidirectional,
    batch_first,
):
    shape_out = [*input.shape[:-1]]
    shape_out.append(hx[0].shape[2] * 2 if bidirectional else hx[0].shape[2])
    out = input.new_empty(shape_out)
    hy = hx[0].new_empty(hx[0].size())
    cy = hx[1].new_empty(hx[1].size())
    return (out, hy, cy)


@register_meta("ROIAlign_forward")
def meta_ROIAlign_forward(
    input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned
):
    return input.new_empty(
        (rois.shape[0], input.shape[1], pooled_height, pooled_width)
    ).to(memory_format=torch._prims_common.suggest_memory_format(input))


@register_meta("ROIAlign_backward")
def meta_ROIAlign_backward(
    grad,
    rois,
    spatial_scale,
    pooled_height,
    pooled_width,
    batch_size,
    channels,
    height,
    width,
    sampling_ratio,
    aligned,
    is_channels_last,
):
    return grad.new_empty((batch_size, channels, height, width)).to(
        memory_format=(
            torch.channels_last if is_channels_last else torch.contiguous_format
        )
    )


@register_meta("batch_norm_forward")
def meta_batch_norm_forward(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    train,
    momentum,
    eps,
):
    memory_format = torch._prims_common.suggest_memory_format(input)
    output = input.new_empty(input.shape).to(memory_format=memory_format)
    out_running_mean = input.new_empty(input.shape[1])
    out_running_var = input.new_empty(input.shape[1])
    return (output, out_running_mean, out_running_var)


@register_meta("batch_norm_backward")
def meta_batch_norm_backward(
    grad_output,
    input,
    weight,
    save_mean,
    save_var,
    train,
    eps,
    grad_input_mask,
):
    backend_grad_input = None
    backend_grad_weight = None
    backend_grad_bias = None
    if grad_input_mask[0]:
        memory_format = torch._prims_common.suggest_memory_format(input)
        backend_grad_input = input.new_empty(input.shape).to(
            memory_format=memory_format
        )
    if grad_input_mask[1]:
        backend_grad_weight = weight.new_empty(weight.shape)
    if grad_input_mask[2]:
        backend_grad_bias = weight.new_empty(weight.shape[0])
    return (backend_grad_input, backend_grad_weight, backend_grad_bias)


@register_meta("bmm_add")
def meta_bmm_add(
    input,
    batch1,
    batch2,
    alpha,
):
    return batch1.new_empty((*batch1.shape[:-1], batch2.shape[-1]))


@register_meta("add_softmax_")
def meta_add_softmax_(
    input1,
    input2,
):
    return input1


@register_meta("cumsum")
def meta_cumsum(
    input,
    dim,
    dtype=None,
):
    return input.new_empty(input.shape)


@register_meta("tpp_linear")
def meta_tpp_linear(
    input,
    weight,
    out_features,
):
    return input.new_empty((*input.shape[:-1], out_features))


@register_meta("tpp_linear_bias")
def meta_tpp_linear_bias(
    input,
    weight,
    bias,
    out_features,
):
    return input.new_empty((*input.shape[:-1], out_features))


@register_meta("choose_tpp_linear_weight")
def meta_choose_tpp_linear_weight(x, weight, weight_for_large_batch):
    M = x.numel() // x.size(-1)
    return (
        weight_for_large_batch
        if weight_for_large_batch is not None and M >= 256
        else weight
    )


@register_meta("tpp_linear_gelu")
def meta_tpp_linear_gelu(
    input,
    weight,
    bias,
    out_features,
):
    return input.new_empty((*input.shape[:-1], out_features))


@register_meta("tpp_linear_add_add")
def meta_tpp_linear_add_add(
    input,
    input1,
    input2,
    weight,
    bias,
    scale,
    out_features,
):
    return input.new_empty((*input.shape[:-1], out_features))


@register_meta("tpp_linear_relu")
def meta_tpp_linear_relu(
    input,
    weight,
    bias,
    out_features,
):
    return input.new_empty((*input.shape[:-1], out_features))


@register_meta("tpp_linear_silu")
def meta_tpp_linear_silu(
    input,
    weight,
    bias,
    out_features,
):
    return input.new_empty((*input.shape[:-1], out_features))


@register_meta("tpp_linear_add")
def meta_tpp_linear_add(
    input,
    input1,
    weight,
    bias,
    scale,
    out_features,
):
    return input.new_empty((*input.shape[:-1], out_features))


@register_meta("tpp_linear_mul")
def meta_tpp_linear_mul(
    input,
    input1,
    weight,
    bias,
    out_features,
):
    return input.new_empty((*input.shape[:-1], out_features))


@register_meta("masked_multihead_self_attention")
def meta_masked_multihead_self_attention(
    query,
    key,
    value,
    key_cache,
    value_cache,
    beam_idx,
    seq_info,
    scale_attn,
    max_positions,
    head_mask,
    attention_mask,
    add_casual_mask=None,
):
    attn_output = query.new_empty(
        (query.shape[0], query.shape[2], query.shape[1], query.shape[3])
    )
    if query.dtype == torch.bfloat16:
        attn_output.as_strided_(
            attn_output.shape,
            (
                query.shape[1] * query.shape[2] * query.shape[3],
                query.shape[3],
                query.shape[2] * query.shape[3],
                1,
            ),
        )
    attn_weights = None
    key_cache_out = query.new_empty(
        (key_cache.shape[0], key_cache.shape[1], key.shape[2], key.shape[3])
    )
    value_cache_out = query.new_empty(
        (value_cache.shape[0], value_cache.shape[1], value.shape[2], value.shape[3])
    )
    beam_idx_out = query.new_empty(beam_idx.shape)
    return (attn_output, attn_weights, key_cache_out, value_cache_out, beam_idx_out)


@register_meta("rotary_position_embedding")
def meta_rotary_position_embedding(
    t_in,
    t_emb_pos,
    t_pos,
    N,
    H,
    offset,
    rotary_ndims,
):
    ndims = t_in.dim()
    stride_s = t_in.stride(1)
    batch = t_in.shape[0]
    seq_len = t_in.shape[1]
    concat_qkv = False
    if ndims == 3 and stride_s > N * H:
        concat_qkv = True
        kv_head = (t_in.shape[2] - N * H) // (2 * H)
    if not concat_qkv:
        return (
            t_in.new_empty(t_in.shape).contiguous(),
            None,
            None,
        )
    else:
        return (
            torch.empty(batch, seq_len, N, H, dtype=t_in.dtype, device=t_in.device),
            torch.empty(
                batch, seq_len, kv_head, H, dtype=t_in.dtype, device=t_in.device
            ),
            torch.empty(
                batch, seq_len, kv_head, H, dtype=t_in.dtype, device=t_in.device
            ),
        )


@register_meta("rmsnorm")
def meta_rmsnorm(
    input,
    weight,
    eps,
):
    return input.new_empty(input.shape)
