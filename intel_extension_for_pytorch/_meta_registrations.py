import functools
from typing import List, Optional

import torch
import torch.library
from torch._prims_common import IntLike

@functools.lru_cache(None)
def get_meta_lib():
    return torch.library.Library("torch_ipex", "IMPL", "Meta")

def register_meta(op_name, overload_name="default"):
    def wrapper(fn):
        get_meta_lib().impl(getattr(getattr(torch.ops.torch_ipex, op_name), overload_name), fn)
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

    use_channels_last = is_channels_last(input) or is_channels_last_3d(input) or weight_channels_last
    memory_format = torch.contiguous_format
    if use_channels_last:
        if input.dim() == 4:
            memory_format = torch.channels_last
        elif input.dim() == 5:
            memory_format = torch.channels_last_3d

    out = input.new_empty(shape_out)
    out = out.to(memory_format=memory_format)  # type: ignore[call-overload]
    return out

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

    use_channels_last = is_channels_last(input) or is_channels_last_3d(input) or weight_channels_last
    memory_format = torch.contiguous_format
    if use_channels_last:
        if input.dim() == 4:
            memory_format = torch.channels_last
        elif input.dim() == 5:
            memory_format = torch.channels_last_3d

    out = input.new_empty(shape_out)
    out = out.to(memory_format=memory_format)  # type: ignore[call-overload]
    return out

@register_meta("ipex_linear")
def meta_ipex_linear(
    input,
    weight,
    bias,
    W_prepack,
    out_features,
):
    return input.new_empty((*input.shape[:-1], out_features))

@register_meta("ipex_MKLSGEMM")
def meta_ipex_MKLSGEMM(
    input,
    weight,
    bias,
    W_prepack,
    out_features,
):
    return input.new_empty((*input.shape[:-1], out_features))

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
    input,
    rois,
    spatial_scale,
    pooled_height,
    pooled_width,
    sampling_ratio,
    aligned
):
    return input.new_empty((rois.shape[0], input.shape[1], pooled_height, pooled_width))