import torch
import torch._custom_ops
import torch.library
import functools
from torch._meta_registrations import calc_conv_nd_return_shape


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


@register_meta("_convolution_pointwise", "default")
def meta_torch_ipex_convolution_default(
    input_tensor,
    weight,
    bias,
    padding,
    stride,
    dilation,
    groups,
    attr,
    scalars,
    algorithm,
):
    shape_out = calc_conv_nd_return_shape(
        input_tensor, weight, stride, padding, dilation, False, groups, []
    )
    out = input_tensor.new_empty(shape_out)
    out_memory_format = torch.channels_last
    out = out.to(memory_format=out_memory_format)  # type: ignore[call-overload]
    return out


@register_meta("_convolution_pointwise", "binary")
def meta_torch_ipex_convolution_binary(
    input_tensor,
    other,
    weight,
    bias,
    padding,
    stride,
    dilation,
    groups,
    binary_attr,
    alpha,
    unary_attr,
    unary_scalars,
    unary_algorithm,
):
    out = input_tensor.new_empty(other.size())
    out = out.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
    return out


@register_meta("_convolution_pointwise_", "binary")
def meta_torch_ipex_convolution_binary_inplace(
    input_tensor,
    other,
    weight,
    bias,
    padding,
    stride,
    dilation,
    groups,
    binary_attr,
    alpha,
    unary_attr,
    unary_scalars,
    unary_algorithm,
):
    return other
