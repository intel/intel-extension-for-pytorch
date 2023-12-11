# Custom lowerings overriding those from PyTorch
import torch
import contextlib
import functools
from torch._inductor.lowering import ELEMENTWISE_TYPE_PROMOTION_KIND, make_fallback

lowering_overrides = {}


def _register_lowering(
    aten_fn,
    decomp_fn,
    broadcast=False,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    convert_input_to_bool=False,
):
    if not isinstance(aten_fn, (list, tuple)):
        aten_fn = [aten_fn]
    else:
        aten_fn = list(aten_fn)
    for fn in aten_fn:
        lowering_overrides.update(
            {fn: (decomp_fn, broadcast, type_promotion_kind, convert_input_to_bool)}
        )


def register_lowering(
    aten_fn,
    broadcast=False,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    convert_input_to_bool=False,
):
    return functools.partial(
        _register_lowering,
        aten_fn,
        broadcast=broadcast,
        type_promotion_kind=type_promotion_kind,
        convert_input_to_bool=convert_input_to_bool,
    )


make_fallback(torch.ops.torch_ipex.convolution_forward)
make_fallback(torch.ops.torch_ipex.convolution_backward)
make_fallback(torch.ops.torch_ipex.conv_transpose)
make_fallback(torch.ops.torch_ipex.conv_transpose_backward)
make_fallback(torch.ops.torch_ipex.ipex_linear)
make_fallback(torch.ops.torch_ipex.linear_backward)
make_fallback(torch.ops.torch_ipex.ipex_MKLSGEMM)
make_fallback(torch.ops.torch_ipex.ipex_linear_eltwise)
make_fallback(torch.ops.torch_ipex.linear_eltwise_backward)
make_fallback(torch.ops.torch_ipex.embedding_bag)
make_fallback(torch.ops.torch_ipex.ipex_lstm)
make_fallback(torch.ops.torch_ipex.ROIAlign_forward)
make_fallback(torch.ops.torch_ipex.ROIAlign_backward)
make_fallback(torch.ops.torch_ipex.batch_norm_forward)
make_fallback(torch.ops.torch_ipex.batch_norm_backward)
make_fallback(torch.ops.torch_ipex.cumsum)
make_fallback(torch.ops.torch_ipex.tpp_linear)
make_fallback(torch.ops.torch_ipex.tpp_linear_bias)
make_fallback(torch.ops.torch_ipex.tpp_linear_gelu)
make_fallback(torch.ops.torch_ipex.tpp_linear_add_add)
make_fallback(torch.ops.torch_ipex.tpp_linear_relu)
make_fallback(torch.ops.torch_ipex.tpp_linear_silu)
make_fallback(torch.ops.torch_ipex.tpp_linear_add)
make_fallback(torch.ops.torch_ipex.tpp_linear_mul)
make_fallback(torch.ops.torch_ipex.masked_multihead_self_attention)
make_fallback(torch.ops.torch_ipex.rotary_position_embedding)

make_fallback(torch.ops.torch_ipex.add_softmax_)
make_fallback(torch.ops.torch_ipex.bmm_add)


@contextlib.contextmanager
def patch_lowering():
    import copy
    from torch._inductor.lowering import lowerings
    from torch._inductor.lowering import register_lowering

    old_lowerings = lowerings
    lowerings = copy.copy(lowerings)
    for fn, (
        decomp_fn,
        broadcast,
        type_promotion_kind,
        convert_input_to_bool,
    ) in lowering_overrides.items():
        register_lowering(
            fn,
            broadcast=broadcast,
            type_promotion_kind=type_promotion_kind,
            convert_input_to_bool=convert_input_to_bool,
        )(decomp_fn)
    yield
    lowerings = old_lowerings
