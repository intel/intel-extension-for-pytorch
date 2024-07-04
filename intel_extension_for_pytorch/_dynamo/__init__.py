import torch
from torch._dynamo.backends.registry import register_backend
from typing import List
from functools import lru_cache


def _get_device_from_graph_module(graph_module: torch.fx.GraphModule):
    """
    This function gathers the device types from the fx graph nodes.
    """
    _example_value_list = [
        node.meta["example_value"]
        for node in graph_module.graph.nodes
        if "example_value" in node.meta
    ]
    if "xpu" in (
        value.device.type
        for value in _example_value_list
        if isinstance(value, torch.Tensor)
    ):
        return "xpu"
    else:
        return "cpu"


@lru_cache(None)
def register_fallback_aten_ops():
    from torch._inductor.lowering import make_fallback

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


@register_backend
def ipex(
    graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor], **kwargs
):
    """
    This function implements the interface for a unified compile backend 'ipex'.
    """
    # register fallback ops lazily to avoid trigger XPU runtime via import torch._inductor.lowering
    register_fallback_aten_ops()

    options = kwargs["options"] if "options" in kwargs else None
    if _get_device_from_graph_module(graph_module) == "cpu":
        from .._inductor.cpu.compiler import compile

        return compile(graph_module, example_inputs, options=options)
    else:
        from ..utils.utils import _is_syngraph_available
        from torch._inductor.compile_fx import compile_fx

        if _is_syngraph_available():
            # FIXME: For now the syngraph is not integrated into the IPEX,
            # so here raise runtime error for using syngraph compiler
            raise RuntimeError(
                "For now the syngraph is not integrated into the IPEX, \
                so here syngraph compiler is not available"
            )
        else:
            from .._inductor.xpu import register_xpu_fusion_to_inductor

            # register xpu fusion to inductor to avoid the circular import via torch._inductor.compile_fx
            register_xpu_fusion_to_inductor()

            return compile_fx(graph_module, example_inputs, config_patches=options)
