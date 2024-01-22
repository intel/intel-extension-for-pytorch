from typing import List
import torch
from torch._dynamo.backends.registry import register_backend
from torch._inductor.compile_fx import compile_fx


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


@register_backend
def ipex(
    graph_module: torch.fx.GraphModule, example_inputs: List[torch.Tensor], **kwargs
):
    """
    This function implements the interface for a unified compile backend 'ipex'.
    """
    options = kwargs["options"] if "options" in kwargs else None
    if _get_device_from_graph_module(graph_module) == "cpu":
        from .._inductor.compiler import compile

        return compile(graph_module, example_inputs, options=options)
    else:
        from ..utils.utils import _is_syngraph_available

        if _is_syngraph_available():
            # FIXME: For now the syngraph is not integrated into the IPEX,
            # so here raise runtime error for using syngraph compiler
            raise RuntimeError(
                "For now the syngraph is not integrated into the IPEX, \
                so here syngraph compiler is not available"
            )
        else:
            return compile_fx(graph_module, example_inputs, config_patches=options)
