from typing import List
import torch

# Note: import order is significant here due to the defect of triton.compile
# in XPU backend. Here codecache is a temp WA.
from torch._inductor import codecache  # noqa
from torch._dynamo.device_interface import register_interface_for_device
from torch._dynamo.backends.registry import register_backend
from torch._inductor.compile_fx import compile_fx

from .device_interface import XPUInterface
from .register import _register_module_function_to_dynamo

if torch.xpu.is_available():
    # Register XPU device interfaces in PyTorch Dynamo.
    register_interface_for_device("xpu", XPUInterface)

    # Register XPU modules and functions to PyTorch Dynamo.
    """
    [Dynamo]
    In torch 2.1, dynamo maintains some containers to conditionally gather all allowed/disallowed
    torch modules or functions for building variables to track. Most of the torch modules and 
    functions should be wrappered as the TorchVariable by dynamo and the CALL_FUNCTION bytecode 
    from python frame will be translated to be a call function of TorchVariable to further 
    jit the recognized frame, for example, when we call `torch.cuda.Stream`, the `torch` 
    is wrappered as a TorchVariable natively as below.
        `torch` -  TorchVariable
    then interpreter tries to get the attr `cuda` from the `torch` from bytecode GET_ATTR, 
    it is also wrappered as TorchVariable because most of the torch.cuda functions are 
    registered internally when torch is imported, 
        `torch.cuda` -  TorchVariable
    then interpreter tries to get the attr `Stream` from the `torch.cuda`. Eventually the 
    CALL_FUNCTION(torch.cuda.Stream()) is executed and dynamo barf the FX Node onto the 
    graph(if support this function in dynamo). Thus, the out-of-tree registration for xpu is 
    needed to upload all builtin legal modules and functions to dynamo, for example,
        `torch.xpu` - aka - `intel_extension_for_pytorch.xpu`
    otherwise it will be wrappered as UserDefinedObjectVariable or PythonModuleVariable.
    Here registration is changed since torch 2.2, so it is only for torch<2.1.
    """
    # TODO: when torch 2.2 is rebased, here registration should be changed soon
    # TODO: when xpu is upstream, here registration is not needed
    _register_module_function_to_dynamo(torch.xpu)


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
        from .._inductor.cpu.compiler import compile

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
