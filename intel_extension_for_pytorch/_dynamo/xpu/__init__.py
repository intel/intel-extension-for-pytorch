import torch

from .register import _register_module_function_to_dynamo
from ...utils.utils import has_xpu

if torch.xpu._is_compiled() and has_xpu():

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
