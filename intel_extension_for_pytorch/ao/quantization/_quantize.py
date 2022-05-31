import torch
import copy
from typing import Tuple, Any

import intel_extension_for_pytorch._C as core
import torch.fx.experimental.optimization as optimization
from ._quantize_utils import auto_prepare, auto_convert
import warnings
from ... import nn

def prepare(
    model,
    configure,
    example_inputs,
    inplace=True):
    r"""
    Prepare an FP32 torch.nn.Module model to do calibration or to convert to quantized model.
    Args:
        model (torch.nn.Module): The FP32 model to be prepared.
        configure (torch.quantization.qconfig.QConfig): The observer settings about activation and weight.
        example_inputs (tuple or torch.Tensor): A tuple of example inputs that
            will be passed to the function while running to init quantizaiton state. 
        inplace: (bool): It will do overide the original model.
    Returns:
        torch.nn.Module
    """

    try:
        prepare_model = optimization.fuse(model, inplace=inplace)
    except:  # noqa E722
        if inplace:
            prepare_model = model
        else:
            try:
                prepare_model = copy.deepcopy(model)
            except:
                assert False, "The model's copy is failed, please try set inplace to True to do the prepare"
        warnings.warn("Conv BatchNorm folding failed during the prepare process.")
    # Special case for common case of passing a single Tensor
    if isinstance(example_inputs, (torch.Tensor, dict)):
        example_inputs = (example_inputs,)
    elif not isinstance(example_inputs, tuple):
        example_inputs = tuple(example_inputs)
    return auto_prepare(prepare_model, configure, example_inputs)

def convert(model):
    r"""
    Convert an FP32 prepared model to a model which will automatically insert fake quant
    before a quantizable module or operator.
    Args:
        model (torch.nn.Module): The FP32 model to be convert.
    Returns:
        torch.torch.nn.Module
    """

    assert isinstance(model, torch.nn.Module), "Only support nn.Module convert for quantization path"
    # Vonvert linear and weight's dtype when use autocast, which will reduce the dtype conversion.
    # TODO: check whether can be removed or not?
    if torch.is_autocast_cpu_enabled() and core.get_autocast_dtype() == torch.bfloat16:
        model = nn.utils._model_convert.convert_module_data_type(model, torch.bfloat16)
    convert_model = auto_convert(model)
    return convert_model
