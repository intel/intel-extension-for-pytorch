import copy
from typing import Tuple, Any
import warnings

import torch
from torch.ao.quantization import PlaceholderObserver
import torch.fx.experimental.optimization as optimization

import intel_extension_for_pytorch._C as core
from ._quantize_utils import auto_prepare, auto_convert, copy_prepared_model
from .. import nn

def prepare(
    model,
    configure,
    example_inputs,
    inplace=False):
    r"""
    Prepare an FP32 torch.nn.Module model to do calibration or to convert to quantized model.

    Args:
        model (torch.nn.Module): The FP32 model to be prepared.
        configure (torch.quantization.qconfig.QConfig): The observer settings about activation and weight.
        example_inputs (tuple or torch.Tensor): A tuple of example inputs that
            will be passed to the function while running to init quantization state.
        inplace: (bool): It will change the given model in-place if True. The default value is ``False``.

    Returns:
        torch.nn.Module
    """
    assert isinstance(model, torch.nn.Module), "Only support nn.Module prepare for quantization path"
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
    # replace dropout with identity to enable more fusion pattern.
    nn.utils._model_convert.replace_dropout_with_identity(prepare_model)
    # Special case for common case of passing a single Tensor
    if isinstance(example_inputs, (torch.Tensor, dict)):
        example_inputs = (example_inputs,)
    elif not isinstance(example_inputs, tuple):
        example_inputs = tuple(example_inputs)
    return auto_prepare(prepare_model, configure, example_inputs)

def convert(
    model,
    inplace=False):
    r"""
    Convert an FP32 prepared model to a model which will automatically insert fake quant
    before a quantizable module or operator.

    Args:
        model (torch.nn.Module): The FP32 model to be convert.
        inplace: (bool): It will change the given model in-place if True. The default value is ``False``.

    Returns:
        torch.torch.nn.Module
    """
    assert isinstance(model, torch.nn.Module), "Only support nn.Module convert for quantization path"
    assert hasattr(model, 'q_config'), "Please do prepare the model before doing convert"

    if inplace:
        convert_model = model
    else:
        try:
            convert_model = copy_prepared_model(model)
        except:
            assert False, "The model's copy is failed, please try set inplace to True to do the convert"

    # If the module's activation's qconfig is PlaceholderObserver,
    # we can say that the module want to run dynamic quantization path.
    if isinstance(convert_model.q_config.activation(), PlaceholderObserver):
        qconfig_spec = {
            torch.nn.Linear : convert_model.q_config,
            torch.nn.LSTM : convert_model.q_config,
            torch.nn.GRU : convert_model.q_config,
            torch.nn.LSTMCell : convert_model.q_config,
            torch.nn.RNNCell : convert_model.q_config,
            torch.nn.GRUCell : convert_model.q_config,
        }
        return torch.quantization.quantize_dynamic(convert_model, qconfig_spec=qconfig_spec, inplace=True)

    # Convert linear, conv, and Embedding's weight dtype when use autocast,
    # which will reduce the dtype conversion.
    # TODO: check whether can be removed or not?
    if torch.is_autocast_cpu_enabled() and core.get_autocast_dtype() == torch.bfloat16:
        convert_model = nn.utils._model_convert.convert_module_data_type(convert_model, torch.bfloat16)

    convert_model = auto_convert(convert_model)
    return convert_model
