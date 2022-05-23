from typing import Dict, Callable, Any, Optional

import torch
import torch.nn as nn

from torch.ao.quantization import swap_module
import torch.nn.quantized.dynamic as nnqd


# Default map for swapping dynamic modules
DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS : Dict[Callable, Any] = {
    nn.Linear: nnqd.Linear,
    nn.LSTM: nnqd.LSTM,
    # TODO: support more RNN module
    #nn.GRUCell: nnqd.GRUCell,
    #nn.GRU: nnqd.GRU,
    #nn.LSTMCell: nnqd.LSTMCell,
    #nn.RNNCell: nnqd.RNNCell,
}

def _get_qconfig_dtypes(qconfig):
    r"""
    Returns the qconfig tuple for qconfig:
    (activation_dtype, weight_dtype, activation_compute_dtype)
    """
    assert qconfig is not None
    activation = qconfig.activation()
    weight = qconfig.weight()
    compute_dtype = activation.compute_dtype if hasattr(activation, 'compute_dtype') else None
    return (activation.dtype, weight.dtype, compute_dtype)

def _op_is_int8_dynamically_quantized(qconfig) -> bool:
    r""" 
    Given a qconfig, returns True if this op is using int8 dynamic
    quantization
    """
    activation_dtype, weight_dtype, activation_compute_dtype = \
        _get_qconfig_dtypes(qconfig)
    return (
        activation_dtype is torch.float and
        # for now, the lines below assume fbgemm or qnnpack
        weight_dtype is torch.qint8 and
        activation_compute_dtype is torch.quint8
    )


def swap_child_modules(
    module: torch.nn.Module,
    dynamic_mappings: Dict[Callable, Any] = DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS,
    parent_fqn: Optional[str] = None,
) -> None:
    """
    For each direct child of `module`, swaps it using `dyanamic_mappings`
    if the qconfig for that child is using int8 dynamic quantization,
    and the module type is in the mapping.
    Recursively calls itself on each child.
    """

    if hasattr(module, '_auto_quant_state'):
        qstate = module._auto_quant_state
        for _, qopinfo in qstate.idx_to_seen_q_op_infos.items():
            qconfig = qopinfo.qconfig
            if not qconfig:
                continue
            fqn = qopinfo.fqn
            if not fqn:
                continue
            op_int8_dynamically_quantized = _op_is_int8_dynamically_quantized(qconfig)

            if op_int8_dynamically_quantized:
                mod = module._modules[fqn]
                if not type(mod) in dynamic_mappings:
                    continue
                mod.qconfig = qconfig
                module._modules[fqn] = swap_module(mod, dynamic_mappings, {})

    for _, child in module.named_children():
        swap_child_modules(child)
