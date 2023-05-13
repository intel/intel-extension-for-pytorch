from typing import Dict, Callable, Any, Optional

import torch
import torch.nn as nn

from torch.ao.quantization import swap_module
import torch.nn.quantized.dynamic as nnqd
from torch.quantization.qconfig import QConfig

# Default map for swapping dynamic modules
DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS: Dict[Callable, Any] = {
    nn.Linear: nnqd.Linear,
    nn.LSTM: nnqd.LSTM,
    # TODO: support more RNN module
    # nn.GRUCell: nnqd.GRUCell,
    # nn.GRU: nnqd.GRU,
    # nn.LSTMCell: nnqd.LSTMCell,
    # nn.RNNCell: nnqd.RNNCell,
}


def _get_qconfig_dtypes(qconfig):
    r"""
    Returns the qconfig tuple for qconfig:
    (activation_dtype, weight_dtype, activation_compute_dtype)
    """
    assert qconfig is not None
    activation = qconfig.activation()
    weight = qconfig.weight()
    compute_dtype = (
        activation.compute_dtype if hasattr(activation, "compute_dtype") else None
    )
    return (activation.dtype, weight.dtype, compute_dtype)


def _op_is_int8_dynamically_quantized(qconfig) -> bool:
    r"""
    Given a qconfig, returns True if this op is using int8 dynamic
    quantization
    """
    activation_dtype, weight_dtype, activation_compute_dtype = _get_qconfig_dtypes(
        qconfig
    )
    return (
        activation_dtype is torch.float
        and
        # for now, the lines below assume fbgemm or qnnpack
        weight_dtype is torch.qint8
        and activation_compute_dtype is torch.quint8
    )


def _swap_child_modules(
    module: torch.nn.Module,
    fqn_qconfig: Dict[str, QConfig],
    dynamic_mappings: Dict[Callable, Any] = DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS,
    parent_fqn: Optional[str] = None,
) -> None:
    """
    For each direct child of `module`, swaps it using `dyanamic_mappings`
    if the qconfig for that child is using int8 dynamic quantization,
    and the module type is in the mapping.
    Recursively calls itself on each child.
    """
    reassign = {}
    for local_fqn, mod in module.named_children():
        if parent_fqn is None:
            global_fqn = local_fqn
        else:
            global_fqn = f"{parent_fqn}.{local_fqn}"
        _swap_child_modules(mod, fqn_qconfig, dynamic_mappings, global_fqn)

        if global_fqn in fqn_qconfig:
            qconfig = fqn_qconfig[global_fqn]
            if not qconfig:
                continue
            mod.qconfig = qconfig
            op_int8_dynamically_quantized = _op_is_int8_dynamically_quantized(qconfig)
            if op_int8_dynamically_quantized:
                if not type(mod) in dynamic_mappings:
                    continue
                reassign[local_fqn] = swap_module(mod, dynamic_mappings, {})

    for key, value in reassign.items():
        module._modules[key] = value


def swap_child_modules(
    module: torch.nn.Module,
    dynamic_mappings: Dict[Callable, Any] = DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS,
) -> None:
    fqn_qconfig = {}
    for _, v in module._fqn_to_auto_quant_state_map.items():
        if len(v.idx_to_seen_q_op_infos) > 0:
            for _, op_info in v.idx_to_seen_q_op_infos.items():
                fqn_qconfig[op_info.fqn] = op_info.qconfig

    _swap_child_modules(module, fqn_qconfig, dynamic_mappings)
