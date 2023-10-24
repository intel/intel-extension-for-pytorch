from collections import namedtuple
from enum import IntEnum
import torch
from torch.ao.quantization import (
    PlaceholderObserver,
    PerChannelMinMaxObserver,
    HistogramObserver,
    QConfig,
    QConfigMapping,
)
from ._smooth_quant import SmoothQuantActivationObserver, SmoothQuantWeightObserver


_default_weight_observer = PerChannelMinMaxObserver.with_args(
    dtype=torch.qint8, qscheme=torch.per_channel_symmetric
)

default_static_qconfig = QConfig(
    activation=HistogramObserver.with_args(reduce_range=False),
    weight=_default_weight_observer,
)
"""
Default qconfig configuration for static quantization.
"""

default_static_qconfig_mapping = QConfigMapping().set_global(default_static_qconfig)

default_dynamic_qconfig = QConfig(
    activation=PlaceholderObserver.with_args(dtype=torch.float, is_dynamic=True),
    weight=_default_weight_observer,
)
"""
Default qconfig configuration for dynamic quantization.
"""

default_dynamic_qconfig_mapping = QConfigMapping().set_global(default_dynamic_qconfig)


# Define QConfig for SmoothQuant by extending PyTorch's QConfig
QConfigSmoothQuant = namedtuple(
    "QConfigSmoothQuant", [*QConfig._fields, "share_weight_observers"]
)


def get_smooth_quant_qconfig_mapping(
    alpha=0.5,
    act_observer=None,
    act_ic_observer=None,
    wei_observer=None,
    wei_ic_observer=None,
    share_weight_observers=True,
):
    """
    Configuration with SmoothQuant for static quantization of large language models (LLM)
    For SmoothQuant, see https://arxiv.org/pdf/2211.10438.pdf

    Args:
        alpha: Hyper-parameter for SmoothQuant.
        act_observer: Observer for activation of ops other than nn.Linear.
            HistogramObserver by default. For nn.Linear with SmoothQuant
            enabled, q-param is calculated based on act_ic_observer's and
            wei_ic_observer's min/max. It is not affected by this argument.
        act_ic_observer: Per-input-channel Observer for activation.
            For nn.Linear with SmoothQuant enabled only.
            PerChannelMinMaxObserver by default.
        wei_observer: Observer for weight of all weighted ops.
            For nn.Linear with SmoothQuant enabled, it calculates q-params
            after applying scaling factors. PerChannelMinMaxObserver by
            default.
        wei_ic_observer: Per-input-channel Observer for weight.
            For nn.Linear with SmoothQuant enabled only.
            PerChannelMinMaxObserver by default.

    Returns:
        torch.ao.quantization.QConfig
    """
    qconfig = QConfigSmoothQuant(
        activation=SmoothQuantActivationObserver.with_args(
            reduce_range=False,
            alpha=alpha,
            act_observer=act_observer,
            act_ic_observer=act_ic_observer,
        ),
        weight=SmoothQuantWeightObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            alpha=alpha,
            wei_observer=wei_observer,
            wei_ic_observer=wei_ic_observer,
        ),
        share_weight_observers=share_weight_observers,
    )
    return QConfigMapping().set_global(qconfig)


# For weight-only quantization
class WoqLowpMode(IntEnum):
    NONE = 0
    FP16 = 1
    BF16 = 2
    INT8 = 3


class WoqActQuantMode(IntEnum):
    NONE = -1
    PER_TENSOR = 0
    PER_IC_BLOCK = 1  # IC = Input Channel
    PER_BATCH = 2
    PER_BATCH_IC_BLOCK = 3


QConfigWoq = namedtuple("QConfigWoq", [*QConfig._fields, "lowp_mode", "act_quant_mode"])


def get_weight_only_quant_qconfig_mapping(
    *,
    weight_dtype: torch.dtype = torch.qint8,
    lowp_mode: int = WoqLowpMode.NONE,
    act_quant_mode: int = WoqActQuantMode.PER_IC_BLOCK
):
    """
    Configuration for weight-only quantization (WOQ) for LLM.
    Arguments:
        weight_dtype:   Data type for weight, torch.qint8 or torch.quint4x2.
        lowp_mode:      specify the lowest precision data type for computation. Data types
                        that has even lower precision won't be used.
                        Not necessarily related to activation or weight dtype.
                        - NONE(0): Use the activation data type for computation.
                        - FP16(1): Use float16 (a.k.a. half) as the lowest precision for computation.
                        - BF16(2): Use bfloat16 as the lowest precision for computation.
                        - INT8(3): Use INT8 as the lowest precision for computation.
                                   Activation is quantized to int8 at runtime in this case.
                        Note that lowp_mode=INT8(3) is only available when weight_dtype=torch.quint4x2.
                        In other cases, it will fall back to lowp_mode=BF16(2).
        act_quant_mode: Quantization granularity of activation. It only works for lowp_mode=INT8.
                        It has no effect in other cases. The tensor is divided into groups, and
                        each group is quantized with its own quantization parameters.
                        Suppose the activation has shape batch_size by input_channel (IC).
                        - PER_TENSOR(0): Use the same quantization parameters for the entire tensor.
                        - PER_IC_BLOCK(1): Tensor is divided along IC with group size = IC_BLOCK.
                        - PER_BATCH(2): Tensor is divided along batch_size with group size = 1.
                        - PER_BATCH_IC_BLOCK(3): Tenosr is divided into blocks of 1 x IC_BLOCK.
                        Note that IC_BLOCK is determined by IC automatically, not configurable.
    """
    dtype_to_qscheme = {
        torch.qint8: torch.per_channel_affine,
        torch.quint8: torch.per_channel_affine,
        # It is required to use per_channel_affine_float_qparams for quint4x2 by PyTorch
        torch.quint4x2: torch.per_channel_affine_float_qparams,
    }
    weight_qscheme = dtype_to_qscheme[weight_dtype]
    _weight_only_quant_qconfig = QConfigWoq(
        activation=PlaceholderObserver.with_args(dtype=torch.float, is_dynamic=False),
        weight=PerChannelMinMaxObserver.with_args(
            dtype=weight_dtype, qscheme=weight_qscheme
        ),
        lowp_mode=lowp_mode,
        act_quant_mode=act_quant_mode,
    )
    weight_only_quant_qconfig_mapping = QConfigMapping().set_global(
        _weight_only_quant_qconfig
    )
    return weight_only_quant_qconfig_mapping
