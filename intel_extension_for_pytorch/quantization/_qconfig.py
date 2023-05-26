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


def get_smooth_quant_qconfig_mapping(
    alpha=0.5,
    act_observer=None,
    act_ic_observer=None,
    wei_observer=None,
    wei_ic_observer=None,
):
    """
    Configuration with SmoothQuant for static quantization of large language models (LLM)
    For SmoothQuant, see https://arxiv.org/pdf/2211.10438.pdf
    Arguments:
        alpha:              Hyper-parameter for SmoothQuant.
        act_observer:       Observer for activation of ops other than nn.Linear. HistogramObserver by default.
                            For nn.Linear with SmoothQuant enabled, q-param is calculated based on act_ic_observer's
                            and wei_ic_observer's min/max. It is not affected by this argument.
        act_ic_observer:    Per-input-channel Observer for activation. For nn.Linear with SmoothQuant enabled only.
                            PerChannelMinMaxObserver by default.
        wei_observer:       Observer for weight of all weighted ops. For nn.Linear with SmoothQuant enabled, it
                            calculates q-params after applying scaling factors. PerChannelMinMaxObserver by default.
        wei_ic_observer:    Per-input-channel Observer for weight. For nn.Linear with SmoothQuant enabled only.
                            PerChannelMinMaxObserver by default.
    """
    qconfig = QConfig(
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
    )
    return QConfigMapping().set_global(qconfig)


# For weight-only quantization
class WoqLowpMode(IntEnum):
    NONE = 0
    FP16 = 1
    BF16 = 2

QConfigWoq = namedtuple('QConfigWoq', [*QConfig._fields, 'lowp_mode'])
def get_weight_only_quant_qconfig_mapping(
        *,
        weight_dtype: torch.dtype = torch.qint8,
        lowp_mode: int = WoqLowpMode.NONE):
    dtype_to_qscheme = {
        torch.qint8: torch.per_channel_affine,
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
    )
    weight_only_quant_qconfig_mapping = QConfigMapping().set_global(
        _weight_only_quant_qconfig
    )
    return weight_only_quant_qconfig_mapping
