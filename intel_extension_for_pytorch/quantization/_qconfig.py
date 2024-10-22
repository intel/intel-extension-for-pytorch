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
from ._smooth_quant import (
    SmoothQuantActivationObserver,
    SmoothQuantWeightObserver,
)
import warnings


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
            Example: ``torch.ao.quantization.MinMaxObserver``
        act_ic_observer: Per-input-channel Observer for activation.
            For nn.Linear with SmoothQuant enabled only.
            PerChannelMinMaxObserver by default.
            Example: ``torch.ao.quantization.PerChannelMinMaxObserver.with_args(ch_axis=1)``
        wei_observer: Observer for weight of all weighted ops.
            For nn.Linear with SmoothQuant enabled, it calculates q-params
            after applying scaling factors. PerChannelMinMaxObserver by
            default.
            Example: ``torch.ao.quantization.PerChannelMinMaxObserver.with_args(\
dtype=torch.qint8, qscheme=torch.per_channel_symmetric)``
        wei_ic_observer: Per-input-channel Observer for weight.
            For nn.Linear with SmoothQuant enabled only.
            PerChannelMinMaxObserver by default.
            Example: ``torch.ao.quantization.PerChannelMinMaxObserver.with_args(ch_axis=1)``

    Returns:
        torch.ao.quantization.QConfig
    """
    qconfig = QConfigSmoothQuant(
        activation=SmoothQuantActivationObserver.with_args(
            reduce_range=False,
            alpha=float(alpha),
            act_observer=act_observer,
            act_ic_observer=act_ic_observer,
        ),
        weight=SmoothQuantWeightObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            alpha=float(alpha),
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


WOQ_LOWP_MODE_TO_STR = {
    WoqLowpMode.NONE: "none",
    WoqLowpMode.FP16: "fp16",
    WoqLowpMode.BF16: "bf16",
    WoqLowpMode.INT8: "int8",
}


class WoqActQuantMode(IntEnum):
    NONE = -1
    PER_TENSOR = 0
    PER_IC_BLOCK = 1  # IC = Input Channel
    PER_BATCH = 2
    PER_BATCH_IC_BLOCK = 3
    PER_TENSOR_SYM = 4
    PER_IC_BLOCK_SYM = 5
    PER_BATCH_SYM = 6
    PER_BATCH_IC_BLOCK_SYM = 7


WOQ_ACT_QUANT_MODE_TO_STR = {
    WoqActQuantMode.NONE: "none",
    WoqActQuantMode.PER_TENSOR: "per_tensor",
    WoqActQuantMode.PER_IC_BLOCK: "per_ic_block",
    WoqActQuantMode.PER_BATCH: "per_batch",
    WoqActQuantMode.PER_BATCH_IC_BLOCK: "per_batch_ic_block",
    WoqActQuantMode.PER_TENSOR_SYM: "per_tensor_sym",
    WoqActQuantMode.PER_IC_BLOCK_SYM: "per_ic_block_sym",
    WoqActQuantMode.PER_BATCH_SYM: "per_batch_sym",
    WoqActQuantMode.PER_BATCH_IC_BLOCK_SYM: "per_batch_ic_block_sym",
}


# Start from 1 to align with kernel
class WoqWeightDtype(IntEnum):
    INT8 = 1
    INT4 = 2
    NF4 = 3


WOQ_DTYPE_TO_STR = {
    WoqWeightDtype.INT8: "int8",
    WoqWeightDtype.INT4: "int4",
    WoqWeightDtype.NF4: "nf4",
}


class WoqWeightQScheme(IntEnum):
    UNDEFINED = 0
    ASYMMETRIC = 1
    SYMMETRIC = 2


WOQ_QSCHEME_TO_STR = {
    WoqWeightQScheme.UNDEFINED: "undefined",
    WoqWeightQScheme.ASYMMETRIC: "asymmetric",
    WoqWeightQScheme.SYMMETRIC: "symmetric",
}


QConfigWoq = namedtuple(
    "QConfigWoq",
    [
        *QConfig._fields,
        "lowp_mode",
        "act_quant_mode",
        "weight_dtype",
        "group_size",
        "cache_weight_for_large_batch",
        "weight_qscheme",
    ],
)


def get_weight_only_quant_qconfig_mapping(
    *,
    weight_dtype: int = WoqWeightDtype.INT8,
    lowp_mode: int = WoqLowpMode.NONE,
    act_quant_mode: int = WoqActQuantMode.PER_BATCH_IC_BLOCK_SYM,
    group_size: int = -1,
    weight_qscheme: int = WoqWeightQScheme.UNDEFINED,
):
    """
    Configuration for weight-only quantization (WOQ) for LLM.
    Arguments:
        weight_dtype:   Data type for weight, WoqWeightDtype.INT8/INT4/NF4, etc.
        lowp_mode:      specify the lowest precision data type for computation. Data types
                        that has even lower precision won't be used.
                        Not necessarily related to activation or weight dtype.
                        - NONE(0): Use the activation data type for computation.
                        - FP16(1): Use float16 (a.k.a. half) as the lowest precision for computation.
                        - BF16(2): Use bfloat16 as the lowest precision for computation.
                        - INT8(3): Use INT8 as the lowest precision for computation.
                                   Activation is quantized to int8 at runtime in this case.
        act_quant_mode: Quantization granularity of activation. It only works for lowp_mode=INT8.
                        It has no effect in other cases. The tensor is divided into groups, and
                        each group is quantized with its own quantization parameters.
                        Suppose the activation has shape batch_size by input_channel (IC).
                        - PER_TENSOR(0): Use the same quantization parameters for the entire tensor.
                        - PER_IC_BLOCK(1): Tensor is divided along IC with group size = IC_BLOCK.
                        - PER_BATCH(2): Tensor is divided along batch_size with group size = 1.
                        - PER_BATCH_IC_BLOCK(3): Tenosr is divided into blocks of 1 x IC_BLOCK.
                        Note that IC_BLOCK is determined by group_size automatically.
        group_size:     Control quantization granularity along input channel (IC) dimension of weight.
                        Must be a positive power of 2 (i.e., 2^k, k > 0) or -1.
                        If group_size = -1:
                            If act_quant_mode = PER_TENSOR ro PER_BATCH:
                                No grouping along IC for both activation and weight
                            If act_quant_mode = PER_IC_BLOCK or PER_BATCH_IC_BLOCK:
                                No grouping along IC for weight. For activation,
                                IC_BLOCK is determined automatically by IC.
                        If group_size > 0:
                            act_quant_mode can be any. If act_quant_mode is PER_IC_BLOCK(_SYM)
                            or PER_BATCH_IC_BLOCK(_SYM), weight is grouped along IC by group_size.
                            The IC_BLOCK for activation is determined by group_size automatically.
                            Each group has its own quantization parameters.
        weight_qscheme: Specify how to quantize weight, asymmetrically or symmetrically. Generally,
                        asymmetric quantization has better accuracy than symmetric quantization at
                        the cost of performance. Symmetric quantization is faster but may have worse
                        accuracy. Default is undefined and determined by weight dtype: asymmetric in
                        most cases and symmetric if
                            (1) weight_dtype is NF4, or
                            (2) weight_dtype is INT8 and lowp_mode is INT8.
                        One must use WoqWeightQScheme.SYMMETRIC in the above two cases.
    """
    assert group_size == -1 or (
        group_size > 0 and (group_size & (group_size - 1)) == 0
    ), "Group size must be -1 or a positive power of 2, but got {}".format(group_size)
    if weight_dtype in (torch.qint8, torch.quint4x2):
        weight_dtype = (
            WoqWeightDtype.INT8 if weight_dtype == torch.qint8 else WoqWeightDtype.INT4
        )
        warnings.warn(
            "torch.qint8 and torch.quint4x2 are deprecated for weight only quantization "
            "in IPEX. Please use data types defined by "
            "intel_extension_for_pytorch.quantization.WoqWeightDtype. "
            "For example, `WoqWeightDtype.INT8` and `WoqWeightDtype.INT4`."
        )
    valid_values = [item.value for item in WoqWeightDtype]
    assert (
        weight_dtype in valid_values
    ), f"Invalid weight data type for weight only quantization: {weight_dtype}"

    if lowp_mode != WoqLowpMode.INT8:
        act_quant_mode = WoqActQuantMode.NONE

    if (weight_dtype == WoqWeightDtype.NF4) or (
        weight_dtype == WoqWeightDtype.INT8 and lowp_mode == WoqLowpMode.INT8
    ):
        assert (
            weight_qscheme != WoqWeightQScheme.ASYMMETRIC
        ), "Asymmetric quantization of weight is not supported when "
        "(1) weight_dtype is NF4 or (2) weight_dtype is INT8 and lowp_mode is INT8."
    if weight_qscheme == WoqWeightQScheme.UNDEFINED:
        weight_qscheme = (
            WoqWeightQScheme.SYMMETRIC
            if weight_dtype == WoqWeightDtype.NF4
            or (weight_dtype == WoqWeightDtype.INT8 and lowp_mode == WoqLowpMode.INT8)
            else WoqWeightQScheme.ASYMMETRIC
        )

    _weight_only_quant_qconfig = QConfigWoq(
        activation=PlaceholderObserver.with_args(dtype=torch.float, is_dynamic=False),
        weight=PerChannelMinMaxObserver(),
        lowp_mode=lowp_mode,
        act_quant_mode=act_quant_mode,
        weight_dtype=weight_dtype,
        group_size=group_size,
        cache_weight_for_large_batch=False,
        weight_qscheme=weight_qscheme,
    )
    weight_only_quant_qconfig_mapping = QConfigMapping().set_global(
        _weight_only_quant_qconfig
    )
    return weight_only_quant_qconfig_mapping
