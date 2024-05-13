import copy
import functools
from ..utils._logger import logger, WarningType

import torch
from torch.ao.quantization import PlaceholderObserver, QConfig, QConfigMapping
from torch.ao.quantization.quantization_mappings import (
    get_default_dynamic_quant_module_mappings,
)
import torch.fx.experimental.optimization as optimization
from torch.ao.nn.quantized.modules.utils import _quantize_weight
import torch.ao.nn.quantized.dynamic as nnqd
from intel_extension_for_pytorch.cpu.utils.linear_bn_folding import linear_bn_fuse
from intel_extension_for_pytorch.nn.utils._weight_prepack import (
    may_import_deepspeed_modules,
    _all_reduce_and_bias_add,
    _pre_ipex_gemm,
)
from ._quantize_utils import auto_prepare, auto_convert, copy_prepared_model
from .. import nn
from typing import Dict


def prepare(
    model,
    configure,
    example_inputs=None,
    inplace=False,
    bn_folding=True,
    example_kwarg_inputs=None,
):
    r"""
    Prepare an FP32 torch.nn.Module model to do calibration or to convert to quantized model.

    Args:
        model (torch.nn.Module): The FP32 model to be prepared.
        configure (torch.quantization.qconfig.QConfig): The observer settings about activation and weight.
        example_inputs (tuple or torch.Tensor): A tuple of example inputs that
            will be passed to the function while running to init quantization state. Only one of this
            argument or ``example_kwarg_inputs`` should be specified.
        inplace: (bool): It will change the given model in-place if True. The default value is ``False``.
            Note that if ``bn_folding`` is ``True``, the returned model is a different object from the
            original model even if ``inplace=True``. So, with the following code
            >>> prepared_model = prepare(original_model, ..., inplace=True)
            please use ``prepared_model`` for later operations to avoid unexpected behaviors.
        bn_folding: (bool): whether to perform ``conv_bn`` and ``linear_bn`` folding.
            The default value is ``True``.
        example_kwarg_inputs (dict):  A dict of example inputs that will be passed to the function while
            running to init quantization state. Only one of this argument or ``example_inputs`` should be
            specified.

    Returns:
        torch.nn.Module
    """
    assert isinstance(
        model, torch.nn.Module
    ), "Only support nn.Module prepare for quantization path"
    assert isinstance(configure, QConfigMapping) or isinstance(
        configure, QConfig
    ), f"IPEX quantization: prepare configure should be an instance of QConfigMapping or QConfig, but got {type(configure)}"
    if isinstance(configure, QConfig):
        logger.warning(
            "\nIPEX quantization: QConfig are deprecated. Please use QConfigMapping instead.\nUsage:"
            + "\n    qconfig_mapping = ipex.quantization.default_static_qconfig_mapping # for static quantization"
            + "\n    qconfig_mapping = ipex.quantization.default_dynamic_qconfig_mapping # for dynamic quantization"
            + "\n    prepared_model = ipex.quantization.prepare(model_fp32, qconfig_mapping, ...)",
            _type=WarningType.AmbiguousArgument,
        )
    if isinstance(configure, QConfigMapping):
        configure = configure.global_qconfig
    if not isinstance(configure.activation(), PlaceholderObserver):
        assert example_inputs is not None or example_kwarg_inputs is not None, (
            "IPEX quantization.prepare: example_inputs and example_kwarg_inputs cannot be none at same time "
            "for static quantization."
        )
    # auto model channels_last memory format conversion
    from ..frontend import (
        auto_channels_last,
        _convert_convNd_deconvNd_weight_memory_format,
    )

    if auto_channels_last:
        _convert_convNd_deconvNd_weight_memory_format(model)
    if inplace:
        prepare_model = model
    else:
        try:
            prepare_model = copy.deepcopy(model)
        except BaseException:
            AssertionError(
                False
            ), "The model's copy is failed, please try set inplace to True to do the prepare"
    if bn_folding:
        try:
            prepare_model = optimization.fuse(prepare_model, inplace=inplace)
            prepare_model = linear_bn_fuse(prepare_model, inplace=inplace)
        except BaseException:
            logger.warning(
                "BatchNorm folding failed during the prepare process.",
                _type=WarningType.NotSupported,
            )

    # replace dropout with identity to enable more fusion pattern.
    nn.utils._model_convert.replace_dropout_with_identity(prepare_model)
    assert (
        example_inputs is None or example_kwarg_inputs is None
    ), "IPEX quantization.prepare: example_inputs and example_kwarg_inputs cannot be set at same time."
    # Special case for common case of passing a single Tensor
    if isinstance(example_inputs, (torch.Tensor, dict)):
        example_inputs = (example_inputs,)
    elif not isinstance(example_inputs, tuple) and example_inputs is not None:
        example_inputs = tuple(example_inputs)
    if example_kwarg_inputs is not None:
        assert isinstance(
            example_kwarg_inputs, Dict
        ), "IPEX quantization.prepare: example_kwarg_inputs must be type of Dict."
    return auto_prepare(prepare_model, configure, example_inputs, example_kwarg_inputs)


def _may_insert_deepspeed_modules(
    torch_modules,
    q_linear_layer_module,
    q_linear_all_reduce_module,
    q_lm_head_linear_all_reduce_module,
):
    deepspeed_modules = may_import_deepspeed_modules()
    if deepspeed_modules is not None:
        LinearAllreduce, LinearLayer = deepspeed_modules[:2]
        deepspeed_modules_dict = {
            LinearLayer: q_linear_layer_module,
            LinearAllreduce: q_linear_all_reduce_module,
        }
        if len(deepspeed_modules) > 2:
            LmHeadLinearAllreduce = deepspeed_modules[2]
            deepspeed_modules_dict.update(
                {
                    LmHeadLinearAllreduce: q_lm_head_linear_all_reduce_module,
                }
            )
        torch_modules.update(deepspeed_modules_dict)
    return torch_modules


@functools.lru_cache(None)
def IPEX_DYNAMIC_QUANTIZATION_MODULE_CPU():
    # TODO: have to override Linear here for GPT-J performance.
    # AutoTP model does not support deepcopy, thus need to do inplace convert.
    # The str(module) is called in GPT-J, which invokes the __repr__ function:
    #
    #   torch/ao/nn/quantized/modules/linear.py:__repr__()
    # -> return _hide_packed_params_repr(self, LinearPackedParams)
    #   torch/ao/nn/quantized/modules/utils.py:_hide_packed_params_repr()
    # -> extra_repr = self.extra_repr()
    #   torch/ao/nn/quantized/dynamic/modules/linear.py:extra_repr()
    # -> extra_repr_str += ', qscheme={}'.format(self.weight().qscheme())
    #   torch/ao/nn/quantized/modules/linear.py:weight()
    # -> return self._weight_bias()[0]
    #   torch/ao/nn/quantized/modules/linear.py:_weight_bias()
    # -> return self._packed_params._weight_bias()
    # > torch/ao/nn/quantized/modules/linear.py:_weight_bias()
    # -> return torch.ops.quantized.linear_unpack(self._packed_params)
    #
    # The quantized::linear_unpack function is quite slow.
    # We will override the __repr__ function of nnqd.Linear in DynamicQuantizedLinearLayer.
    torch_modules = {
        torch.nn.Linear: DynamicQuantizedLinearLayer,
    }

    torch_modules = _may_insert_deepspeed_modules(
        torch_modules,
        DynamicQuantizedLinearLayer,
        DynamicQuantizedLinearAllreduce,
        DynamicQuantizedLmHeadLinearAllreduce,
    )
    return torch_modules


@functools.lru_cache(None)
def IPEX_WEIGHT_ONLY_QUANTIZATION_MODULE_CPU():
    torch_modules = {}
    torch_modules = _may_insert_deepspeed_modules(
        torch_modules,
        nn.modules.weight_only_quantization.WeightOnlyQuantizedLinear,
        nn.modules.weight_only_quantization.IpexWoqLinearAllreduce,
        nn.modules.weight_only_quantization.IpexWoqLmHeadLinearAllreduce,
    )
    return torch_modules


class _IPEXDynamicQuantizedLinear(nnqd.Linear):
    @classmethod
    def from_float(cls, mod):
        assert (
            type(mod) in cls._float_module()
        ), "DynamicQuantizedLinearLayer.from_float only works for one of" + str(
            [float_mod.__name__ for float_mod in cls._float_module()]
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"

        if mod.qconfig is not None and mod.qconfig.weight is not None:
            weight_observer = mod.qconfig.weight()
        else:
            from torch.ao.quantization.qconfig import default_dynamic_qconfig

            weight_observer = default_dynamic_qconfig.weight()

        dtype = weight_observer.dtype
        assert dtype in [torch.qint8], (
            "The only supported dtypes for "
            "DynamicQuantizedLinearLayer is qint8 got: {}".format(dtype)
        )
        weight_observer(mod.weight)
        qweight = _quantize_weight(mod.weight.float(), weight_observer)

        qlinear = cls._init_cls(mod, dtype, qweight)
        return qlinear


class DynamicQuantizedLinearLayer(_IPEXDynamicQuantizedLinear):
    @classmethod
    def _init_cls(cls, mod, dtype, qweight):
        qlinear = cls(mod.weight.size()[1], mod.weight.size()[0], dtype=dtype)
        qlinear.set_weight_bias(qweight, mod.bias)
        return qlinear

    @classmethod
    @functools.lru_cache(None)
    def _float_module(cls):
        _FLOAT_MODULE = [torch.nn.Linear]
        deepspeed_modules = may_import_deepspeed_modules()
        if deepspeed_modules is not None:
            LinearLayer = deepspeed_modules[1]
            _FLOAT_MODULE.extend([LinearLayer])
        return _FLOAT_MODULE

    def __repr__(self):
        return "DynamicQuantizedLinearLayer()"


class DynamicQuantizedLinearAllreduce(_IPEXDynamicQuantizedLinear):
    @classmethod
    def _init_cls(cls, mod, dtype, qweight):
        qlinear = cls(
            mod.weight.size()[1],
            mod.weight.size()[0],
            mod.mp_group,
            mod.bias,
            dtype=dtype,
        )
        # For bias handling, please refer to the comment in __init__ of _IPEXLinearAllreduce
        qlinear.set_weight_bias(qweight, None)
        return qlinear

    @classmethod
    @functools.lru_cache(None)
    def _float_module(cls):
        deepspeed_modules = may_import_deepspeed_modules()
        assert (
            deepspeed_modules is not None
        ), "DynamicQuantizedLinearAllreduce requires deepspeed to be installed"
        LinearAllreduce = deepspeed_modules[0]
        _FLOAT_MODULE = [LinearAllreduce]
        return _FLOAT_MODULE

    def __init__(
        self,
        in_features,
        out_features,
        mp_group,
        bias_value,
        bias_=True,
        dtype=torch.qint8,
    ):
        # Save the original bias here
        # For bias handling, please refer to the comment in __init__ of _IPEXLinearAllreduce
        super().__init__(in_features, out_features, bias_, dtype=dtype)
        self.mp_group = mp_group
        self.original_bias = bias_value

    def pre_ipex_gemm(self, input):
        return input

    def forward(self, x):
        x = self.pre_ipex_gemm(x)

        if self._packed_params.dtype == torch.qint8:
            if self.version is None or self.version < 4:
                Y = torch.ops.quantized.linear_dynamic(
                    x, self._packed_params._packed_params
                )
            else:
                Y = torch.ops.quantized.linear_dynamic(
                    x, self._packed_params._packed_params, reduce_range=True
                )
        elif self._packed_params.dtype == torch.float16:
            Y = torch.ops.quantized.linear_dynamic_fp16(
                x, self._packed_params._packed_params
            )
        else:
            raise RuntimeError("Unsupported dtype on dynamic quantized linear!")
        output = Y.to(x.dtype)

        return _all_reduce_and_bias_add(self.mp_group, self.original_bias, output)

    def __repr__(self):
        return "DynamicQuantizedLinearAllreduce()"


class DynamicQuantizedLmHeadLinearAllreduce(DynamicQuantizedLinearAllreduce):
    @classmethod
    @functools.lru_cache(None)
    def _float_module(cls):
        deepspeed_modules = may_import_deepspeed_modules()
        assert (
            len(deepspeed_modules) > 2
        ), "DynamicQuantizedLmHeadLinearAllreduce requires deepspeed to be installed and LmHeadLinearAllreduce in deepspeed"
        LmHeadLinearAllreduce = deepspeed_modules[2]
        _FLOAT_MODULE = [LmHeadLinearAllreduce]
        return _FLOAT_MODULE

    def __init__(
        self,
        in_features,
        out_features,
        mp_group,
        rank,
        world_size,
        bias_value,
        bias_=True,
        dtype=torch.qint8,
    ):
        # Save the original bias here
        # For bias handling, please refer to the comment in __init__ of _IPEXLinearAllreduce
        super().__init__(in_features, out_features, mp_group, bias_value, bias_, dtype)
        self.rank = rank
        self.world_size = world_size

    @classmethod
    def _init_cls(cls, mod, dtype, qweight):
        qlinear = cls(
            mod.weight.size()[1],
            mod.weight.size()[0],
            mod.mp_group,
            mod.rank,
            mod.world_size,
            mod.bias,
            dtype=dtype,
        )
        # For bias handling, please refer to the comment in __init__ of _IPEXLinearAllreduce
        qlinear.set_weight_bias(qweight, None)
        return qlinear

    def pre_ipex_gemm(self, input):
        return _pre_ipex_gemm(input, self.world_size, self.rank)

    def __repr__(self):
        return "DynamicQuantizedLmHeadLinearAllreduce()"


def may_quantize_deepspeed_modules(
    IPEX_QUANTIZATION_MODULE, q_config, module_mappings, qconfig_spec
):
    deepspeed_modules = may_import_deepspeed_modules()
    if deepspeed_modules is not None:
        LinearAllreduce, LinearLayer = deepspeed_modules[:2]
        module_mappings.update(IPEX_QUANTIZATION_MODULE)
        deepspeed_qconfig_spec = {
            LinearLayer: q_config,
            LinearAllreduce: q_config,
        }
        if len(deepspeed_modules) > 2:
            LmHeadLinearAllreduce = deepspeed_modules[2]
            deepspeed_qconfig_spec.update(
                {
                    LmHeadLinearAllreduce: q_config,
                }
            )

        qconfig_spec.update(deepspeed_qconfig_spec)
    return module_mappings, qconfig_spec


def convert(model, inplace=False):
    r"""
    Convert an FP32 prepared model to a model which will automatically insert fake quant
    before a quantizable module or operator.

    Args:
        model (torch.nn.Module): The FP32 model to be convert.
        inplace: (bool): It will change the given model in-place if True. The default value is ``False``.

    Returns:
        torch.nn.Module
    """
    assert isinstance(
        model, torch.nn.Module
    ), "Only support nn.Module convert for quantization path"
    assert hasattr(
        model, "q_config"
    ), "Please do prepare the model before doing convert"

    if inplace:
        convert_model = model
    else:
        try:
            convert_model = copy_prepared_model(model)
        except BaseException:
            AssertionError(
                False
            ), "The model's copy is failed, please try set inplace to True to do the convert"

    # For weight only quantization. Activation's observer is also PlaceholderObserver.
    if (
        isinstance(convert_model.q_config.activation(), PlaceholderObserver)
        and not convert_model.q_config.activation().is_dynamic
    ):
        qconfig_spec = {
            torch.nn.Linear: convert_model.q_config,
            torch.nn.LSTM: convert_model.q_config,
            torch.nn.GRU: convert_model.q_config,
            torch.nn.LSTMCell: convert_model.q_config,
            torch.nn.RNNCell: convert_model.q_config,
            torch.nn.GRUCell: convert_model.q_config,
        }
        module_mappings = get_default_dynamic_quant_module_mappings().copy()
        module_mappings[torch.nn.Linear] = (
            nn.modules.weight_only_quantization.WeightOnlyQuantizedLinear
        )

        module_mappings, qconfig_spec = may_quantize_deepspeed_modules(
            IPEX_WEIGHT_ONLY_QUANTIZATION_MODULE_CPU(),
            convert_model.q_config,
            module_mappings,
            qconfig_spec,
        )
        converted_model = torch.quantization.quantize_dynamic(
            convert_model,
            qconfig_spec=qconfig_spec,
            dtype=torch.qint8,
            mapping=module_mappings,
            inplace=inplace,
        )
        return converted_model

    # If the module's activation's qconfig is PlaceholderObserver,
    # we can say that the module want to run dynamic quantization path.
    if isinstance(convert_model.q_config.activation(), PlaceholderObserver):
        module_mappings = get_default_dynamic_quant_module_mappings()
        qconfig_spec = {
            torch.nn.Linear: convert_model.q_config,
            torch.nn.LSTM: convert_model.q_config,
            torch.nn.GRU: convert_model.q_config,
            torch.nn.LSTMCell: convert_model.q_config,
            torch.nn.RNNCell: convert_model.q_config,
            torch.nn.GRUCell: convert_model.q_config,
        }

        module_mappings, qconfig_spec = may_quantize_deepspeed_modules(
            IPEX_DYNAMIC_QUANTIZATION_MODULE_CPU(),
            convert_model.q_config,
            module_mappings,
            qconfig_spec,
        )

        return torch.quantization.quantize_dynamic(
            convert_model,
            qconfig_spec=qconfig_spec,
            mapping=module_mappings,
            inplace=True,
        )

    # Convert linear, conv, and Embedding's weight dtype when use autocast,
    # which will reduce the dtype conversion.
    # TODO: check whether can be removed or not?
    if (
        torch.torch.is_autocast_enabled("cpu")
        and torch.get_autocast_dtype("cpu") == torch.bfloat16
    ):
        convert_model = nn.utils._model_convert.convert_model_data_type(
            convert_model, torch.bfloat16
        )[1]

    convert_model = auto_convert(convert_model)
    return convert_model
