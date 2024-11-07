import torch
from torch import nn
from typing import Optional
from intel_extension_for_pytorch.nn.utils._weight_prepack import (
    may_import_deepspeed_modules,
    _all_reduce_and_bias_add,
    _pre_ipex_gemm,
)
from intel_extension_for_pytorch.quantization import (
    QConfigWoq,
    quantize_per_channel,
    quantize_per_block,
    WoqWeightDtype,
    WoqWeightQScheme,
)
from intel_extension_for_pytorch.nn.utils._model_convert import (
    prepack_awq_weight,
    _convert_optimum_format_to_desired,
)

from intel_extension_for_pytorch.llm.quantization.utils import QuantMethod, QuantDtype
from intel_extension_for_pytorch.quantization._qconfig import (
    WOQ_LOWP_MODE_TO_STR,
    WOQ_ACT_QUANT_MODE_TO_STR,
    WOQ_DTYPE_TO_STR,
    WOQ_QSCHEME_TO_STR,
)


class WeightOnlyQuantizedLinear(nn.Module):
    r"""
    A weight-only quantized (WOQ) linear module with floating point tensor as inputs and outputs.
    Weight is dequantized at runtime for computation.
    """

    def __init__(
        self, in_features, out_features, bias_=True, dtype=WoqWeightDtype.INT8
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias_
        self.dtype = dtype
        # This weight attribute is for queries of dtype, shape, etc.
        # It is a reference of the packed weight in self._op_context
        # The shape is not necessarily = [out_features, in_features] due to packing
        # Its dtype is torch.int8 for INT8 and torch.uint8 for INT4
        self.weight = None
        self._op_context = None
        self._lowp_mode = 0
        self._act_quant_mode = 0
        self._group_size = -1
        self._cache_weight_for_large_batch = False
        self._weight_qscheme = WoqWeightQScheme.ASYMMETRIC

    def pre_ipex_gemm(self, input):
        return input

    def post_ipex_gemm(self, output):
        return output

    def forward(self, x):
        x = self.pre_ipex_gemm(x)

        Y = torch.ops.torch_ipex.ipex_woq_linear(x, self._op_context.get_data_handle())

        return self.post_ipex_gemm(Y)

    def _get_name(self):
        return "WeightOnlyQuantizedLinear"

    def extra_repr(self):
        extra_repr_str = "in_features={}, out_features={}, dtype={}".format(
            self.in_features, self.out_features, WOQ_DTYPE_TO_STR[self.dtype]
        )
        extra_repr_str += ", bias={}".format(self.bias)
        extra_repr_str += ", lowp_mode={}".format(WOQ_LOWP_MODE_TO_STR[self._lowp_mode])
        extra_repr_str += ", act_quant_mode={}".format(
            WOQ_ACT_QUANT_MODE_TO_STR[self._act_quant_mode]
        )
        extra_repr_str += ", group_size={}".format(self._group_size)
        extra_repr_str += ", cache_weight_for_large_batch={}".format(
            self._cache_weight_for_large_batch
        )
        extra_repr_str += ", weight_qscheme={}".format(
            WOQ_QSCHEME_TO_STR[self._weight_qscheme]
        )
        return extra_repr_str

    @classmethod
    def from_float(cls, mod, scales=None, zero_points=None):
        r"""Create a weight-only quantized module from a float module or qparams_dict

        Args:
            mod (Module): an instance of nn.Linear or its subclasses.
            scales: the scales Tensor for quantizing weight. If it is None,
                scales are found by min/max of the weight.
            zero_points: the zero points Tensor for quantizing weight. If it is None,
                zero points are found by min/max of the weight.
        """
        float_modules = [torch.nn.Linear]
        deepspeed_modules = may_import_deepspeed_modules()
        if deepspeed_modules is not None:
            float_modules.extend(deepspeed_modules)
        if any(issubclass(type(mod), float_module) for float_module in float_modules):
            float_modules.extend([type(mod)])

        assert type(mod) in float_modules, (
            "WeightOnlyQuantizedLinear.from_float only works for one of"
            + str([float_mod.__name__ for float_mod in float_modules])
            + f" or their subclasses, but found {type(mod)}"
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        qconfig = mod.qconfig
        if qconfig is None or not isinstance(qconfig, QConfigWoq):
            return mod

        lowp_mode = qconfig.lowp_mode
        act_quant_mode = qconfig.act_quant_mode
        dtype = qconfig.weight_dtype
        group_size = qconfig.group_size
        # if dtype = int8, lowp-mode = int8, we want zero points to be 0
        # otherwise, it may overflow when we subtract zero points from int8 weight.
        sym_quant = qconfig.weight_qscheme == WoqWeightQScheme.SYMMETRIC
        if dtype == WoqWeightDtype.NF4 or (
            dtype == WoqWeightDtype.INT8 and lowp_mode == 3
        ):
            assert (
                sym_quant is True
            ), "WOQ NF4 and INT8 with lowp-mode 3 must use symmetric quantization"

        if group_size == -1:
            qweight, scales, zero_points = quantize_per_channel(
                mod.weight, dtype, scales, zero_points, sym_quant
            )
        else:
            qweight, scales, zero_points = quantize_per_block(
                mod.weight, dtype, group_size, scales, zero_points, sym_quant
            )
        if not hasattr(mod, "in_features"):
            mod.in_features = mod.weight.size()[1]
        if not hasattr(mod, "out_features"):
            mod.out_features = mod.weight.size()[0]
        cache_weight_for_large_batch = (
            qconfig.cache_weight_for_large_batch and lowp_mode in (2, 3)
        )

        qlinear = cls._init_cls(
            mod,
            dtype,
            qweight,
            scales,
            zero_points,
            None,  # g_idx
            group_size,
            lowp_mode,
            act_quant_mode,
            cache_weight_for_large_batch,
        )
        del qweight
        mod.weight = torch.nn.Parameter()
        return qlinear

    @classmethod
    def from_float_and_int4_weight(
        cls, mod, qweight, scales, zero_points, bias=None, group_size=-1, g_idx=None
    ):
        r"""Create a weight-only quantized module from a float module and int4 weight

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
                          utilities or provided by the user
            qweight (Tensor): tensor in int32 dtype and contains actually int4 data
            bias (Tensor or None): bias for linear
            scales (Tensor): scales for qweight
            zero_points (Tensor): zero points for qweight
            group_size: Group size for weight quantization
            g_idx: Indices of groups for each input channel of weight. Generated by
                GPTQ with act-order.
        """
        float_modules = [torch.nn.Linear]
        deepspeed_modules = may_import_deepspeed_modules()
        if deepspeed_modules is not None:
            float_modules.extend(deepspeed_modules)
        if any(issubclass(type(mod), float_module) for float_module in float_modules):
            float_modules.extend([type(mod)])

        assert type(mod) in float_modules, (
            "WeightOnlyQuantizedLinear.from_float only works for one of"
            + str([float_mod.__name__ for float_mod in float_modules])
            + f" or their subclasses, but found {type(mod)}"
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"

        lowp_mode = 0
        act_quant_mode = 0
        cache_weight_for_large_batch = False
        if mod.qconfig is not None:
            if hasattr(mod.qconfig, "lowp_mode"):
                lowp_mode = mod.qconfig.lowp_mode
            if hasattr(mod.qconfig, "act_quant_mode"):
                act_quant_mode = mod.qconfig.act_quant_mode
            if hasattr(mod.qconfig, "cache_weight_for_large_batch"):
                cache_weight_for_large_batch = (
                    mod.qconfig.cache_weight_for_large_batch and lowp_mode in (2, 3)
                )

        w_dtype = qweight.dtype
        supported_qw_dtype = [
            torch.int32,
            torch.uint8,
            torch.quint4x2,
            torch.bfloat16,
            torch.float32,
        ]
        assert (
            w_dtype in supported_qw_dtype
        ), "Data type of int4 weight should be in {}, but got: {}".format(
            supported_qw_dtype, w_dtype
        )
        if not hasattr(mod, "in_features"):
            mod.in_features = mod.weight.size()[1]
        if not hasattr(mod, "out_features"):
            mod.out_features = mod.weight.size()[0]

        qlinear = cls(mod.in_features, mod.out_features, dtype=WoqWeightDtype.INT4)
        if mod.bias is not None:
            bias = mod.bias
        if bias is not None and torch.count_nonzero(bias) == 0:
            bias = None
        qlinear._op_context = torch.ops.ipex_prepack.weight_only_qlinear_prepack_int4(
            qweight,
            scales,
            zero_points,
            bias,
            g_idx,
            None,
            group_size,
            int(lowp_mode),
            act_quant_mode,
            cache_weight_for_large_batch,
        )
        qlinear.weight = qlinear._op_context.get_weight()
        qlinear.bias = bias is not None
        qlinear._lowp_mode = lowp_mode
        qlinear._act_quant_mode = act_quant_mode
        qlinear._group_size = group_size
        qlinear._cache_weight_for_large_batch = cache_weight_for_large_batch
        qlinear._weight_qscheme = (
            WoqWeightQScheme.ASYMMETRIC
            if zero_points is not None
            else WoqWeightQScheme.SYMMETRIC
        )
        del qweight
        return qlinear

    @classmethod
    def from_int4_weight(
        cls,
        qweight,
        scales,
        zero_points,
        in_features,
        out_features,
        quant_method,
        qconfig=None,
        bias=None,
        group_size=-1,
        g_idx=None,
    ):
        r"""Create a weight-only quantized module from int4 weight including autoAWQ and autoGPTQ format

        Args:
            qweight (Tensor): tensor in int32 dtype and contains actually int4 data
            scales (Tensor): scales for qweight
            zero_points (Tensor): zero points for qweight
            bias (Tensor or None): bias for linear
            in_features (int): size of each input sample
            out_features (int): size of each output sample
            qconfig (object): Defining the IPEX quantization recipe for Weight only quantization.
                Default value is ``None``.
            group_size: Group size for weight quantization
            g_idx: Indices of groups for each input channel of weight. Generated by
                GPTQ with act-order.
        """

        lowp_mode = 2
        act_quant_mode = 1
        cache_weight_for_large_batch = False
        if qconfig is not None and hasattr(qconfig, "global_qconfig"):
            if hasattr(qconfig.global_qconfig, "lowp_mode"):
                lowp_mode = qconfig.global_qconfig.lowp_mode
            if hasattr(qconfig.global_qconfig, "act_quant_mode"):
                act_quant_mode = qconfig.global_qconfig.act_quant_mode
            if hasattr(qconfig.global_qconfig, "cache_weight_for_large_batch"):
                cache_weight_for_large_batch = (
                    qconfig.global_qconfig.cache_weight_for_large_batch
                    and lowp_mode in (2, 3)
                )

        w_dtype = qweight.dtype
        supported_qw_dtype = [
            torch.int32,
            torch.uint8,
            torch.quint4x2,
        ]
        assert (
            w_dtype in supported_qw_dtype
        ), "Data type of int4 weight should be in {}, but got: {}".format(
            supported_qw_dtype, w_dtype
        )

        qlinear = cls(in_features, out_features, dtype=WoqWeightDtype.INT4)
        if quant_method == QuantMethod.AWQ_GEMM:
            qweight, scales, zero_points = prepack_awq_weight(
                qweight, zero_points, scales, 4, group_size
            )
        elif quant_method == QuantMethod.GPTQ_GEMM:
            qweight, scales, zero_points = _convert_optimum_format_to_desired(
                qweight, scales, zero_points, inplace=False
            )

        if bias is not None and torch.count_nonzero(bias) == 0:
            bias = None
        qlinear._op_context = torch.ops.ipex_prepack.weight_only_qlinear_prepack_int4(
            qweight,
            scales,
            zero_points,
            bias,
            g_idx,
            None,
            group_size,
            int(lowp_mode),
            act_quant_mode,
            cache_weight_for_large_batch,
        )
        qlinear.weight = qlinear._op_context.get_weight()
        qlinear.bias = bias is not None
        qlinear._lowp_mode = lowp_mode
        qlinear._act_quant_mode = act_quant_mode
        qlinear._group_size = group_size
        qlinear._cache_weight_for_large_batch = cache_weight_for_large_batch
        qlinear._weight_qscheme = (
            WoqWeightQScheme.ASYMMETRIC
            if zero_points is not None
            else WoqWeightQScheme.SYMMETRIC
        )
        del qweight
        return qlinear

    @classmethod
    def from_weight(
        cls,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        in_features: int,
        out_features: int,
        qconfig=None,
        bias: Optional[torch.Tensor] = None,
        group_size: int = -1,
        g_idx: Optional[torch.Tensor] = None,
        dtype: int = 0,
        quant_method: QuantMethod = QuantMethod.GPTQ_GEMM,
        **kwargs,
    ):
        r"""Create a weight-only quantized module from weight

        Args:
            qweight (Tensor): tensor in int32 dtype and contains actually int4 data
            scales (Tensor): scales for qweight
            zero_points (Tensor): zero points for qweight
            in_features (int): size of each input sample
            out_features (int): size of each output sample
            qconfig (object): Defining the IPEX quantization recipe for Weight only quantization.
                Default value is ``None``.
            bias (Tensor or None): bias for linear
            group_size: Group size for weight quantization
            g_idx: Indices of groups for each input channel of weight. Generated by
                GPTQ with act-order.
            quant_method (int): Quantization method, such as GPTQ=0, AWQ=1, ...
            dtype (int): quantization data type, INT4=0

        """
        if (
            quant_method in [QuantMethod.GPTQ_GEMM, QuantMethod.AWQ_GEMM]
            and dtype == QuantDtype.INT4
        ):
            return cls.from_int4_weight(
                qweight,
                scales,
                zero_points,
                in_features,
                out_features,
                quant_method,
                qconfig,
                bias,
                group_size,
                g_idx,
            )
        else:
            raise AssertionError(
                "Currently ipex.llm.quantization.IPEXWeightOnlyQuantizedLinear.from_weight() supports 4bits with AWQ or GPTQ."
            )

    @classmethod
    def _init_cls(
        cls,
        mod,
        dtype,
        qweight,
        scales,
        zero_points,
        g_idx,
        group_size,
        lowp_mode,
        act_quant_mode,
        cache_weight_for_large_batch,
    ):
        qlinear = cls(
            mod.in_features, mod.out_features, mod.bias is not None, dtype=dtype
        )
        qlinear._op_context = torch.ops.ipex_prepack.weight_only_qlinear_prepack(
            qweight,
            dtype,
            [mod.out_features, mod.in_features],
            scales,
            zero_points,
            mod.bias,
            g_idx,
            None,
            group_size,
            int(lowp_mode),
            act_quant_mode,
            cache_weight_for_large_batch,
        )
        qlinear.weight = qlinear._op_context.get_weight()
        qlinear.bias = mod.bias is not None
        qlinear._lowp_mode = lowp_mode
        qlinear._act_quant_mode = act_quant_mode
        qlinear._group_size = group_size
        qlinear._cache_weight_for_large_batch = cache_weight_for_large_batch
        qlinear._weight_qscheme = (
            WoqWeightQScheme.ASYMMETRIC
            if zero_points is not None
            else WoqWeightQScheme.SYMMETRIC
        )
        return qlinear


class IpexWoqLinearAllreduce(WeightOnlyQuantizedLinear):
    def __init__(
        self,
        in_features,
        out_features,
        mp_group,
        bias_value,
        bias_=True,
        dtype=WoqWeightDtype.INT8,
    ):
        # Save the original bias here
        # For bias handling, please refer to the comment in __init__ of _IPEXLinearAllreduce
        super().__init__(in_features, out_features, bias_, dtype=dtype)
        self.mp_group = mp_group
        self.original_bias = bias_value

    @classmethod
    def _init_from_mod(cls, mod, dtype):
        return cls(
            mod.in_features,
            mod.out_features,
            mod.mp_group,
            mod.bias,  # save the original bias value
            mod.bias is not None,
            dtype=dtype,
        )

    @classmethod
    def _init_cls(
        cls,
        mod,
        dtype,
        qweight,
        scales,
        zero_points,
        g_idx,
        group_size,
        lowp_mode,
        act_quant_mode,
        cache_weight_for_large_batch,
    ):
        qlinear = cls._init_from_mod(mod, dtype)

        qlinear._op_context = torch.ops.ipex_prepack.weight_only_qlinear_prepack(
            qweight,
            dtype,
            [mod.out_features, mod.in_features],
            scales,
            zero_points,
            None,  # Set bias to None when prepacking. Please refer to the comment in __init__ of _IPEXLinearAllreduce
            g_idx,
            None,  # batch_size
            group_size,
            lowp_mode,
            act_quant_mode,
            cache_weight_for_large_batch,
        )
        qlinear.weight = qlinear._op_context.get_weight()
        qlinear._lowp_mode = lowp_mode
        qlinear._act_quant_mode = act_quant_mode
        qlinear._group_size = group_size
        qlinear._cache_weight_for_large_batch = cache_weight_for_large_batch is not None

        return qlinear

    def post_ipex_gemm(self, output):
        return _all_reduce_and_bias_add(self.mp_group, self.original_bias, output)


class IpexWoqLmHeadLinearAllreduce(IpexWoqLinearAllreduce):
    def __init__(
        self,
        in_features,
        out_features,
        mp_group,
        rank,
        world_size,
        bias_value,
        bias_=True,
        dtype=WoqWeightDtype.INT8,
    ):
        # Save the original bias here
        # For bias handling, please refer to the comment in __init__ of _IPEXLinearAllreduce
        super().__init__(in_features, out_features, mp_group, bias_value, bias_, dtype)
        self.rank = rank
        self.world_size = world_size

    @classmethod
    def _init_from_mod(cls, mod, dtype):
        return cls(
            mod.in_features,
            mod.out_features,
            mod.mp_group,
            mod.rank,
            mod.world_size,
            mod.bias,  # save the original bias value
            mod.bias is not None,
            dtype=dtype,
        )

    def pre_ipex_gemm(self, input):
        return _pre_ipex_gemm(input, self.world_size, self.rank)
