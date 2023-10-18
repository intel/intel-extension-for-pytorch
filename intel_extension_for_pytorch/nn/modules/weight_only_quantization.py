import torch
from torch import nn
from torch.ao.nn.quantized.modules.utils import _clamp_weights
from ...quantization._qconfig import get_weight_only_quant_qconfig_mapping
from intel_extension_for_pytorch.nn.utils._weight_prepack import (
    may_import_deepspeed_modules,
    _all_reduce_and_bias_add,
    _pre_ipex_gemm,
)

# Port from PyTorch with a few changes
def _quantize_weight(float_wt, observer):
    wt_scale, wt_zp = observer.calculate_qparams()
    dtype = observer.dtype
    if observer.qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]:
        qweight = torch.quantize_per_tensor(
            float_wt,
            float(wt_scale), int(wt_zp), dtype)
        qweight = _clamp_weights(qweight, observer, wt_scale, wt_zp)
    elif observer.qscheme in [torch.per_channel_symmetric, torch.per_channel_affine]:
        wt_axis = observer.ch_axis
        qweight = torch.quantize_per_channel(
            float_wt,
            wt_scale.to(torch.double), wt_zp.to(torch.int64), wt_axis, dtype)
        qweight = _clamp_weights(qweight, observer, wt_scale, wt_zp)
    elif observer.qscheme in [torch.per_channel_affine_float_qparams]:
        qweight = torch.quantize_per_channel(
            float_wt,
            wt_scale.to(torch.float), wt_zp.to(torch.float), observer.ch_axis, dtype)
        qweight = _clamp_weights(qweight, observer, wt_scale, wt_zp)
    else:
        raise ValueError("Unexpected qscheme " + observer.qscheme)
    return qweight

class IpexWoqLinear(nn.Module):
    r"""
    A weight-only quantized (WOQ) linear module with floating point tensor as inputs and outputs.
    Weight is dequantized at runtime for computation.
    """
    # version used in this class is different from the parent class nnq.Linear
    _version = 4

    def __init__(self, in_features, out_features, bias_=True, dtype=torch.qint8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias_
        self.dtype = dtype
        self._op_context = None
        self._lowp_mode = 0
        self._num_concats = 1

    def pre_ipex_gemm(self, input):
        return input

    def post_ipex_gemm(self, output):
        return output

    def forward(self, x):
        x = self.pre_ipex_gemm(x)

        Y = torch.ops.torch_ipex.ipex_woq_linear(
            x, self._op_context.get_data_handle()
        )

        return self.post_ipex_gemm(Y)

    def _get_name(self):
        return "IpexWeightOnlyQuantizedLinear"

    def extra_repr(self):
        extra_repr_str = "in_features={}, out_features={}, dtype={}".format(
            self.in_features, self.out_features, self.dtype
        )
        extra_repr_str += ", bias={}".format(self.bias)
        extra_repr_str += ", lowp_mode={}".format(self._lowp_mode)
        extra_repr_str += ", num_concats={}".format(self._num_concats)
        return extra_repr_str

    @classmethod
    def from_float(cls, mod):
        r"""Create a weight-only quantized module from a float module or qparams_dict

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
                          utilities or provided by the user
        """
        float_modules = [torch.nn.Linear]
        deepspeed_modules = may_import_deepspeed_modules()
        if deepspeed_modules is not None:
            float_modules.extend(deepspeed_modules)
        if any(issubclass(type(mod), float_module) for float_module in float_modules):
            float_modules.extend([type(mod)])

        assert (
            type(mod) in float_modules
        ), "IpexWoqLinear.from_float only works for one of" + str(
            [float_mod.__name__ for float_mod in float_modules]
        ) + f" or their subclasses, but found {type(mod)}"
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        lowp_mode = 0
        if mod.qconfig is not None and mod.qconfig.weight is not None:
            weight_observer = mod.qconfig.weight()
            if hasattr(mod.qconfig, 'lowp_mode'):
                lowp_mode = mod.qconfig.lowp_mode
                if mod.qconfig.lowp_mode == 3 and weight_observer.dtype == torch.qint8:
                    # lowp_mode=3 (INT8) is not yet supported for INT8 weight
                    # Fall back to lowp_mode=2 in this case
                    # TODO(Weiwen) Support lowp_mode=3
                    lowp_mode = 2
                    print('Warning: lowp_mode=3(INT8) is not supported yet in this case. '
                          'Falling back to 2(BF16).')
        else:
            weight_observer = (
                get_weight_only_quant_qconfig_mapping().global_qconfig.weight()
            )
        num_concats = 1
        if hasattr(mod, '_num_concats'):
            num_concats = mod._num_concats
        dtype = weight_observer.dtype
        assert dtype in [torch.quint8, torch.qint8, torch.quint4x2], (
            "The only supported dtypes for "
            "weight-only quantized linear are quint8, qint8 and quint4x2 got: {}".format(dtype)
        )
        weight_observer(mod.weight)
        qweight = _quantize_weight(mod.weight.float(), weight_observer)
        if not hasattr(mod, "in_features"):
            mod.in_features = mod.weight.size()[1]
        if not hasattr(mod, "out_features"):
            mod.out_features = mod.weight.size()[0]

        qlinear = cls._init_cls(mod, dtype, qweight, lowp_mode, num_concats)
        del qweight
        return qlinear

    @classmethod
    def from_float_and_int4_weight(cls, mod, qweight, scales, zero_points, bias=None):
        r"""Create a weight-only quantized module from a float module and int4 weight

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
                          utilities or provided by the user
            qweight (Tensor): tensor in int32 dtype and contains actually int4 data
            bias (Tensor or None): bias for linear
            scales (Tensor): scales for qweight
            zero_points (Tensor): zero points for qweight
        """
        float_modules = [torch.nn.Linear]
        deepspeed_modules = may_import_deepspeed_modules()
        if deepspeed_modules is not None:
            float_modules.extend(deepspeed_modules)
        if any(issubclass(type(mod), float_module) for float_module in float_modules):
            float_modules.extend([type(mod)])

        assert (
            type(mod) in float_modules
        ), "IpexWoqLinear.from_float only works for one of" + str(
            [float_mod.__name__ for float_mod in float_modules]
        ) + f" or their subclasses, but found {type(mod)}"
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"

        lowp_mode = 0
        if mod.qconfig is not None and hasattr(mod.qconfig, 'lowp_mode'):
            lowp_mode = mod.qconfig.lowp_mode
        num_concats = 1
        if hasattr(mod, '_num_concats'):
            num_concats = mod._num_concats

        w_dtype = qweight.dtype
        assert w_dtype in [torch.int32, torch.quint4x2, torch.bfloat16, torch.float32], (
            "Quantized int4 weight should have data type int32 or quint4x2, but got: {}".format(w_dtype)
        )
        if not hasattr(mod, "in_features"):
            mod.in_features = mod.weight.size()[1]
        if not hasattr(mod, "out_features"):
            mod.out_features = mod.weight.size()[0]

        qlinear = cls(mod.in_features, mod.out_features, dtype=torch.quint4x2)
        if bias is None:
            bias = mod.bias
        qlinear._op_context = torch.ops.ipex_prepack.weight_only_qlinear_prepack_int4(
            qweight, scales, zero_points, bias, None, int(lowp_mode), num_concats
        )
        qlinear._lowp_mode = lowp_mode
        qlinear._num_concats = num_concats
        del qweight
        return qlinear

    @classmethod
    def _init_cls(cls, mod, dtype, qweight, lowp_mode, num_concats):
        qlinear = cls(mod.in_features, mod.out_features, mod.bias is not None, dtype=dtype)
        qlinear._op_context = torch.ops.ipex_prepack.weight_only_qlinear_prepack(
            qweight, mod.bias, None, int(lowp_mode), num_concats
        )
        qlinear._lowp_mode = lowp_mode
        qlinear._num_concats = num_concats
        return qlinear

class IpexWoqLinearAllreduce(IpexWoqLinear):
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

    @classmethod
    def _init_from_mod(cls, mod, dtype):
        return cls(
            mod.in_features,
            mod.out_features,
            mod.mp_group,
            mod.bias,  # save the original bias value
            dtype=dtype,
        )

    @classmethod
    def _init_cls(cls, mod, dtype, qweight, lowp_mode, num_concats):
        qlinear = cls._init_from_mod(mod, dtype)

        qlinear._op_context = torch.ops.ipex_prepack.weight_only_qlinear_prepack(
            qweight,
            None,  # Set bias to None when prepacking. Please refer to the comment in __init__ of _IPEXLinearAllreduce
            None,  # batch_size
            lowp_mode,
            num_concats
        )

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
        dtype=torch.qint8,
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
            dtype=dtype,
        )

    def pre_ipex_gemm(self, input):
        return _pre_ipex_gemm(input, self.world_size, self.rank)
