import os

import torch
import torch.ao.nn.quantized as nnq
from torch.ao.nn.quantized.modules.utils import _quantize_weight
import torch.ao.nn.intrinsic as nni
from ...quantization._qconfig import get_weight_only_quant_qconfig_mapping
from intel_extension_for_pytorch.nn.utils._weight_prepack import (
    may_import_deepspeed_modules,
    _all_reduce_and_bias_add,
)

class ConcatLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, num_concats=1):
        super().__init__(in_features, out_features, bias, device, dtype)
        self._num_concats = num_concats

    def _get_name(self):
        return "IpexConcatLinear"

    def forward(self, input):
        y = super().forward(input)
        if self._num_concats > 1:
            # split results of concated linears
            y = y.view(-1, self._num_concats, y.size(-1)//self._num_concats).transpose(0, 1).contiguous()
            # reshape to match woq linear's output
            # Need to reshape back to get the correct shape
            y = y.view(-1, y.size(-1))
        return y

class IpexWoqLinear(nnq.Linear):
    r"""
    A weight-only quantized (WOQ) linear module with floating point tensor as inputs and outputs.
    Weight is dequantized at runtime for computation.
    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear for documentation.

    Similar to :class:`torch.nn.Linear`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module which are of
                         shape :math:`(\text{out\_features}, \text{in\_features})`.
        bias (Tensor): the non-learnable floating point bias of the module of shape
                       :math:`(\text{out\_features})`. If :attr:`bias` is ``True``,
                       the values are initialized to zero.

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = ipex.nn.IpexWoqLinear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    # version used in this class is different from the parent class nnq.Linear
    _version = 4

    def __init__(self, in_features, out_features, bias_=True, dtype=torch.qint8):
        # nnq.Linear does not support quint4x2 so we set qint8 here as a hack
        # This dtype is used for weight prepacking and we do not rely on the prepacking
        # of nnq.Linear. So, it won't affect our implementation here.
        super().__init__(in_features, out_features, bias_, dtype=torch.qint8)
        self._op_context = None
        self._weight_qscheme = self.weight().qscheme()
        self._lowp_mode = 0
        self._num_concats = 1

    def post_ipex_gemm(self, output):
        return output

    def forward(self, x):
        # return torch.ops.torch_ipex.ipex_woq_linear(
        #         x, self._op_context.get_data_handle()
        #     )
        # Note that we can handle self.bias == None case.
        if self._packed_params.dtype in [torch.qint8, torch.quint4x2]:
            Y = torch.ops.torch_ipex.ipex_woq_linear(
                x, self._op_context.get_data_handle()
            )
        else:
            raise RuntimeError("Unsupported dtype of wegiht only quantized linear!")
        if Y.dtype == x.dtype:
            output = Y
        else:
            output = Y.to(x.dtype)

        return self.post_ipex_gemm(output)

    def _get_name(self):
        return "IpexWeightOnlyQuantizedLinear"

    def extra_repr(self):
        extra_repr_str = "in_features={}, out_features={}, dtype={}".format(
            self.in_features, self.out_features, self._packed_params.dtype
        )
        if self._packed_params.dtype in [torch.qint8, torch.quint4x2]:
            extra_repr_str += ", qscheme={}".format(self._weight_qscheme)
        extra_repr_str += ", lowp_mode={}".format(self._lowp_mode)
        return extra_repr_str

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        assert (
            not keep_vars
        ), "can not using keep_vars true when to save IpexWoqLinear's parameters"
        if self.bias is not None:
            bias = self.bias.float()
            destination[prefix + "bias"] = bias.detach()
        weight = self.weight.float()
        destination[prefix + "weight"] = self.ctx.to_public(weight.detach())

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        with torch.no_grad():
            w_name = prefix + "weight"
            b_name = prefix + "bias"
            fp32_loaded_weight = state_dict[w_name]
            loaded_weight = fp32_loaded_weight.to(self.weight.dtype)
            if b_name in state_dict:
                loaded_bias = state_dict[b_name]
                loaded_bias = loaded_bias.to(self.bias.dtype)
            else:
                loaded_bias = None
            self._op_context = torch.ops.ipex_prepack.weight_only_qlinear_prepack(
                loaded_weight, loaded_bias, None
            )

    @classmethod
    def from_float(cls, mod):
        r"""Create a weight-only quantized module from a float module or qparams_dict

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
                          utilities or provided by the user
        """
        float_modules = [torch.nn.Linear, ConcatLinear]
        deepspeed_modules = may_import_deepspeed_modules()
        if deepspeed_modules is not None:
            float_modules.extend(deepspeed_modules)

        assert (
            type(mod) in float_modules
        ), "IpexWoqLinear.from_float only works for one of" + str(
            [float_mod.__name__ for float_mod in float_modules]
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        lowp_mode = 0
        if mod.qconfig is not None and mod.qconfig.weight is not None:
            weight_observer = mod.qconfig.weight()
            if hasattr(mod.qconfig, 'lowp_mode'):
                lowp_mode = mod.qconfig.lowp_mode
        else:
            weight_observer = (
                get_weight_only_quant_qconfig_mapping().global_qconfig.weight()
            )
        num_concats = 1
        if hasattr(mod, '_num_concats'):
            num_concats = mod._num_concats
        dtype = weight_observer.dtype
        assert dtype in [torch.qint8, torch.quint4x2], (
            "The only supported dtypes for "
            "weight-only quantized linear are qint8 and quint4x2 got: {}".format(dtype)
        )
        weight_observer(mod.weight)
        if dtype in [torch.qint8, torch.quint4x2]:
            qweight = _quantize_weight(mod.weight.float(), weight_observer)
        else:
            raise RuntimeError(
                "Unsupported dtype specified for dynamic quantized Linear!"
            )
        if not hasattr(mod, "in_features"):
            mod.in_features = mod.weight.size()[1]
        if not hasattr(mod, "out_features"):
            mod.out_features = mod.weight.size()[0]

        qlinear = cls._init_cls(mod, dtype, qweight, lowp_mode, num_concats)
        del qweight
        return qlinear

    @classmethod
    def _init_cls(cls, mod, dtype, qweight, lowp_mode, num_concats):
        qlinear = cls(mod.in_features, mod.out_features, dtype=dtype)
        qlinear._op_context = torch.ops.ipex_prepack.weight_only_qlinear_prepack(
            qweight, mod.bias, None, int(lowp_mode), num_concats
        )
        qlinear._lowp_mode = lowp_mode
        qlinear._num_concats = num_concats
        qlinear._weight_qscheme = qlinear.weight().qscheme()
        return qlinear

    @classmethod
    def from_reference(cls, ref_qlinear):
        """Create a weight-only quantized module from a reference quantized module
        Args:
            ref_qlinear (Module): a reference quantized  module, either produced by
            torch.ao.quantization functions or provided by the user
        """
        qlinear = cls(
            ref_qlinear.in_features,
            ref_qlinear.out_features,
            dtype=ref_qlinear.weight_dtype,
        )
        qweight = ref_qlinear.get_quantized_weight()
        bias = ref_qlinear.bias
        # qlinear.set_weight_bias(qweight, bias)
        qlinear._op_context = torch.ops.ipex_prepack.weight_only_qlinear_prepack(
            qweight, bias, None
        )
        qlinear.weight_qscheme = qlinear.weight().qscheme()
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
    def _init_cls(cls, mod, dtype, qweight, lowp_mode, num_concats):
        qlinear = cls(
            mod.in_features,
            mod.out_features,
            mod.mp_group,
            mod.bias,  # save the original bias value
            dtype=dtype,
        )
        # For bias handling, please refer to the comment in __init__ of _IPEXLinearAllreduce
        qlinear.set_weight_bias(qweight, None)

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
