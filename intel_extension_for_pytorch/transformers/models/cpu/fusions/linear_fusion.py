import torch
from torch import nn
import math
import warnings
from intel_extension_for_pytorch.nn.modules import IpexWoqLinear
from intel_extension_for_pytorch.quantization import (
    get_weight_only_quant_qconfig_mapping,
    dequantize_per_channel,
    dequantize_per_block,
)


class _IPEXlinearFusionCPU(nn.Module):
    def __init__(self, linear, tpp=False, woq=False):
        super().__init__()
        self.tpp = tpp
        self.woq = woq
        self.dtype = linear.weight.dtype if self.tpp else None

    def extra_repr(self):
        extra_repr_str = f"dtype = {self.dtype}, tpp = {self.tpp}, woq = {self.woq}"
        return extra_repr_str


class _IPEXlinearSiluCPU(_IPEXlinearFusionCPU):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__(module, tpp=tpp, woq=woq)
        self.linear = module

    def forward(self, x):
        if self.tpp and not self.linear.tpp_fallback:
            x = x.to(self.dtype).contiguous()
            return torch.ops.torch_ipex.tpp_linear_silu(
                x,
                self.linear.weight.detach(),
                self.linear.bias.detach()
                if self.linear.bias is not None
                else x.new_empty(0),
                self.linear.out_features,
            )
        else:  # fallback path
            return nn.functional.silu(self.linear(x))


class _IPEXlinearReluCPU(_IPEXlinearFusionCPU):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__(module, tpp=tpp, woq=woq)
        self.linear = module

    def forward(self, x):
        if self.tpp and not self.linear.tpp_fallback:
            x = x.to(self.dtype).contiguous()
            return torch.ops.torch_ipex.tpp_linear_relu(
                x,
                self.linear.weight.detach(),
                self.linear.bias.detach()
                if self.linear.bias is not None
                else x.new_empty(0),
                self.linear.out_features,
            )
        else:  # fallback path
            return nn.functional.relu(self.linear(x))


class _IPEXlinearMulCPU(_IPEXlinearFusionCPU):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__(module, tpp=tpp, woq=woq)
        self.linear = module

    def forward(self, x, y):
        if self.tpp and not self.linear.tpp_fallback:
            x = x.to(self.dtype).contiguous()
            y = y.to(self.dtype).contiguous()
            return torch.ops.torch_ipex.tpp_linear_mul(
                x,
                y,
                self.linear.weight.detach(),
                self.linear.bias.detach()
                if self.linear.bias is not None
                else x.new_empty(0),
                self.linear.out_features,
            )
        else:  # fallback path
            return self.linear(x) * y


class _IPEXlinearAddCPU(_IPEXlinearFusionCPU):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__(module, tpp=tpp, woq=woq)
        self.linear = module

    def forward(self, x, y):
        if self.tpp and not self.linear.tpp_fallback:
            x = x.to(self.dtype).contiguous()
            y = y.to(self.dtype).contiguous()
            return torch.ops.torch_ipex.tpp_linear_add(
                x,
                y,
                self.linear.weight.detach(),
                self.linear.bias.detach()
                if self.linear.bias is not None
                else x.new_empty(0),
                1.0,
                self.linear.out_features,
            )
        if (
            self.woq
            and hasattr(self.linear, "_op_context")
            and self.linear._op_context is not None
        ):
            return torch.ops.torch_ipex.woq_linear_add(
                x,
                self.linear._op_context.get_data_handle(),
                [y],
            )
        else:  # fallback path
            return self.linear(x) + y


class _IPEXlinearAddAddCPU(_IPEXlinearFusionCPU):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__(module, tpp=tpp, woq=woq)
        self.linear = module

    def forward(self, x, y, z):
        if self.tpp and not self.linear.tpp_fallback:
            x = x.to(self.dtype).contiguous()
            y = y.to(self.dtype).contiguous()
            z = z.to(self.dtype).contiguous()
            return torch.ops.torch_ipex.tpp_linear_add_add(
                x,
                y,
                z,
                self.linear.weight.detach(),
                self.linear.bias.detach()
                if self.linear.bias is not None
                else x.new_empty(0),
                1.0,
                self.linear.out_features,
            )
        if (
            self.woq
            and hasattr(self.linear, "_op_context")
            and self.linear._op_context is not None
        ):
            return torch.ops.torch_ipex.woq_linear_add_add(
                x,
                self.linear._op_context.get_data_handle(),
                [y, z],
            )
        else:  # fallback path
            return self.linear(x) + y + z


class _IPEXlinearNewGeluCPU(_IPEXlinearFusionCPU):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__(module, tpp=tpp, woq=woq)
        self.linear = module

    def forward(self, x):
        if self.tpp and not self.linear.tpp_fallback:
            x = x.to(self.dtype).contiguous()
            return torch.ops.torch_ipex.tpp_linear_gelu(
                x,
                self.linear.weight.detach(),
                self.linear.bias.detach()
                if self.linear.bias is not None
                else x.new_empty(0),
                self.linear.out_features,
            )
        elif (
            self.woq
            and hasattr(self.linear, "_op_context")
            and self.linear._op_context is not None
        ):
            return torch.ops.torch_ipex.woq_linear_new_gelu(
                x,
                self.linear._op_context.get_data_handle(),
            )
        else:  # fallback path
            x = self.linear(x)
            return (
                0.5
                * x
                * (
                    1.0
                    + torch.tanh(
                        math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                    )
                )
            )


class _IPEXlinearGeluCPU(_IPEXlinearFusionCPU):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__(module, tpp=tpp, woq=woq)
        self.linear = module
        self.gelu = nn.GELU()

    def forward(self, x):
        if self.tpp and not self.linear.tpp_fallback:
            x = x.to(self.dtype).contiguous()
            return torch.ops.torch_ipex.tpp_linear_gelu(
                x,
                self.linear.weight.detach(),
                self.linear.bias.detach()
                if self.linear.bias is not None
                else x.new_empty(0),
                self.linear.out_features,
            )
        if (
            self.woq
            and hasattr(self.linear, "_op_context")
            and self.linear._op_context is not None
        ):
            return torch.ops.torch_ipex.woq_linear_gelu(
                x,
                self.linear._op_context.get_data_handle(),
            )
        else:  # fallback path
            x = self.gelu(self.linear(x))
            return x


class _IPEXConcatLinearCPU(_IPEXlinearFusionCPU):
    def __init__(self, module, tpp=False, woq=False):
        assert hasattr(module, "linear_0")
        super().__init__(module.linear_0, tpp=tpp, woq=woq)
        assert hasattr(module, "num_concat")
        self.num_concat = module.num_concat
        self.concat_linear = None
        self.linear_list = []
        self.woq = woq
        self.tpp = tpp
        if woq:
            for i in range(self.num_concat):
                attr_name = f"linear_{i}"
                assert hasattr(module, attr_name)
                self.linear_list.append(getattr(module, attr_name))
        if woq and all(
            isinstance(linear, IpexWoqLinear) for linear in self.linear_list
        ):
            # Quantization is done before lowering to CPU.
            # We assume weights are all in shape [N, K].
            # We need to unpack weights then concat them
            weights_list = []
            scales_list = []
            zeros_list = []
            bias_list = []
            w_dtype = self.linear_list[0].dtype
            lowp_mode = self.linear_list[0]._lowp_mode
            act_quant_mode = self.linear_list[0]._act_quant_mode
            group_size = self.linear_list[0]._group_size
            qconfig_mapping = get_weight_only_quant_qconfig_mapping(
                weight_dtype=w_dtype,
                lowp_mode=lowp_mode,
                act_quant_mode=act_quant_mode,
                group_size=group_size,
            )
            qconfig = qconfig_mapping.global_qconfig
            for i in range(self.num_concat):
                linear = self.linear_list[i]
                if not hasattr(linear, "_op_context"):
                    warnings.warn(
                        "Concat linear fusion for CPU WOQ failed "
                        "because linear is not converted to WOQ Linear. "
                        "Falling back to separate linears."
                    )
                    weights_list = []
                    break
                qw = linear._op_context.to_public(linear._op_context.get_weight())
                scales = linear._op_context.get_scales()
                zero_points = linear._op_context.get_zero_points()
                is_int4 = w_dtype == torch.quint4x2
                weight_shape = linear._op_context.get_weight_shape()
                if group_size > 0:
                    weights_list.append(
                        dequantize_per_block(
                            qw, scales, zero_points, is_int4, group_size, weight_shape
                        )
                    )
                else:
                    weights_list.append(
                        dequantize_per_channel(
                            qw, scales, zero_points, is_int4, weight_shape
                        )
                    )
                # OC of Weight may be padded to a multiple of block_n. So are scales and zero points.
                bias = linear._op_context.get_bias()
                assert scales.shape == zero_points.shape
                assert bias is None or bias.shape[0] == scales.shape[0]
                if weight_shape[0] < scales.shape[0]:
                    original_n = weight_shape[0]
                    scales_list.append(scales.narrow(0, 0, original_n).contiguous())
                    zeros_list.append(zero_points.narrow(0, 0, original_n).contiguous())
                    bias_list.append(bias.narrow(0, 0, original_n).contiguous())
                else:
                    assert weight_shape[0] == scales.shape[0]
                    scales_list.append(scales)
                    zeros_list.append(zero_points)
                    bias_list.append(bias)
                w_dtype = linear.dtype
            if weights_list:
                concat_weight = torch.concat(weights_list, 0)
                concat_scales = torch.concat(scales_list, 0)
                concat_zeros = torch.concat(zeros_list, 0)
                use_bias = all(bias_list)
                concat_bias = torch.concat(bias_list, 0) if use_bias else None
                mod = nn.Linear(
                    concat_weight.shape[1], concat_weight.shape[0], use_bias
                )
                mod.weight = nn.Parameter(concat_weight)
                mod.bias = nn.Parameter(concat_bias) if use_bias else None
                mod.qconfig = qconfig
                mod._num_concats = len(weights_list)
                self._num_concats = mod._num_concats
                if w_dtype == torch.quint4x2:
                    self.concat_linear = IpexWoqLinear.from_float_and_int4_weight(
                        mod,
                        concat_weight,
                        concat_scales,
                        concat_zeros,
                        group_size=group_size,
                    )
                else:  # qint8
                    assert w_dtype == torch.qint8
                    self.concat_linear = IpexWoqLinear.from_float(
                        mod, concat_scales, concat_zeros
                    )
        else:
            self._num_concats = module._num_concats
            if (
                self.tpp
                and hasattr(module, "concat_linear")
                and module.concat_linear is not None
            ):
                self.concat_linear = module.concat_linear
            else:
                for i in range(self.num_concat):
                    attr_name = f"linear_{i}"
                    setattr(self, attr_name, getattr(module, attr_name))

    def forward(self, x):
        if self.concat_linear is not None:
            concat_output = self.concat_linear(x)
            if self.woq:
                num_concats = self._num_concats
                hidden_size = concat_output.shape[-1] // num_concats
                concat_output = concat_output.view(num_concats, -1, hidden_size)
                expected_shape = list(x.shape)[:-1] + [hidden_size]
                return tuple(
                    [concat_output[i].view(expected_shape) for i in range(num_concats)]
                )
            else:
                return concat_output

        output_list = []
        for i in range(self.num_concat):
            assert hasattr(self, f"linear_{i}")
            linear = getattr(self, f"linear_{i}")
            y = linear(x)
            output_list.append(y)
        return tuple(output_list)


class _IPEXlinearSiluMulCPU(nn.Module):
    def __init__(self, module_s, module_m, tpp=False, woq=False):
        super().__init__()
        self.tpp = tpp
        self.woq = woq
        self.linear_s = module_s
        self.linear_m = module_m
        self.dtype = module_s.weight.dtype if self.tpp else None

    def forward(self, x):
        if (
            self.tpp
            and not self.linear_s.tpp_fallback
            and not self.linear_m.tpp_fallback
        ):
            x = x.to(self.dtype).contiguous()
            return torch.ops.torch_ipex.tpp_fused_gate_up_proj(
                x,
                self.linear_s.weight.detach(),
                self.linear_s.bias.detach()
                if self.linear_s.bias is not None
                else x.new_empty(0),
                self.linear_m.weight.detach(),
                self.linear_m.bias.detach()
                if self.linear_m.bias is not None
                else x.new_empty(0),
            )
        else:  # fallback path
            return nn.functional.silu(self.linear_s(x)) * self.linear_m(x)
