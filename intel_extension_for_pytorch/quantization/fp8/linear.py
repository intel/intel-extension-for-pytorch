################################################################################
# Copyright (C) 2025 Intel Corporation
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you ("License"). Unless the License provides otherwise,
# you may not use, modify, copy, publish, distribute, disclose or transmit
# this software or the related documents without Intel's prior written
# permission. This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly
# stated in the License.
################################################################################
from typing import Union, Optional, Tuple
import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import init
from intel_extension_for_pytorch.quantization.fp8.util import (
    cast_if_needed,
    cast_to_fp8,
)
import intel_extension_for_pytorch._C as ipex
from .base import Fp8BaseModule, prepare_backward
from intel_extension_for_pytorch.quantization.fp8.fp8 import (
    get_fp8_dtype,
)


class _FP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_,
        weight_,
        bias_,
        fp8_meta,
        use_bias,
        activation_dtype,
        is_grad_enabled=False,
        fp8_calibration=False,
    ) -> torch.Tensor:
        input = cast_if_needed(input_, activation_dtype)
        weight = cast_if_needed(weight_, activation_dtype)
        bias = cast_if_needed(bias_, activation_dtype) if use_bias else bias_

        fp8_dtype_forward = get_fp8_dtype(fp8_meta["recipe"], fprop_tensor=True)

        if fp8_calibration:
            fp8_meta["scaling_fwd"].amax_history[0][ipex.FP8FwdTensors.GEMM1_INPUT] = (
                torch.abs(inputmat).max().float()
            )
            fp8_meta["scaling_fwd"].amax_history[0][ipex.FP8FwdTensors.GEMM1_WEIGHT] = (
                torch.abs(weight).max().float()
            )
            calibration_out = input @ weight.transpose(0, 1)
            if use_bias:
                calibration_out = calibration_out + bias
            return calibration_out

        input_fp8 = cast_to_fp8(
            input,
            fp8_meta["scaling_fwd"],
            ipex.FP8FwdTensors.GEMM1_INPUT,
            fp8_dtype_forward,
        )
        weight_fp8 = cast_to_fp8(
            weight,
            fp8_meta["scaling_fwd"],
            ipex.FP8FwdTensors.GEMM1_WEIGHT,
            fp8_dtype_forward,
        )
        output = torch.ops.torch_ipex.fp8_gemm(
            input_fp8,
            fp8_dtype_forward,
            ipex.FP8FwdTensors.GEMM1_INPUT,
            weight_fp8,
            fp8_dtype_forward,
            ipex.FP8FwdTensors.GEMM1_WEIGHT,
            bias,
            fp8_meta["scaling_fwd"].scale,
            fp8_meta["scaling_fwd"].scale_inv,
            fp8_meta["scaling_fwd"].amax_history,
        )

        output = (
            output
            * fp8_meta["scaling_fwd"].scale_inv[0]
            * fp8_meta["scaling_fwd"].scale_inv[1]
        )
        ctx.save_for_backward(
            input,
            input_fp8,
            weight,
            weight_fp8,
            fp8_meta["scaling_fwd"].scale_inv.clone(),
        )
        ctx.activation_dtype = activation_dtype
        ctx.fp8_meta = fp8_meta
        ctx.use_bias = use_bias
        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        (
            input,
            input_fp8,
            weight,
            weight_fp8,
            fp8_scale_invs,
        ) = ctx.saved_tensors

        fp8_dtype_forward = get_fp8_dtype(ctx.fp8_meta["recipe"], fprop_tensor=True)
        fp8_dtype_backward = get_fp8_dtype(ctx.fp8_meta["recipe"], fprop_tensor=False)

        (input, input_fp8, weight, weight_fp8, fwd_scale_inverses) = ctx.saved_tensors
        grad_output_fp8 = cast_to_fp8(
            grad_output,
            ctx.fp8_meta["scaling_bwd"],
            ipex.FP8BwdTensors.GRAD_OUTPUT1,
            fp8_dtype_backward,
        )
        if ctx.use_bias:
            grad_bias = grad_output.sum(dim=0)
        else:
            grad_bias = None

        grad_input = torch.ops.torch_ipex.fp8_gemm_backward(
            grad_output_fp8,
            fp8_dtype_backward,
            ipex.FP8BwdTensors.GRAD_OUTPUT1,
            weight_fp8,
            fp8_dtype_forward,
            ipex.FP8FwdTensors.GEMM1_WEIGHT,
            ctx.fp8_meta["scaling_bwd"].scale,
            ctx.fp8_meta["scaling_bwd"].scale_inv,
            ctx.fp8_meta["scaling_bwd"].amax_history,
        )

        grad_weight = torch.ops.torch_ipex.fp8_gemm_backward(
            (
                torch.permute(grad_output_fp8, (0, 2, 1))
                if grad_output_fp8.dim() == 3
                else torch.transpose(grad_output_fp8, 0, 1)
            ),
            fp8_dtype_backward,
            ipex.FP8BwdTensors.GRAD_OUTPUT1,
            input_fp8,
            fp8_dtype_forward,
            ipex.FP8FwdTensors.GEMM1_INPUT,
            ctx.fp8_meta["scaling_bwd"].scale,
            ctx.fp8_meta["scaling_bwd"].scale_inv,
            ctx.fp8_meta["scaling_bwd"].amax_history,
        )

        grad_input = (
            grad_input
            * ctx.fp8_meta["scaling_fwd"].scale_inv[1]
            * ctx.fp8_meta["scaling_bwd"].scale_inv[0]
        )
        grad_weight = (
            grad_weight
            * ctx.fp8_meta["scaling_fwd"].scale_inv[0]
            * ctx.fp8_meta["scaling_bwd"].scale_inv[0]
        )

        if ctx.use_bias:
            grad_bias = grad_bias * ctx.fp8_meta["scaling_bwd"].scale_inv[0]

        return (grad_input, grad_weight, grad_bias, None, None, None, None, None)


class _FP8Linear_cpu(torch.autograd.Function):
    """FP8Linear implementation with backward."""

    @staticmethod
    def forward(
        ctx,
        input_,
        weight_,
        bias_,
        fp8_meta,
        use_bias,
        activation_dtype,
        is_grad_enabled,
        fp8_calibration,
    ):
        input = cast_if_needed(input_, activation_dtype)
        weight = cast_if_needed(weight_, activation_dtype)
        bias_dtype = (
            torch.bfloat16 if activation_dtype == torch.float32 else activation_dtype
        )
        bias = cast_if_needed(bias_, bias_dtype) if use_bias else bias_

        in_features = weight.shape[-1]
        assert input.shape[-1] == in_features, "GEMM not possible"
        inputmat = input.view((-1, in_features))

        fp8_dtype_forward = get_fp8_dtype(fp8_meta["recipe"], fprop_tensor=True)
        if fp8_calibration:
            fp8_meta["scaling_fwd"].amax_history[0][ipex.FP8FwdTensors.GEMM1_INPUT] = (
                torch.abs(inputmat).max().float()
            )
            fp8_meta["scaling_fwd"].amax_history[0][ipex.FP8FwdTensors.GEMM1_WEIGHT] = (
                torch.abs(weight).max().float()
            )
            calibration_out = input @ weight.transpose(0, 1)
            if use_bias:
                calibration_out = calibration_out + bias
            return calibration_out

        inputmat_fp8 = cast_to_fp8(
            inputmat,
            fp8_meta["scaling_fwd"],
            ipex.FP8FwdTensors.GEMM1_INPUT,
            fp8_dtype_forward,
        )

        weight_fp8 = cast_to_fp8(
            weight,
            fp8_meta["scaling_fwd"],
            ipex.FP8FwdTensors.GEMM1_WEIGHT,
            fp8_dtype_forward,
        )

        dim_size = list(inputmat.size())
        dim_size[-1] = weight.size(0)
        out = torch.empty(dim_size, dtype=activation_dtype, device=input.device)

        _ = torch.ops.torch_ipex.fp8_linear(
            inputmat_fp8,
            fp8_meta["scaling_fwd"].scale_inv,
            ipex.FP8FwdTensors.GEMM1_INPUT,
            fp8_dtype_forward,
            weight_fp8,
            fp8_meta["scaling_fwd"].scale_inv,
            ipex.FP8FwdTensors.GEMM1_WEIGHT,
            fp8_dtype_forward,
            bias,
            out,
        )

        if is_grad_enabled:
            ctx.save_for_backward(
                inputmat,
                inputmat_fp8,
                weight,
                weight_fp8,
                fp8_meta["scaling_fwd"].scale_inv.clone(),
            )
            ctx.fp8_meta = fp8_meta
            ctx.inp_shape = input.shape
            ctx.activation_dtype = activation_dtype
            ctx.fp8_meta = fp8_meta
            ctx.use_bias = use_bias
        return out.view(-1, *input.shape[1:-1], out.shape[-1])

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        with prepare_backward(ctx.fp8_meta):
            (
                inp,
                inp_fp8,
                weight,
                weight_fp8,
                fwd_scale_inverses,
            ) = ctx.saved_tensors
            fp8_dtype_forward = get_fp8_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=False
            )
            fp8_dtype_backward = get_fp8_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=False
            )

            # grad_output preprocess
            grad_output = grad_output.contiguous()
            grad_output_mat = grad_output.view((-1, grad_output.shape[-1]))
            grad_output_fp8 = cast_to_fp8(
                grad_output_mat,
                ctx.fp8_meta["scaling_bwd"],
                ipex.FP8BwdTensors.GRAD_OUTPUT1,
                fp8_dtype_backward,
            )
            if ctx.use_bias:
                grad_bias = grad_output_mat.sum(dim=0)
            else:
                grad_bias = None

            # TODO:  Should store weight transpose
            dgrad = torch.ops.torch_ipex.fp8_linear(
                grad_output_fp8,
                ctx.fp8_meta["scaling_bwd"].scale_inv,
                ipex.FP8BwdTensors.GRAD_OUTPUT1,
                fp8_dtype_backward,
                weight_fp8.transpose(0, 1).contiguous(),
                fwd_scale_inverses,
                ipex.FP8FwdTensors.GEMM1_WEIGHT,
                fp8_dtype_forward,
                None,
                None,
            )

            # TODO:  Should store inp transpose
            wgrad = torch.ops.torch_ipex.fp8_linear(
                grad_output_fp8.transpose(0, 1).contiguous(),
                ctx.fp8_meta["scaling_bwd"].scale_inv,
                ipex.FP8BwdTensors.GRAD_OUTPUT1,
                fp8_dtype_backward,
                inp_fp8.transpose(0, 1).contiguous(),
                fwd_scale_inverses,
                ipex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                None,
                None,
            )
        return (
            dgrad.view(ctx.inp_shape),
            wgrad,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
        )


class FP8Linear(Fp8BaseModule):
    """Linear function of FP8 data type,  from Fp8BaseModule."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Union[torch.device, str] = "xpu",
        params_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        params_dtype = (
            torch.get_default_dtype() if params_dtype is None else params_dtype
        )
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weight = Parameter(
            torch.empty(
                self.out_features, self.in_features, device=device, dtype=params_dtype
            )
        )

        if self.use_bias:
            self.bias = Parameter(
                torch.empty(self.out_features, device=device, dtype=params_dtype)
            )
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor):
        is_cpu = True if input.device.type == "cpu" else False

        with self.prepare_forward(device=input.device):
            if torch.is_grad_enabled():
                fn = _FP8Linear.apply if not is_cpu else _FP8Linear_cpu.apply
                args = []
            else:
                fn = _FP8Linear.forward if not is_cpu else _FP8Linear_cpu.forward
                args = [None]

            activation_dtype = torch.float32
            if torch.xpu.is_autocast_xpu_enabled():
                activation_dtype = torch.get_autocast_gpu_dtype()
            elif is_cpu and torch.is_autocast_cpu_enabled():
                activation_dtype = torch.get_autocast_cpu_dtype()

            args += (
                input,
                self.weight,
                self.bias,
                self.fp8_meta,
                self.use_bias,
                activation_dtype,
                torch.is_grad_enabled(),
                self.fp8_calibration,
            )
            out = fn(*args)
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"
