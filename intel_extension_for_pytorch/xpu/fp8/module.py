from typing import Union, Optional, Tuple
import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import init
from intel_extension_for_pytorch.xpu.fp8.utils import cast_to_fp8, cast_if_needed

import intel_extension_for_pytorch._isa_help as ipex

from .fp8 import (
    is_fp8_enabled,
    get_fp8_recipe,
    get_fp8_dtype,
    amax_and_scale_update
)

class Fp8BaseModule(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super(Fp8BaseModule, self).__init__(**kwargs)
        self.fp8 = False
        self.fp8_meta = {}
        self.fp8_meta_tensors_initialized = False

    def set_meta_tensor(self, fwd: bool) -> None:
        """Init scales and amaxes for fwd | bwd."""
        fp8_meta_tensor_key = "scaling_fwd" if fwd else "scaling_bwd"
        num_fp8_tensors = (
            self.fp8_meta["num_gemms"] * 2 if fwd else self.fp8_meta["num_gemms"]
        )

        self.fp8_meta[fp8_meta_tensor_key] = ipex.FP8TensorMeta()
        self.fp8_meta[fp8_meta_tensor_key].scale = torch.ones(
            num_fp8_tensors, dtype=torch.float32, device="xpu"
        )
        self.fp8_meta[fp8_meta_tensor_key].scale_inv = torch.ones(
            num_fp8_tensors, dtype=torch.float32, device="xpu"
        )
        self.fp8_meta[fp8_meta_tensor_key].amax_history = torch.zeros(
            [self.fp8_meta["recipe"].amax_history_len,
             num_fp8_tensors],
            dtype=torch.float32,
            device="xpu",
        )

    def init_fp8_meta_tensors(self) -> None:
        """Init scales and amaxes."""
        # Checkpoint loaded
        if self.fp8_meta_tensors_initialized:
            return

        self.set_meta_tensor(True)
        self.set_meta_tensor(False)

        self.fp8_meta_tensors_initialized = True

    def fp8_init(self, num_gemms: int = 1) -> None:
        """Initialize fp8 related metadata and tensors during fprop."""
        # If fp8 isn't enabled, turn off and return.
        if not is_fp8_enabled():
            self.fp8 = False
            return

        # FP8 is already enabled and recipe is the same, don't do anything.
        if self.fp8 and get_fp8_recipe() == self.fp8_meta["recipe"]:
            return

        # Set FP8, recipe, and other FP8 metadata
        self.fp8 = True
        self.fp8_meta["recipe"] = get_fp8_recipe()
        self.fp8_meta["num_gemms"] = num_gemms

        # Set FP8_MAX per tensor according to recipe
        self.fp8_meta["fp8_max_fwd"] = self.fp8_meta["recipe"].fp8_format.value.max_fwd
        self.fp8_meta["fp8_max_bwd"] = self.fp8_meta["recipe"].fp8_format.value.max_bwd

        # Allocate scales and amaxes
        self.init_fp8_meta_tensors()

    def prepare_forward_backward(self, num_gemms: int = 1) -> None:
        self.fp8_init(num_gemms)

        if self.fp8:
            amax_and_scale_update(self.fp8_meta, True)
            amax_and_scale_update(self.fp8_meta, False)


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_,
        weight_,
        bias_,
        fp8_meta,
        use_bias,
        activation_dtype
    ) -> torch.Tensor:
        input = cast_if_needed(input_, activation_dtype)
        weight = cast_if_needed(weight_, activation_dtype)
        bias = cast_if_needed(bias_, activation_dtype) if use_bias else bias_

        fp8_dtype_forward = get_fp8_dtype(fp8_meta["recipe"], fprop_tensor=True)

        # TODO calibration path
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
            fp8_meta["scaling_fwd"].amax_history)
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
    def backward(ctx, grad_output: torch.Tensor
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
        grad_input = torch.ops.torch_ipex.fp8_gemm_backward(
            grad_output,
            fp8_dtype_backward,
            ipex.FP8BwdTensors.GRAD_OUTPUT1,
            weight,
            fp8_dtype_forward,
            ipex.FP8FwdTensors.GEMM1_WEIGHT,
            ctx.fp8_meta["scaling_bwd"].scale,
            ctx.fp8_meta["scaling_bwd"].scale_inv,
            ctx.fp8_meta["scaling_bwd"].amax_history)

        grad_weight = torch.ops.torch_ipex.fp8_gemm_backward(
            torch.permute(grad_output, (0, 2, 1)) if grad_output.dim() == 3 else torch.transpose(grad_output, 0, 1),
            fp8_dtype_backward,
            ipex.FP8BwdTensors.GRAD_OUTPUT1,
            input,
            fp8_dtype_forward,
            ipex.FP8FwdTensors.GEMM1_INPUT,
            ctx.fp8_meta["scaling_bwd"].scale,
            ctx.fp8_meta["scaling_bwd"].scale_inv,
            ctx.fp8_meta["scaling_bwd"].amax_history)

        return (grad_input, grad_weight, None, None, None, None)


class Linear(Fp8BaseModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=torch.device("xpu"),
        params_dtype: torch.dtype = torch.float32,
        activation_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.activation_dtype = activation_dtype

        self.weight = Parameter(
            torch.empty(
                (self.out_features,
                 self.in_features),
                device=torch.device("xpu"),
                dtype=params_dtype,
            )
        )

        if self.use_bias:
            self.bias = Parameter(
                torch.empty(
                    self.out_features,
                    device=torch.device("xpu"),
                    dtype=params_dtype,
                )
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

    def forward(
        self,
        inp: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:

        self.prepare_forward_backward()

        if torch.is_grad_enabled():
            linear_fn = _Linear.apply
            args = []
        else:
            linear_fn = _Linear.forward
            args = [None]

        activation_dtype = self.activation_dtype
        if torch.xpu.is_autocast_xpu_enabled():
            if torch.is_grad_enabled():
                activation_dtype = torch.bfloat16
            else:
                torch.get_autocast_gpu_dtype()

        args += (
            inp,
            self.weight,
            self.bias,
            self.fp8_meta,
            self.use_bias,
            activation_dtype
        )

        out = linear_fn(*args)

        return out
