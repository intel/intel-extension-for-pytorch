import torch
import torch.nn as nn
from importlib import import_module
from typing import Optional


class IPEXFP8ScaledQuant(nn.Module):
    r"""
    A FP8 scaled quant module with floating point tensor as inputs and float8 as outputs.
    """

    def __init__(self, fp8_linear_impl):
        super().__init__()
        self.fp8_linear_impl = fp8_linear_impl

    module_mapping = {
        "xpu": "intel_extension_for_pytorch.nn.utils._quantize_convert",
    }

    impl_name = "FP8ScaledQuant"

    @classmethod
    def scaled_fp8_quant(
        cls,
        input: torch.Tensor,
        out_dtype: Optional[torch.dtype] = torch.float8_e5m2,
        scale: Optional[torch.Tensor] = None,
        num_token_padding: Optional[int] = None,
        scale_ub: Optional[torch.Tensor] = None,
        use_per_token_if_dynamic: bool = False,
    ):
        """
        Quantize input tensor to FP8 and return quantized tensor and scale.

        This function supports both static and dynamic quantization: If you
        provide the scale, it will use static scaling and if you omit it,
        the scale will be determined dynamically. The function also allows
        optional padding of the output tensors for downstream kernels that
        will benefit from padding.

        Args:
            input: The input tensor to be quantized to FP8
            scale: Optional scaling factor for the FP8 quantization
            scale_ub: Optional upper bound for scaling factor in dynamic
                per token case
            num_token_padding: If specified, pad the first dimension
                of the output to at least this value.
            use_per_token_if_dynamic: Whether to do per_tensor or per_token
                in the dynamic quantization case.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
                scaling factor.
        """
        device_type = input.device.type
        assert device_type in {"xpu"}, "Device type not supported."
        fp8_quant_impl_cls = getattr(
            import_module(cls.module_mapping[device_type]), cls.impl_name
        )
        return fp8_quant_impl_cls.scaled_fp8_quant(
            input,
            out_dtype,
            scale,
            num_token_padding,
            scale_ub,
            use_per_token_if_dynamic,
        )
