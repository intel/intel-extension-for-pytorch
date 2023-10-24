import torch
import torch._inductor.ir as torch_ir
from torch._inductor.lowering import register_lowering
from . import ir as xpu_ir


def register_onednn_fusion_ops():
    @register_lowering(torch.ops.torch_ipex._convolution_pointwise)
    def convolution_unary(
        x: torch_ir.TensorBox,
        weight: torch_ir.TensorBox,
        bias: torch_ir.TensorBox,
        padding,
        stride,
        dilation,
        groups,
        attr,
        scalars,
        algorithm,
    ):
        return torch_ir.TensorBox.create(
            xpu_ir.ConvolutionUnary.create(
                x,
                weight,
                bias,
                padding,
                stride,
                dilation,
                groups,
                attr,
                scalars,
                algorithm,
            )
        )

    @register_lowering(torch.ops.torch_ipex._convolution_pointwise.binary)
    def convolution_binary(
        x: torch_ir.TensorBox,
        other: torch_ir.TensorBox,
        weight: torch_ir.TensorBox,
        bias: torch_ir.TensorBox,
        padding,
        stride,
        dilation,
        groups,
        binary_attr,
        binary_alpha,
        unary_attr,
        unary_scalars,
        unary_algorithm,
    ):
        return torch_ir.TensorBox.create(
            xpu_ir.ConvolutionBinary.create(
                x,
                other,
                weight,
                bias,
                padding,
                stride,
                dilation,
                groups,
                binary_attr,
                binary_alpha,
                unary_attr,
                unary_scalars,
                unary_algorithm,
            )
        )

    @register_lowering(torch.ops.torch_ipex._convolution_pointwise_.binary)
    def convolution_binary_inplace(
        x: torch_ir.TensorBox,
        other: torch_ir.TensorBox,
        weight: torch_ir.TensorBox,
        bias: torch_ir.TensorBox,
        padding,
        stride,
        dilation,
        groups,
        binary_attr,
        binary_alpha,
        unary_attr,
        unary_scalars,
        unary_algorithm,
    ):
        return torch_ir.TensorBox.create(
            xpu_ir.ConvolutionBinaryInplace.create(
                x,
                other,
                weight,
                bias,
                padding,
                stride,
                dilation,
                groups,
                binary_attr,
                binary_alpha,
                unary_attr,
                unary_scalars,
                unary_algorithm,
            )
        )

    @register_lowering(torch.ops.torch_ipex._linear_pointwise)
    def linear_unary(
        x: torch_ir.TensorBox,
        w: torch_ir.TensorBox,
        b: torch_ir.TensorBox,
        attr,
        scalars,
        algorithm,
    ):
        return torch_ir.TensorBox.create(
            xpu_ir.LinearUnary.create(x, w, b, attr, scalars, algorithm)
        )

    @register_lowering(torch.ops.torch_ipex._linear_pointwise.binary)
    def linear_binary(
        x: torch_ir.TensorBox,
        y: torch_ir.TensorBox,
        w: torch_ir.TensorBox,
        b: torch_ir.TensorBox,
        attr,
    ):
        return torch_ir.TensorBox.create(xpu_ir.LinearBinary.create(x, y, w, b, attr))


register_onednn_fusion_ops()
