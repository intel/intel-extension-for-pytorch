import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
import intel_extension_for_pytorch  # noqa
from torch.testing._internal.common_utils import TestCase

import torchvision

# import _ext as _backend


class _DCNv2(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        offset,
        mask,
        weight,
        bias,
        stride,
        padding,
        dilation,
        deformable_groups,
    ):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.deformable_groups = deformable_groups
        output = torch.ops.torch_ipex.dcn_v2_forward(
            input,
            weight,
            bias,
            offset,
            mask,
            ctx.kernel_size[0],
            ctx.kernel_size[1],
            ctx.stride[0],
            ctx.stride[1],
            ctx.padding[0],
            ctx.padding[1],
            ctx.dilation[0],
            ctx.dilation[1],
            ctx.deformable_groups,
        )
        ctx.save_for_backward(input, offset, mask, weight, bias)
        return output


dcn_v2_conv = _DCNv2.apply


class DCNv2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
    ):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        assert (
            2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
            == offset.shape[1]
        )
        assert (
            self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
            == mask.shape[1]
        )
        return dcn_v2_conv(
            input,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.deformable_groups,
        )


class DCN(DCNv2):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
    ):
        super(DCN, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            deformable_groups,
        )

        channels_ = (
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        )
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return dcn_v2_conv(
            input,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.deformable_groups,
        )


class TestNNMethod(TestCase):
    def test_dcn(self, dtype=torch.float):
        bs = 2
        cn = 3
        im_h = 8
        im_w = 8
        k_h = 3
        k_w = 3
        pad_h = 1
        pad_w = 1
        dilation_h = 1
        dilation_w = 1
        stride_h = 2
        stride_w = 2
        o_h = int((im_h + pad_h * 2 - (dilation_h * (k_h - 1) + 1)) / stride_h + 1)
        o_w = int((im_w + pad_w * 2 - (dilation_w * (k_w - 1) + 1)) / stride_w + 1)
        o_c = 5
        grp = 3
        input = torch.randn(bs, cn, im_h, im_w)
        weight = torch.randn(o_c, cn, k_h, k_w)
        mask = torch.randn(bs, grp * k_h * k_w, o_h, o_w)
        offset = torch.randn(bs, 2 * grp * k_h * k_w, o_h, o_w)
        bias = None
        ref = torchvision.ops.deform_conv2d(
            input,
            offset,
            weight,
            bias,
            (stride_h, stride_w),
            (pad_h, pad_w),
            (dilation_h, dilation_w),
            mask,
        )
        input_xpu = input.to("xpu")
        offset_xpu = offset.to("xpu")
        weight_xpu = weight.to("xpu")
        mask_xpu = mask.to("xpu")
        ret = torch.ops.torch_ipex.dcn_v2_forward(
            input_xpu,
            weight_xpu,
            bias,
            offset_xpu,
            mask_xpu,
            k_h,
            k_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            grp,
        )
        self.assertEqual(ret.to("cpu"), ref)
