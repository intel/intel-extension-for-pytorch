import torch
import torch.nn as nn

from torch.testing._internal.common_utils import dtype2prec_DONTUSE
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import repeat_test_for_types
from torch.testing._internal.common_device_type import dtypes

import intel_extension_for_pytorch  # noqa
import pytest
ALL_TENSORTYPES = [torch.float,
                   torch.double,
                   torch.bfloat16]

#  backup default dtype
dtype_origin = torch.get_default_dtype()

# Depthwise convolution is very similar to test_Conv2d_naive_groups but with special care to handle
# the number of groups == number of input channels


class TestNN(NNTestCase):
    _do_xpu_memory_leak_check = True
    _do_xpu_non_default_stream = True

    @repeat_test_for_types(ALL_TENSORTYPES)
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_conv2d_depthwise(self, dtype=torch.float):
        torch.set_default_dtype(torch.double)
        for depth_multiplier in [1, 2]:
            m = nn.Conv2d(2, 2 * depth_multiplier, kernel_size=3, groups=2).to("xpu", dtype)
            i = torch.randn(2, 2, 6, 6, device="xpu", dtype=dtype).div_(2).requires_grad_()
            output = m(i)
            grad_output = torch.randn(2, 2 * depth_multiplier, 4, 4, device="xpu", dtype=dtype) / 2
            output.backward(grad_output)

            offset = 1 * depth_multiplier

            m1 = nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).to("xpu", dtype)
            m1.weight.data = m.weight.data[:offset].clone()
            m1.bias.data = m.bias.data[:offset].clone()
            i1 = i.detach()[:, :1].clone().requires_grad_()
            output1 = m1(i1)
            output1.backward(grad_output[:, :offset].contiguous())

            m2 = nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).to("xpu", dtype)
            m2.weight.data.copy_(m.weight.data[offset:])
            m2.bias.data.copy_(m.bias.data[offset:])
            i2 = i.detach()[:, 1:].clone().requires_grad_()
            output2 = m2(i2)
            output2.backward(grad_output[:, offset:].contiguous())

            self.assertEqual(output, torch.cat([output1, output2], 1),
                             atol=dtype2prec_DONTUSE[dtype], rtol=0)
            self.assertEqual(i.grad.data,
                             torch.cat([i1.grad.data, i2.grad.data], 1),
                             atol=dtype2prec_DONTUSE[dtype], rtol=0)
            self.assertEqual(m.bias.grad.data,
                             torch.cat([m1.bias.grad.data,
                                        m2.bias.grad.data], 0),
                             atol=dtype2prec_DONTUSE[dtype], rtol=0)
            self.assertEqual(m.weight.grad.data,
                             torch.cat([m1.weight.grad.data,
                                        m2.weight.grad.data], 0),
                             atol=dtype2prec_DONTUSE[dtype], rtol=0)
        torch.set_default_dtype(dtype_origin)

    @repeat_test_for_types(ALL_TENSORTYPES)
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_Conv3d_depthwise(self, dtype=torch.float):
        torch.set_default_dtype(torch.double)
        for depth_multiplier in [1, 2]:
            m = nn.Conv3d(2, 2 * depth_multiplier, kernel_size=3, groups=2).to("xpu", dtype)
            i = torch.randn(2, 2, 6, 6, 6, device="xpu", dtype=dtype).div_(2).requires_grad_()
            output = m(i)
            grad_output = torch.randn(2, 2 * depth_multiplier, 4, 4, 4, device="xpu", dtype=dtype) / 2
            output.backward(grad_output)

            offset = 1 * depth_multiplier

            m1 = nn.Conv3d(1, 1 * depth_multiplier, kernel_size=3).to("xpu", dtype)
            m1.weight.data = m.weight.data[:offset].clone()
            m1.bias.data = m.bias.data[:offset].clone()
            i1 = i.detach()[:, :1].clone().requires_grad_()
            output1 = m1(i1)
            output1.backward(grad_output[:, :offset].contiguous())

            m2 = nn.Conv3d(1, 1 * depth_multiplier, kernel_size=3).to("xpu", dtype)
            m2.weight.data.copy_(m.weight.data[offset:])
            m2.bias.data.copy_(m.bias.data[offset:])
            i2 = i.detach()[:, 1:].clone().requires_grad_()
            output2 = m2(i2)
            output2.backward(grad_output[:, offset:].contiguous())

            self.assertEqual(output, torch.cat([output1, output2], 1),
                             atol=dtype2prec_DONTUSE[dtype], rtol=0)
            self.assertEqual(i.grad.data,
                             torch.cat([i1.grad.data, i2.grad.data], 1),
                             atol=dtype2prec_DONTUSE[dtype], rtol=0)
            self.assertEqual(m.bias.grad.data,
                             torch.cat([m1.bias.grad.data,
                                        m2.bias.grad.data], 0),
                             atol=dtype2prec_DONTUSE[dtype], rtol=0)
            self.assertEqual(m.weight.grad.data,
                             torch.cat([m1.weight.grad.data,
                                        m2.weight.grad.data], 0),
                             atol=dtype2prec_DONTUSE[dtype], rtol=0)
        torch.set_default_dtype(dtype_origin)

    @dtypes(torch.double)
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_Conv2d_backward_depthwise(self, device="xpu", dtype=torch.double):
        torch.set_default_dtype(torch.double)
        x = torch.randn(2, 2, 4, 20, device=device, dtype=dtype, requires_grad=True)
        weight = torch.randn(2, 1, 3, 5, device=device, dtype=dtype, requires_grad=True)

        def conv2d_depthwise(x, weight):
            return torch.nn.functional.conv2d(
                x, weight, bias=None, stride=(1, 10), groups=2)

        torch.autograd.gradcheck(conv2d_depthwise, (x, weight))
        torch.set_default_dtype(dtype_origin)

    @dtypes(torch.double)
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_Conv3d_backward_depthwise(self, device="xpu", dtype=torch.double):
        torch.set_default_dtype(torch.double)
        x = torch.randn(1, 2, 5, 5, 5, device=device, dtype=dtype, requires_grad=True)
        weight = torch.randn(4, 1, 3, 3, 3, device=device, dtype=dtype, requires_grad=True)

        def conv3d_depthwise(x, weight):
            return torch.nn.functional.conv3d(
                x, weight, bias=None, stride=(1, 1, 2), groups=2)

        torch.autograd.gradcheck(conv3d_depthwise, (x, weight))
        torch.set_default_dtype(dtype_origin)
