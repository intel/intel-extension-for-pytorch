import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

# Note:
# In order to press the gradient of weight below 1,
# the default weight should be set to 1e-ks (ks is kernel_size).
# For now, precision could not be pressed to 1e-5,
# but only if there is a real model which suffers the accuracy problem,
# we won't delve into this issue.


class TestNNMethod(TestCase):
    def test_conv2d(self, dtype=torch.float):
        x_cpu = torch.randn([1, 64, 256, 256], dtype=dtype, device=cpu_device, requires_grad=True)
        grad_cpu = torch.full([1, 64, 256, 256], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True)
        conv_cpu = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()

        x_dpcpp = x_cpu.to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        print("ref (cpu):\n", "output:\n", y_cpu, "\ngrad weight:\n", y_cpu_gw)
        print("real (dpcpp):\n", "output:\n", y_dpcpp.cpu(), "\ngrad weight:\n", y_dpcpp_gw.cpu())

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=0)

    def test_conv2d_with_bias(self, dtype=torch.float):
        x_cpu = torch.randn([1, 64, 256, 256], dtype=dtype, device=cpu_device, requires_grad=True)
        grad_cpu = torch.full([1, 64, 256, 256], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True)
        conv_cpu = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()

        x_dpcpp = x_cpu.to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        print("ref (cpu):\n", "output:\n", y_cpu, "\ngrad weight:\n", y_cpu_gw)
        print("real (dpcpp):\n", "output:\n", y_dpcpp.cpu(), "\ngrad weight:\n", y_dpcpp_gw.cpu())

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=0)

    def test_conv2d_dilated(self, dtype=torch.float):
        x_cpu = torch.randn([1, 64, 256, 256], dtype=dtype, device=cpu_device, requires_grad=True)
        grad_cpu = torch.full([1, 64, 254, 254], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True)
        conv_cpu = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=2, bias=False)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()

        x_dpcpp = x_cpu.to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        print("ref (cpu):\n", "output:\n", y_cpu, "\ngrad weight:\n", y_cpu_gw)
        print("real (dpcpp):\n", "output:\n", y_dpcpp.cpu(), "\ngrad weight:\n", y_dpcpp_gw.cpu())

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=0)

    def test_conv2d_dilated_with_bias(self, dtype=torch.float):
        x_cpu = torch.randn([1, 64, 256, 256], dtype=dtype, device=cpu_device, requires_grad=True)
        grad_cpu = torch.full([1, 64, 254, 254], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True)
        conv_cpu = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=2, bias=True)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()

        x_dpcpp = x_cpu.to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        print("ref (cpu):\n", "output:\n", y_cpu, "\ngrad weight:\n", y_cpu_gw)
        print("real (dpcpp):\n", "output:\n", y_dpcpp.cpu(), "\ngrad weight:\n", y_dpcpp_gw.cpu())

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=0)

    def test_conv3d(self, dtype=torch.float):
        x_cpu = torch.randn([2, 16, 10, 128, 128], dtype=dtype, device=cpu_device, requires_grad=True)
        grad_cpu = torch.full([2, 32, 10, 128, 128], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True)
        conv_cpu = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()

        x_dpcpp = x_cpu.to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        print("ref (cpu):\n", "output:\n", y_cpu, "\ngrad weight:\n", y_cpu_gw)
        print("real (dpcpp):\n", "output:\n", y_dpcpp.cpu(), "\ngrad weight:\n", y_dpcpp_gw.cpu())

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=0)

    def test_conv3d_channels_last(self, dtype=torch.float):
        x_cpu = torch.randn([2, 16, 10, 128, 128], dtype=dtype, device=cpu_device, requires_grad=True)
        conv_cpu = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        y_cpu = conv_cpu(x_cpu)

        x_dpcpp = x_cpu.to(dpcpp_device).to(memory_format=torch.channels_last_3d)
        conv_dpcpp = conv_cpu.to(dpcpp_device).to(memory_format=torch.channels_last_3d)
        y_dpcpp = conv_dpcpp(x_dpcpp)

        print("ref (cpu):\n", "output:\n", y_cpu)
        print("real (dpcpp):\n", "output:\n", y_dpcpp.cpu())

        self.assertEqual(y_cpu, y_dpcpp.cpu())

    def test_conv3d_with_bias(self, dtype=torch.float):
        x_cpu = torch.randn([2, 16, 10, 128, 128], dtype=dtype, device=cpu_device, requires_grad=True)
        grad_cpu = torch.full([2, 32, 10, 128, 128], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True)
        conv_cpu = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()

        x_dpcpp = x_cpu.to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        print("ref (cpu):\n", "output:\n", y_cpu, "\ngrad weight:\n", y_cpu_gw)
        print("real (dpcpp):\n", "output:\n", y_dpcpp.cpu(), "\ngrad weight:\n", y_dpcpp_gw.cpu())

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=0)

    def test_conv3d_dilated(self, dtype=torch.float):
        x_cpu = torch.randn([2, 16, 10, 128, 128], dtype=dtype, device=cpu_device, requires_grad=True)
        grad_cpu = torch.full([2, 32, 6, 124, 124], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True)
        conv_cpu = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, dilation=3, bias=False)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()

        x_dpcpp = x_cpu.to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        print("ref (cpu):\n", "output:\n", y_cpu, "\ngrad weight:\n", y_cpu_gw)
        print("real (dpcpp):\n", "output:\n", y_dpcpp.cpu(), "\ngrad weight:\n", y_dpcpp_gw.cpu())

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=0)

    def test_conv3d_dilated_with_bias(self, dtype=torch.float):
        x_cpu = torch.randn([2, 16, 10, 128, 128], dtype=dtype, device=cpu_device, requires_grad=True)
        grad_cpu = torch.full([2, 32, 6, 124, 124], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True)
        conv_cpu = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, dilation=3, bias=True)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()

        x_dpcpp = x_cpu.to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        print("ref (cpu):\n", "output:\n", y_cpu, "\ngrad weight:\n", y_cpu_gw)
        print("real (dpcpp):\n", "output:\n", y_dpcpp.cpu(), "\ngrad weight:\n", y_dpcpp_gw.cpu())

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=0)

    def test_conv_with_nosquare_kernel_size(self, dtype=torch.float):
        x_cpu = torch.randn([20, 16, 50, 100], dtype=dtype, device=cpu_device, requires_grad=True)
        grad_cpu = torch.full([20, 33, 26, 100], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True)
        conv_cpu = nn.Conv2d(16, 33, kernel_size=(3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1), bias=True)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()

        x_dpcpp = x_cpu.to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        print("ref (cpu):\n", "output:\n", y_cpu, "\ngrad weight:\n", y_cpu_gw)
        print("real (dpcpp):\n", "output:\n", y_dpcpp.cpu(), "\ngrad weight:\n", y_dpcpp_gw.cpu())

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=0)

    def test_primitive_cache(self, dtype=torch.float):
        x_cpu = torch.randn([1, 2, 3, 3], dtype=dtype, device=cpu_device, requires_grad=True)
        conv1_cpu = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
        conv2_cpu = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
        conv3_cpu = nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1, bias=False)
        conv4_cpu = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        conv5_cpu = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        conv6_cpu = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1, bias=False)
        y_cpu = conv6_cpu(conv5_cpu(conv4_cpu(conv3_cpu(conv2_cpu(conv1_cpu(x_cpu))))))

        conv1 = conv1_cpu.to("xpu")
        conv2 = conv2_cpu.to("xpu")
        conv3 = conv3_cpu.to("xpu")
        conv4 = conv4_cpu.to("xpu")
        conv5 = conv5_cpu.to("xpu")
        conv6 = conv6_cpu.to("xpu")
        x = x_cpu.to("xpu")
        y = conv6(conv5(conv4(conv3(conv2(conv1(x))))))

        print("ref: ", y_cpu)
        print("real: ", y.cpu())

        self.assertEqual(y_cpu, y.cpu())

    def test_group_conv_fwd(self, dtype=torch.float):
        conv = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False, groups=2).to(dpcpp_device)
        x = torch.randn([1, 256, 3, 3], dtype=torch.float, device=cpu_device, requires_grad=True).to(dpcpp_device)
        real = conv(x)

        conv = conv.cpu()
        x = x.cpu()
        ref = conv(x)

        print("real: ", real.cpu())
        print("ref: ", ref)

        self.assertEqual(real.cpu(), ref)

    def test_group_conv_channels_last_fwd(self, dtype=torch.float):
        conv = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False, groups=2).to(cpu_device)
        x = torch.randn([1, 256, 3, 3], dtype=torch.float, device=cpu_device, requires_grad=True).to(cpu_device)
        ref = conv(x)

        conv_xpu = conv.to(dpcpp_device).to(memory_format=torch.channels_last)
        x_xpu = x.to(dpcpp_device).to(memory_format=torch.channels_last)
        real = conv_xpu(x_xpu)

        print("real: ", real.cpu())
        print("ref: ", ref)

        self.assertEqual(real.cpu(), ref)

    @pytest.mark.skipif(not torch.xpu.has_channels_last_1d() or torch.xpu.using_layout_opt(), reason="doesn't enable channels last 1d or channels last does not support onednn block format")
    def test_channels_last_1d_fwd(self, dtype=torch.float):
        shapes = [(2, 2, 3), (4, 4, 4), (4, 4, 1), (4, 1, 4),
                  (4, 1, 1), (1, 4, 4), (1, 4, 1)]
        for shape in shapes:
            print("\n====== test shape: ", shape, "======")
            N, C, H, W = shape[0], shape[1], 1, shape[2]
            x = torch.ones(N, C, H, W, dtype=torch.float)
            conv_ic, conv_oc, conv_ks = C, 5, 3
            w = torch.ones(conv_oc, conv_ic, conv_ks, conv_ks, dtype=torch.float)
            # cpu doesn't support channels last 1d, so we use conv2d to simulate conv1d here
            conv = torch.nn.Conv2d(conv_ic, conv_oc, kernel_size=conv_ks, stride=1, padding=1, bias=False)
            conv.weight.data = w
            conv = conv.to(memory_format=torch.channels_last)
            ref = conv(x)
            # convert 4D tensor to 3D tensor here to make comparison below happy
            ref = ref.view(ref.shape[0], ref.shape[1], ref.shape[3])

            N, C, L = shape[0], shape[1], shape[2]
            x = torch.ones(N, C, L, dtype=torch.float)
            conv_ic, conv_oc, conv_ks = C, 5, 3
            w = torch.ones(conv_oc, conv_ic, conv_ks, dtype=torch.float)
            conv = torch.nn.Conv1d(conv_ic, conv_oc, kernel_size=conv_ks, stride=1, padding=1, bias=False)
            conv.weight.data = w
            conv = conv.to(memory_format=torch.channels_last_1d)

            for test_weight_pollution in [False, True]:
                print("\n---test_weight_pollution: ", test_weight_pollution)
                if not test_weight_pollution:
                    x = x.to(memory_format=torch.channels_last_1d).to("xpu")
                else:
                    x = x.to(memory_format=torch.contiguous_format).to("xpu")
                w = w.to("xpu")
                conv.weight.data = w
                conv = conv.to(memory_format=torch.channels_last_1d)
                print("input.is_contiguous(): ", x.is_contiguous())
                print("input.is_contiguous(memory_format=torch.channels_last_1d): ",
                      x.is_contiguous(memory_format=torch.channels_last_1d))
                print("conv.weight.data.is_contiguous(): ", conv.weight.data.is_contiguous())
                print("conv.weight.data.is_contiguous(memory_format=torch.channels_last_1d): ",
                      conv.weight.data.is_contiguous(memory_format=torch.channels_last_1d))
                real = conv(x)

                print(real.shape)
                print(real.stride())
                print("real is CF: ", real.is_contiguous())
                print("real is CL: ", real.is_contiguous(memory_format=torch.channels_last_1d))
                if 1 == real.shape[1] or (1 == real.shape[2]) or \
                   (1 == real.shape[1] and 1 == real.shape[2]):
                    self.assertEqual(real.is_contiguous(), True)
                    self.assertEqual(real.is_contiguous(memory_format=torch.channels_last_1d), True)
                else:
                    self.assertEqual(real.is_contiguous(), False)
                    self.assertEqual(real.is_contiguous(memory_format=torch.channels_last_1d), True)
                print(real.contiguous().cpu())

                self.assertEqual(real.contiguous().cpu(), ref)

    @pytest.mark.skipif(not torch.xpu.has_channels_last_1d() or torch.xpu.using_layout_opt(), reason="doesn't enable channels last 1d or channels last does not support onednn block format")
    def test_channels_last_1d_bwd(self, dtype=torch.float):
        shapes = [(1, 7, 15000), (2, 2, 3), (4, 4, 4), (4, 4, 1), (4, 1, 4),
                  (4, 1, 1), (1, 4, 4), (1, 4, 1)]
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            N, C, H, W = shape[0], shape[1], 1, shape[2]
            x_cpu = torch.randn([N, C, H, W], dtype=dtype, device=cpu_device, requires_grad=True)

            for test_weight_pollution in [False, True]:
                x_cpu = x_cpu.to(memory_format=torch.channels_last)
                conv_ic, conv_oc, conv_ks = C, 5, 1
                grad_cpu = torch.full([N, conv_oc, H, W], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True)
                w = torch.ones(conv_oc, conv_ic, conv_ks, conv_ks, dtype=torch.float)
                # cpu doesn't support channels last 1d, so we use conv2d to simulate conv1d here
                conv_cpu = nn.Conv2d(conv_ic, conv_oc, kernel_size=conv_ks, stride=1, bias=True)
                conv_cpu.weight.data = w
                conv_cpu.bias.data.fill_(0.01)
                y_cpu = conv_cpu(x_cpu)
                y_cpu.backward(grad_cpu)
                y_cpu_gw = conv_cpu.weight.grad.detach().clone()
                # convert 4D tensor to 3D tensor here to make comparison below happy
                y_cpu = y_cpu.view(y_cpu.shape[0], y_cpu.shape[1], y_cpu.shape[3])
                y_cpu_gw = y_cpu_gw.view(y_cpu_gw.shape[0], y_cpu_gw.shape[1], y_cpu_gw.shape[3])
                conv_cpu.zero_grad()

                x_dpcpp = x_cpu.view(x_cpu.shape[0], x_cpu.shape[1], x_cpu.shape[3])
                x_dpcpp = x_dpcpp.to(dpcpp_device).requires_grad_()
                x_dpcpp = x_dpcpp.to(memory_format=torch.channels_last_1d)
                print("\n---test_weight_pollution: ", test_weight_pollution)
                grad_dpcpp = grad_cpu.view(grad_cpu.shape[0], grad_cpu.shape[1], grad_cpu.shape[3])
                if not test_weight_pollution:
                    grad_dpcpp = grad_dpcpp.to(dpcpp_device).to(memory_format=torch.channels_last_1d)
                else:
                    grad_dpcpp = grad_dpcpp.to(dpcpp_device).to(memory_format=torch.contiguous_format)

                w = torch.ones(conv_oc, conv_ic, conv_ks, dtype=torch.float)
                conv_dpcpp = nn.Conv1d(conv_ic, conv_oc, kernel_size=conv_ks, stride=1, bias=True)
                conv_dpcpp.weight.data = w
                conv_dpcpp.bias.data.fill_(0.01)
                conv_dpcpp = conv_dpcpp.to(dpcpp_device).to(memory_format=torch.channels_last_1d)
                y_dpcpp = conv_dpcpp(x_dpcpp)
                print("y_dpcpp.shape: ", y_dpcpp.shape)
                print("y_dpcpp.is_contiguous(): ", y_dpcpp.is_contiguous())
                print("y_dpcpp.is_contiguous(memory_format=torch.channels_last_1d): ",
                      y_dpcpp.is_contiguous(memory_format=torch.channels_last_1d))
                print("grad_dpcpp.shape: ", grad_dpcpp.shape)
                print("grad_dpcpp.is_contiguous(): ", grad_dpcpp.is_contiguous())
                print("grad_dpcpp.is_contiguous(memory_format=torch.channels_last_1d): ",
                      grad_dpcpp.is_contiguous(memory_format=torch.channels_last_1d))
                print("x_dpcpp.shape: ", x_dpcpp.shape)
                print("x_dpcpp.is_contiguous(): ", x_dpcpp.is_contiguous())
                print("x_dpcpp.is_contiguous(memory_format=torch.channels_last_1d): ",
                      x_dpcpp.is_contiguous(memory_format=torch.channels_last_1d))
                y_dpcpp.backward(grad_dpcpp)
                y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()
                print("y_dpcpp_gw.shape: ", y_dpcpp_gw.shape)
                print("y_dpcpp_gw.is_contiguous(): ", y_dpcpp_gw.is_contiguous())
                print("y_dpcpp_gw.is_contiguous(memory_format=torch.channels_last_1d): ",
                      y_dpcpp_gw.is_contiguous(memory_format=torch.channels_last_1d))
                conv_dpcpp.zero_grad()

                self.assertEqual(y_cpu, y_dpcpp.cpu(), atol=5 * 1e-5, rtol=0)
                self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=0)

    @pytest.mark.skipif(torch.xpu.using_layout_opt(), reason="channels last does not support onednn block format")
    def test_channels_last_fwd(self, dtype=torch.float):
        shapes = [(2, 2, 3, 3), (4, 4, 4, 4), (4, 4, 1, 1), (4, 1, 4, 4),
                  (4, 1, 4, 1), (4, 1, 1, 4), (1, 4, 1, 4), (1, 4, 4, 1), (4, 1, 1, 1)]
        for shape in shapes:
            print("\n====== test shape: ", shape, "======")
            N, C, H, W = shape[0], shape[1], shape[2], shape[3]
            x = torch.ones(N, C, H, W, dtype=torch.float)
            conv_ic, conv_oc, conv_ks = C, 5, 3
            w = torch.ones(conv_oc, conv_ic, conv_ks, conv_ks, dtype=torch.float)
            conv = torch.nn.Conv2d(conv_ic, conv_oc, kernel_size=conv_ks, stride=1, padding=1, bias=False)
            conv.weight.data = w
            conv = conv.to(memory_format=torch.channels_last)
            ref = conv(x)

            for test_weight_pollution in [False, True]:
                print("\n---test_weight_pollution: ", test_weight_pollution)
                if not test_weight_pollution:
                    x = x.to(memory_format=torch.channels_last).to("xpu")
                else:
                    x = x.to(memory_format=torch.contiguous_format).to("xpu")
                w = w.to("xpu")
                conv.weight.data = w
                conv = conv.to(memory_format=torch.channels_last)
                print("input.is_contiguous(): ", x.is_contiguous())
                print("input.is_contiguous(memory_format=torch.channels_last): ",
                      x.is_contiguous(memory_format=torch.channels_last))
                print("conv.weight.data.is_contiguous(): ", conv.weight.data.is_contiguous())
                print("conv.weight.data.is_contiguous(memory_format=torch.channels_last): ",
                      conv.weight.data.is_contiguous(memory_format=torch.channels_last))
                real = conv(x)

                print(real.shape)
                print(real.stride())
                print("real is CF: ", real.is_contiguous())
                print("real is CL: ", real.is_contiguous(memory_format=torch.channels_last))
                if 1 == real.shape[1] or (1 == real.shape[2] and 1 == real.shape[3]) or \
                   (1 == real.shape[1] and 1 == real.shape[2] and 1 == real.shape[3]):
                    self.assertEqual(real.is_contiguous(), True)
                    self.assertEqual(real.is_contiguous(memory_format=torch.channels_last), True)
                else:
                    self.assertEqual(real.is_contiguous(), False)
                    self.assertEqual(real.is_contiguous(memory_format=torch.channels_last), True)
                print(real.contiguous().cpu())

                self.assertEqual(real.contiguous().cpu(), ref)

    @pytest.mark.skipif(torch.xpu.using_layout_opt(), reason="channels last does not support onednn block format")
    def test_channels_last_bwd(self, dtype=torch.float):
        shapes = [(1, 7, 1, 15000), (2, 2, 3, 3), (4, 4, 4, 4), (4, 4, 1, 1), (4, 1, 4, 4),
                  (4, 1, 4, 1), (4, 1, 1, 4), (1, 4, 1, 4), (1, 4, 4, 1), (4, 1, 1, 1)]
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            N, C, H, W = shape[0], shape[1], shape[2], shape[3]
            x_cpu = torch.randn([N, C, H, W], dtype=dtype, device=cpu_device, requires_grad=True)

            for test_weight_pollution in [False, True]:
                x_cpu = x_cpu.to(memory_format=torch.channels_last)
                grad_cpu = torch.full([N, 64, H, W], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True)
                conv_cpu = nn.Conv2d(C, 64, kernel_size=1, stride=1, bias=True)
                y_cpu = conv_cpu(x_cpu)
                y_cpu.backward(grad_cpu)
                y_cpu_gw = conv_cpu.weight.grad.detach().clone()
                conv_cpu.zero_grad()

                x_dpcpp = x_cpu.to(dpcpp_device).requires_grad_()
                x_dpcpp = x_dpcpp.to(memory_format=torch.channels_last)
                print("\n---test_weight_pollution: ", test_weight_pollution)
                if not test_weight_pollution:
                    grad_dpcpp = grad_cpu.to(dpcpp_device).to(memory_format=torch.channels_last)
                else:
                    grad_dpcpp = grad_cpu.to(dpcpp_device).to(memory_format=torch.contiguous_format)

                conv_dpcpp = conv_cpu.to(dpcpp_device).to(memory_format=torch.channels_last)
                y_dpcpp = conv_dpcpp(x_dpcpp)
                print("y_dpcpp.shape: ", y_dpcpp.shape)
                print("y_dpcpp.is_contiguous(): ", y_dpcpp.is_contiguous())
                print("y_dpcpp.is_contiguous(memory_format=torch.channels_last): ",
                      y_dpcpp.is_contiguous(memory_format=torch.channels_last))
                print("grad_dpcpp.shape: ", grad_dpcpp.shape)
                print("grad_dpcpp.is_contiguous(): ", grad_dpcpp.is_contiguous())
                print("grad_dpcpp.is_contiguous(memory_format=torch.channels_last): ",
                      grad_dpcpp.is_contiguous(memory_format=torch.channels_last))
                print("x_dpcpp.shape: ", x_dpcpp.shape)
                print("x_dpcpp.is_contiguous(): ", x_dpcpp.is_contiguous())
                print("x_dpcpp.is_contiguous(memory_format=torch.channels_last): ",
                      x_dpcpp.is_contiguous(memory_format=torch.channels_last))
                y_dpcpp.backward(grad_dpcpp)
                y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()
                print("y_dpcpp_gw.shape: ", y_dpcpp_gw.shape)
                print("y_dpcpp_gw.is_contiguous(): ", y_dpcpp_gw.is_contiguous())
                print("y_dpcpp_gw.is_contiguous(memory_format=torch.channels_last): ",
                      y_dpcpp_gw.is_contiguous(memory_format=torch.channels_last))
                conv_dpcpp.zero_grad()

                self.assertEqual(y_cpu, y_dpcpp.cpu(), atol=5 * 1e-5, rtol=0)
                self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=0)

    def test_plain_format_bwd(self, dtype=torch.float):
        x_cpu = torch.randn([1, 7, 1, 15000], dtype=dtype, device=cpu_device, requires_grad=True)
        grad_cpu = torch.full([1, 64, 1, 15000], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True)
        conv_cpu = nn.Conv2d(7, 64, kernel_size=1, stride=1, bias=True)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()
        conv_cpu.zero_grad()

        x_dpcpp = x_cpu.to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()
        conv_cpu.zero_grad()

        self.assertEqual(y_cpu, y_dpcpp.cpu(), atol=5 * 1e-5, rtol=0)
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=0)

    def test_conv2d_corner_case(self, dtype=torch.float):
        conv_cpu = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        x_cpu = torch.randn([1, 3, 300, 300], dtype=dtype, device=cpu_device)
        x_cpu = torch.as_strided(x_cpu, x_cpu.shape, (3, 1, 900, 3))
        y_cpu = conv_cpu(x_cpu)

        x_dpcpp = x_cpu.to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())

    def test_conv2d_bia_bf16_input_fp32_bia(self, dtype=torch.float):
        x_cpu = torch.randn([1, 64, 256, 256], dtype=dtype, device=cpu_device, requires_grad=True)
        grad_cpu = torch.full([1, 64, 256, 256], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True)
        conv_cpu = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()

        dtype_dpcpp = torch.bfloat16
        x_dpcpp = x_cpu.to(dpcpp_device).requires_grad_().to(dtype_dpcpp)
        grad_dpcpp = grad_cpu.to(dpcpp_device).to(dtype_dpcpp)
        conv_dpcpp = conv_cpu.to(dpcpp_device).to(dtype_dpcpp)
        conv_dpcpp.bias.data = conv_dpcpp.bias.data.to(torch.float)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.to(torch.float).cpu(), rtol=10e-4, atol=10e-2)
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.to(torch.float).cpu(), rtol=10e-4, atol=10e-2)

    def test_conv2d_bia_bf16_input_bf16_bia(self, dtype=torch.float):
        x_cpu = torch.randn([1, 64, 256, 256], dtype=dtype, device=cpu_device, requires_grad=True)
        grad_cpu = torch.full([1, 64, 256, 256], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True)
        conv_cpu = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()

        dtype_dpcpp = torch.bfloat16
        x_dpcpp = x_cpu.to(dpcpp_device).requires_grad_().to(dtype_dpcpp)
        grad_dpcpp = grad_cpu.to(dpcpp_device).to(dtype_dpcpp)
        conv_dpcpp = conv_cpu.to(dpcpp_device).to(dtype_dpcpp)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.to(torch.float).cpu(), rtol=10e-4, atol=10e-2)
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.to(torch.float).cpu(), rtol=10e-4, atol=10e-2)
