import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

import ipex

import numpy

cpu_device = torch.device("cpu")
sycl_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_upsamle_last_channel(self, dtype=torch.float):
        conv = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        x_cpu = torch.randn(2, 3, 4, 5)
        grad_i = torch.randn([2, 3, 8, 10], device=cpu_device)

        x_cpu = Variable(x_cpu, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        #  y_cpu = upsample(conv(x_cpu))
        y_cpu = upsample(x_cpu)
        y_cpu.backward(grad_cpu)
        #  print("cpu result ", y_cpu)
        print("cpu grad result ", x_cpu.grad)

        x_xpu = x_cpu.to("xpu").to(memory_format=torch.channels_last)
        grad_xpu = grad_cpu.to("xpu").to(memory_format=torch.channels_last)
        #  x_xpu = x_cpu.to("xpu")
        #  grad_xpu = grad_cpu.to("xpu")
        x_xpu = Variable(x_xpu, requires_grad=True)
        grad_xpu = Variable(grad_xpu, requires_grad=True)
        conv.to("xpu")
        #  y_xpu = upsample(conv(x_xpu))
        y_xpu = upsample(x_xpu)
        y_xpu.backward(grad_xpu)
        #  print("xpu result ", y_xpu.cpu())
        print("xpu grad result ", x_xpu.grad.cpu())
        self.assertEqual(y_cpu, y_xpu.cpu())
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())

    def test_upsamle(self, dtype=torch.float):
        x_cpu = torch.tensor([[[[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]], [[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]]], [
                             [[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]], [[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]]]], dtype=torch.float32, device=cpu_device)
        # x_sycl = torch.tensor([[[[1,2,3,4,5],[4,5,6,7,8]],[[1,2,3,4,5],[4,5,6,7,8]]],[[[1,2,3,4,5],[4,5,6,7,8]],[[1,2,3,4,5],[4,5,6,7,8]]]], dtype=torch.float32, device = sycl_device)
        x_sycl = x_cpu.to("xpu")

        print("cpu result", torch.nn.functional.upsample_nearest(x_cpu, [2, 5]))
        print("sycl result", torch.nn.functional.upsample_nearest(x_sycl, [2, 5]).cpu())

        print("cpu result", torch.nn.functional.upsample_nearest(x_cpu, [4, 10]))
        print("sycl result", torch.nn.functional.upsample_nearest(x_sycl, [4, 10]).cpu())

        print("cpu result", torch.nn.functional.upsample_nearest(x_cpu, [3, 8]))
        print("sycl result", torch.nn.functional.upsample_nearest(x_sycl, [3, 8]).cpu())

        print("cpu result", torch.nn.functional.upsample_nearest(x_cpu, [1, 3]))
        print("sycl result", torch.nn.functional.upsample_nearest(x_sycl, [1, 3]).cpu())
        self.assertEqual(x_cpu, x_sycl.cpu())
        self.assertEqual(torch.nn.functional.upsample_nearest(
            x_cpu, [2, 5]), torch.nn.functional.upsample_nearest(x_sycl, [2, 5]).cpu())
        self.assertEqual(torch.nn.functional.upsample_nearest(
            x_cpu, [4, 10]), torch.nn.functional.upsample_nearest(x_sycl, [4, 10]).cpu())
        # self.assertEqual(torch.nn.functional.upsample_nearest(x_cpu,[3,8]), torch.nn.functional.upsample_nearest(x_sycl,[3,8]).cpu())
        # self.assertEqual(torch.nn.functional.upsample_nearest(x_cpu,[1,3]), torch.nn.functional.upsample_nearest(x_sycl,[1,3]).cpu())

        x_cpu = torch.arange(8).view(1, 2, 2, 2).type(torch.FloatTensor)
        x_sycl = x_cpu.to("xpu")

        expected_out = torch.Tensor(
            [[[[-0.31641, 0.01562, 0.56250, 0.89453],
               [0.34766, 0.67969, 1.22656, 1.55859],
               [1.44141, 1.77344, 2.32031, 2.65234],
               [2.10547, 2.43750, 2.98438, 3.31641]],

              [[3.68359, 4.01562, 4.56250, 4.89453],
               [4.34766, 4.67969, 5.22656, 5.55859],
                [5.44141, 5.77344, 6.32031, 6.65234],
                [6.10547, 6.43750, 6.98438, 7.31641]]]])

        y_cpu = nn.functional.interpolate(x_cpu, scale_factor=2, mode='bicubic', align_corners=False)
        y_sycl = nn.functional.interpolate(x_sycl, scale_factor=2, mode='bicubic', align_corners=False).cpu()

        print("expected result", expected_out)
        print("cpu result", y_cpu)
        print("sycl result", y_sycl)
        self.assertEqual(x_cpu, x_sycl.cpu())
        self.assertEqual(y_cpu, y_sycl.cpu())

        x_cpu = torch.tensor([[[[1, 2, 3, 4], [4, 5, 6, 7], [1, 2, 3, 4], [4, 5, 6, 7]], [[1, 2, 3, 4], [
                             4, 5, 6, 7], [1, 2, 3, 4], [4, 5, 6, 7]]]], dtype=torch.float32, device=cpu_device)
        x_sycl = x_cpu.to("xpu")
        x_cpu.requires_grad_(True)
        x_sycl.requires_grad_(True)
        y_cpu = nn.functional.interpolate(x_cpu, scale_factor=1, mode='bicubic', align_corners=False)
        y_sycl = nn.functional.interpolate(x_sycl, scale_factor=1, mode='bicubic', align_corners=False)

        y_cpu.backward(x_cpu)
        y_sycl.backward(x_sycl)

        print("float type cpu bwd result", x_cpu.grad)
        print("float type sycl bwd result", x_sycl.grad.cpu())
        self.assertEqual(y_cpu, y_sycl.cpu())
        self.assertEqual(x_cpu.grad, x_sycl.grad.cpu())

        '''only available in dpcpp build
      x_cpu = torch.tensor([[[[1,2,3,4],[4,5,6,7],[1,2,3,4],[4,5,6,7]],[[1,2,3,4],[4,5,6,7],[1,2,3,4],[4,5,6,7]]]], dtype=torch.double, device = cpu_device)
      x_sycl=x_cpu.to("xpu")
      x_cpu.requires_grad_(True)
      x_sycl.requires_grad_(True)
      y_cpu = nn.functional.interpolate(x_cpu, scale_factor=1, mode='bicubic', align_corners=False)
      y_sycl = nn.functional.interpolate(x_sycl, scale_factor=1, mode='bicubic', align_corners=False)

      y_cpu.backward(x_cpu)
      y_sycl.backward(x_sycl)
      '''

        print("double type cpu bwd result", x_cpu.grad)
        print("double type sycl bwd result", x_sycl.grad.cpu())

        y_cpu = nn.functional.interpolate(x_cpu, scale_factor=2, mode='nearest')
        y_sycl = nn.functional.interpolate(x_sycl, scale_factor=2, mode='nearest').cpu()

        print("cpu result", y_cpu)
        print("sycl result", y_sycl)
        #  self.assertEqual(x_cpu, x_sycl.cpu())
        self.assertEqual(y_cpu, y_sycl.cpu())


'''
y_cpu = nn.functional.interpolate(x_cpu, scale_factor=2, mode='bilinear', align_corners=False)
y_sycl = nn.functional.interpolate(x_sycl, scale_factor=2, mode='bilinear', align_corners=False).cpu()

print("cpu result", y_cpu)
print("sycl result", y_sycl)

x_cpu = torch.arange(8).view(2, 2, 2).type(torch.FloatTensor)
x_sycl = x_cpu.to("xpu")

y_cpu = nn.functional.interpolate(x_cpu, scale_factor=2, mode='linear', align_corners=False)
y_sycl = nn.functional.interpolate(x_sycl, scale_factor=2, mode='linear', align_corners=False).cpu()

print("cpu result", y_cpu)
print("sycl result", y_sycl)

x_cpu = torch.arange(8).view(1, 1, 2, 2, 2).type(torch.FloatTensor)
x_sycl = x_cpu.to("xpu")

y_cpu = nn.functional.interpolate(x_cpu, scale_factor=2, mode='trilinear', align_corners=False)
y_sycl = nn.functional.interpolate(x_sycl, scale_factor=2, mode='trilinear', align_corners=False).cpu()

print("cpu result", y_cpu)
print("sycl result", y_sycl)

y_cpu = nn.functional.interpolate(x_cpu, scale_factor=2, mode='area')
y_sycl = nn.functional.interpolate(x_sycl, scale_factor=2, mode='area').cpu()

print("cpu result", y_cpu)
print("sycl result", y_sycl)
'''
