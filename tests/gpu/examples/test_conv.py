import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase
import torch_ipex
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
    self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5*1e-5, rtol=0)

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
    self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5*1e-5, rtol=0)

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
    self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5*1e-5, rtol=0)

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
    self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5*1e-5, rtol=0)

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
    self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5*1e-5, rtol=0)

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
    self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5*1e-5, rtol=0)

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
    self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5*1e-5, rtol=0)

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
    self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5*1e-5, rtol=0)

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
    self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5*1e-5, rtol=0)

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

  def test_channels_last_simple_fwd(self, dtype=torch.float):
    x = torch.ones(2, 2, 3, 3, dtype=torch.float)
    w = torch.ones(2, 2, 3, 3, dtype=torch.float)
    conv = torch.nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
    conv.weight.data = w
    ref = conv(x)

    x = x.contiguous(memory_format=torch.channels_last).to("xpu")
    w = w.to("xpu").contiguous(memory_format=torch.channels_last)
    conv.weight.data = w
    real = conv(x)

    print(real.shape)
    print(real.stride())
    print(real.contiguous().cpu())
    print(ref.shape)
    print(ref.stride())
    print(ref)

    self.assertEqual(real.contiguous().cpu(), ref)
