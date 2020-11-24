import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase
import torch_ipex
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")

class TestNNMethod(TestCase):
  def test_conv2d(self, dtype=torch.float):
    x_cpu = torch.randn([1, 64, 256, 256], dtype=dtype, device=cpu_device, requires_grad=True)
    grad_cpu = torch.ones([1, 64, 256, 256], dtype=dtype, device=cpu_device, requires_grad=True)
    conv_cpu = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
    y_cpu = conv_cpu(x_cpu)
    y_cpu.backward(grad_cpu)
    
    x_dpcpp = x_cpu.to(dpcpp_device)
    grad_dpcpp = grad_cpu.to(dpcpp_device)
    conv_dpcpp = conv_cpu.to(dpcpp_device)
    y_dpcpp = conv_dpcpp(x_dpcpp)
    y_dpcpp.backward(grad_dpcpp)

    if not torch_ipex._double_kernel_disabled():
      print("ref: ")
      print(y_cpu)
      print("ref backward: ")
      print(x_cpu)

      print("real: ")
      print(y_dpcpp.cpu())
      print("real backward: ")
      print(x_dpcpp.cpu())

    self.assertEqual(x_cpu,       x_dpcpp.cpu())
    self.assertEqual(grad_cpu, grad_dpcpp.cpu())
    self.assertEqual(y_cpu,       y_dpcpp.cpu())

  def test_conv2d_with_bias(self, dtype=torch.float):
    x_cpu = torch.randn([1, 64, 256, 256], dtype=dtype, device=cpu_device, requires_grad=True)
    grad_cpu = torch.ones([1, 64, 256, 256], dtype=dtype, device=cpu_device, requires_grad=True)
    conv_cpu = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
    y_cpu = conv_cpu(x_cpu)
    y_cpu.backward(grad_cpu)
    
    x_dpcpp = x_cpu.to(dpcpp_device)
    grad_dpcpp = grad_cpu.to(dpcpp_device)
    conv_dpcpp = conv_cpu.to(dpcpp_device)
    y_dpcpp = conv_dpcpp(x_dpcpp)
    y_dpcpp.backward(grad_dpcpp)

    if not torch_ipex._double_kernel_disabled():
      print("ref: ")
      print(y_cpu)
      print("ref backward: ")
      print(x_cpu)

      print("real: ")
      print(y_dpcpp.cpu())
      print("real backward: ")
      print(x_dpcpp.cpu())

    self.assertEqual(x_cpu,       x_dpcpp.cpu())
    self.assertEqual(grad_cpu, grad_dpcpp.cpu())
    self.assertEqual(y_cpu,       y_dpcpp.cpu())

  def test_conv2d_dilated(self, dtype=torch.float):
    x_cpu = torch.randn([1, 64, 256, 256], dtype=dtype, device=cpu_device, requires_grad=True)
    grad_cpu = torch.ones([1, 64, 254, 254], dtype=dtype, device=cpu_device, requires_grad=True)
    conv_cpu = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=2, bias=False)
    y_cpu = conv_cpu(x_cpu)
    y_cpu.backward(grad_cpu)
    
    x_dpcpp = x_cpu.to(dpcpp_device)
    grad_dpcpp = grad_cpu.to(dpcpp_device)
    conv_dpcpp = conv_cpu.to(dpcpp_device)
    y_dpcpp = conv_dpcpp(x_dpcpp)
    y_dpcpp.backward(grad_dpcpp)

    if not torch_ipex._double_kernel_disabled():
      print("ref: ")
      print(y_cpu)
      print("ref backward: ")
      print(x_cpu)

      print("real: ")
      print(y_dpcpp.cpu())
      print("real backward: ")
      print(x_dpcpp.cpu())

    self.assertEqual(x_cpu,       x_dpcpp.cpu())
    self.assertEqual(grad_cpu, grad_dpcpp.cpu())
    self.assertEqual(y_cpu,       y_dpcpp.cpu())

  def test_conv2d_dilated_with_bias(self, dtype=torch.float):
    x_cpu = torch.randn([1, 64, 256, 256], dtype=dtype, device=cpu_device, requires_grad=True)
    grad_cpu = torch.ones([1, 64, 254, 254], dtype=dtype, device=cpu_device, requires_grad=True)
    conv_cpu = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=2, bias=True)
    y_cpu = conv_cpu(x_cpu)
    y_cpu.backward(grad_cpu)
    
    x_dpcpp = x_cpu.to(dpcpp_device)
    grad_dpcpp = grad_cpu.to(dpcpp_device)
    conv_dpcpp = conv_cpu.to(dpcpp_device)
    y_dpcpp = conv_dpcpp(x_dpcpp)
    y_dpcpp.backward(grad_dpcpp)

    if not torch_ipex._double_kernel_disabled():
      print("ref: ")
      print(y_cpu)
      print("ref backward: ")
      print(x_cpu)

      print("real: ")
      print(y_dpcpp.cpu())
      print("real backward: ")
      print(x_dpcpp.cpu())

    self.assertEqual(x_cpu,       x_dpcpp.cpu())
    self.assertEqual(grad_cpu, grad_dpcpp.cpu())
    self.assertEqual(y_cpu,       y_dpcpp.cpu())

  def test_conv3d(self, dtype=torch.float):
    x_cpu = torch.randn([2, 16, 10, 128, 128], dtype=dtype, device=cpu_device, requires_grad=True)
    grad_cpu = torch.ones([2, 32, 10, 128, 128], dtype=dtype, device=cpu_device, requires_grad=True)
    conv_cpu = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
    y_cpu = conv_cpu(x_cpu)
    y_cpu.backward(grad_cpu)
    
    x_dpcpp = x_cpu.to(dpcpp_device)
    grad_dpcpp = grad_cpu.to(dpcpp_device)
    conv_dpcpp = conv_cpu.to(dpcpp_device)
    y_dpcpp = conv_dpcpp(x_dpcpp)
    y_dpcpp.backward(grad_dpcpp)

    if not torch_ipex._double_kernel_disabled():
      print("ref: ")
      print(y_cpu)
      print("ref backward: ")
      print(x_cpu)

      print("real: ")
      print(y_dpcpp.cpu())
      print("real backward: ")
      print(x_dpcpp.cpu())

    self.assertEqual(x_cpu,       x_dpcpp.cpu())
    self.assertEqual(grad_cpu, grad_dpcpp.cpu())
    self.assertEqual(y_cpu,       y_dpcpp.cpu())

  def test_conv3d_with_bias(self, dtype=torch.float):
    x_cpu = torch.randn([2, 16, 10, 128, 128], dtype=dtype, device=cpu_device, requires_grad=True)
    grad_cpu = torch.ones([2, 32, 10, 128, 128], dtype=dtype, device=cpu_device, requires_grad=True)
    conv_cpu = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)
    y_cpu = conv_cpu(x_cpu)
    y_cpu.backward(grad_cpu)
    
    x_dpcpp = x_cpu.to(dpcpp_device)
    grad_dpcpp = grad_cpu.to(dpcpp_device)
    conv_dpcpp = conv_cpu.to(dpcpp_device)
    y_dpcpp = conv_dpcpp(x_dpcpp)
    y_dpcpp.backward(grad_dpcpp)

    if not torch_ipex._double_kernel_disabled():
      print("ref: ")
      print(y_cpu)
      print("ref backward: ")
      print(x_cpu)

      print("real: ")
      print(y_dpcpp.cpu())
      print("real backward: ")
      print(x_dpcpp.cpu())

    self.assertEqual(x_cpu,       x_dpcpp.cpu())
    self.assertEqual(grad_cpu, grad_dpcpp.cpu())
    self.assertEqual(y_cpu,       y_dpcpp.cpu())

  def test_conv3d_dilated(self, dtype=torch.float):
    x_cpu = torch.randn([2, 16, 10, 128, 128], dtype=dtype, device=cpu_device, requires_grad=True)
    grad_cpu = torch.ones([2, 32, 6, 124, 124], dtype=dtype, device=cpu_device, requires_grad=True)
    conv_cpu = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, dilation=3, bias=False)
    y_cpu = conv_cpu(x_cpu)
    y_cpu.backward(grad_cpu)
    
    x_dpcpp = x_cpu.to(dpcpp_device)
    grad_dpcpp = grad_cpu.to(dpcpp_device)
    conv_dpcpp = conv_cpu.to(dpcpp_device)
    y_dpcpp = conv_dpcpp(x_dpcpp)
    y_dpcpp.backward(grad_dpcpp)

    if not torch_ipex._double_kernel_disabled():
      print("ref: ")
      print(y_cpu)
      print("ref backward: ")
      print(x_cpu)

      print("real: ")
      print(y_dpcpp.cpu())
      print("real backward: ")
      print(x_dpcpp.cpu())

    self.assertEqual(x_cpu,       x_dpcpp.cpu())
    self.assertEqual(grad_cpu, grad_dpcpp.cpu())
    self.assertEqual(y_cpu,       y_dpcpp.cpu())

  def test_conv3d_dilated_with_bias(self, dtype=torch.float):
    x_cpu = torch.randn([2, 16, 10, 128, 128], dtype=dtype, device=cpu_device, requires_grad=True)
    grad_cpu = torch.ones([2, 32, 6, 124, 124], dtype=dtype, device=cpu_device, requires_grad=True)
    conv_cpu = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, dilation=3, bias=True)
    y_cpu = conv_cpu(x_cpu)
    y_cpu.backward(grad_cpu)
    
    x_dpcpp = x_cpu.to(dpcpp_device)
    grad_dpcpp = grad_cpu.to(dpcpp_device)
    conv_dpcpp = conv_cpu.to(dpcpp_device)
    y_dpcpp = conv_dpcpp(x_dpcpp)
    y_dpcpp.backward(grad_dpcpp)

    if not torch_ipex._double_kernel_disabled():
      print("ref: ")
      print(y_cpu)
      print("ref backward: ")
      print(x_cpu)

      print("real: ")
      print(y_dpcpp.cpu())
      print("real backward: ")
      print(x_dpcpp.cpu())

    self.assertEqual(x_cpu,       x_dpcpp.cpu())
    self.assertEqual(grad_cpu, grad_dpcpp.cpu())
    self.assertEqual(y_cpu,       y_dpcpp.cpu())

  def test_conv_with_nosquare_kernel_size(self, dtype=torch.float):
    x_cpu = torch.randn([20, 16, 50, 100], dtype=dtype, device=cpu_device, requires_grad=True)
    grad_cpu = torch.ones([20, 33, 26, 100], dtype=dtype, device=cpu_device, requires_grad=True)
    conv_cpu = nn.Conv2d(16, 33, kernel_size=(3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1), bias=True)
    y_cpu = conv_cpu(x_cpu)
    y_cpu.backward(grad_cpu)
    
    x_dpcpp = x_cpu.to(dpcpp_device)
    grad_dpcpp = grad_cpu.to(dpcpp_device)
    conv_dpcpp = conv_cpu.to(dpcpp_device)
    y_dpcpp = conv_dpcpp(x_dpcpp)
    y_dpcpp.backward(grad_dpcpp)

    if not torch_ipex._double_kernel_disabled():
      print("ref: ")
      print(y_cpu)
      print("ref backward: ")
      print(x_cpu)

      print("real: ")
      print(y_dpcpp.cpu())
      print("real backward: ")
      print(x_dpcpp.cpu())

    self.assertEqual(x_cpu,       x_dpcpp.cpu())
    self.assertEqual(grad_cpu, grad_dpcpp.cpu())
    self.assertEqual(y_cpu,       y_dpcpp.cpu())


  def test_primitive_cache(self, dtype=torch.float):
    x_cpu = torch.randn([1, 2, 3, 3], dtype=dtype, device=cpu_device, requires_grad=True)
    conv1_cpu = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
    conv2_cpu = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
    conv3_cpu = nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1, bias=False)
    conv4_cpu = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
    conv5_cpu = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
    conv6_cpu = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1, bias=False)
    y_cpu = conv6_cpu(conv5_cpu(conv4_cpu(conv3_cpu(conv2_cpu(conv1_cpu(x_cpu))))))

    conv1 = conv1_cpu.to("dpcpp")
    conv2 = conv2_cpu.to("dpcpp")
    conv3 = conv3_cpu.to("dpcpp")
    conv4 = conv4_cpu.to("dpcpp")
    conv5 = conv5_cpu.to("dpcpp")
    conv6 = conv6_cpu.to("dpcpp")
    x = x_cpu.to("dpcpp")
    y = conv6(conv5(conv4(conv3(conv2(conv1(x))))))

    print("ref: ", y_cpu)
    print("real: ", y.cpu())

    self.assertEqual(y_cpu,       y.cpu())

