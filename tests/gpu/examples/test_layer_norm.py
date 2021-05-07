from __future__ import print_function
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase
import torch_ipex
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_layer_norm(self, dtype=torch.float):

        layer_norm = nn.LayerNorm([1, 3, 3])
        x_i = torch.randn([1, 1, 3, 3], device=cpu_device, dtype=dtype)
        grad_i = torch.randn([1, 1, 3, 3], device=cpu_device, dtype=dtype)

        x_i[0][0][0][0] = 0.5021
        x_i[0][0][0][1] = -0.9922
        x_i[0][0][0][2] = -0.7365
        x_i[0][0][1][0] = 0.0629
        x_i[0][0][1][1] = -2.0536
        x_i[0][0][1][2] = -0.9989
        x_i[0][0][2][0] = 0.4911
        x_i[0][0][2][1] = 0.9744
        x_i[0][0][2][2] = -1.9760

        grad_i[0][0][0][0] = 0.6259
        grad_i[0][0][0][1] = -0.3097
        grad_i[0][0][0][2] = -0.8985
        grad_i[0][0][1][0] = 0.0328
        grad_i[0][0][1][1] = 1.9637
        grad_i[0][0][1][2] = -1.7078
        grad_i[0][0][2][0] = 0.3252
        grad_i[0][0][2][1] = -0.2873
        grad_i[0][0][2][2] = -0.4864

        # torch.save(layer_norm, "./log/layer_norm.pt")
        # torch.save(x_i, "./log/layer_norm_x.pt")
        # torch.save(grad_i, "./log/layer_norm_grad.pt")

        x_dpcpp_i = x_i.to("xpu")
        grad_dpcpp_i = grad_i.to("xpu")

        x_cpu = Variable(x_i, requires_grad=True)
        y_cpu = layer_norm(x_cpu)

        y_cpu.backward(grad_i)

        print("x_cpu = ", x_cpu)
        print("layer_norm = ", layer_norm.weight.cpu())
        print("y_cpu = ", y_cpu)
        print("x_cpu.grad = ", x_cpu.grad)
        print("layer_norm.grad = ", layer_norm.weight.grad)
        #x_cpu.grad.detach()
        #x_cpu.grad.zero_()

        # layer_norm_dpcpp = torch.load("./log/layer_norm.pt").to(dpcpp_device)
        layer_norm_dpcpp = layer_norm.to(dpcpp_device)
        layer_norm.zero_grad()

        x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
        y_dpcpp = layer_norm_dpcpp(x_dpcpp)

        y_dpcpp.backward(grad_dpcpp_i)

        print("x_dpcpp = ", x_dpcpp.cpu())
        print("layer_norm_dpcpp = ", layer_norm_dpcpp.weight.cpu())
        print("y_dpcpp = ", y_dpcpp.cpu())
        print("x_dpcpp.grad = ", x_dpcpp.grad.cpu())
        print("layer_norm_dpcpp.grad = ", layer_norm_dpcpp.weight.grad.cpu())
        self.assertEqual(x_cpu, x_dpcpp.cpu())
        self.assertEqual(layer_norm.weight.cpu(),
                         layer_norm_dpcpp.weight.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())
        self.assertEqual(layer_norm.weight.grad,
                         layer_norm_dpcpp.weight.grad.cpu())

    def test_layer_norm_bert(self, dtype=torch.float):
        linear = nn.Linear(512, 512)
        layer_norm1 = nn.LayerNorm(512)
        layer_norm2 = nn.LayerNorm([1024, 512])
        x = torch.randn([1024, 512], device=cpu_device, dtype=torch.float)

        y = linear(x)
        ref1 = layer_norm1(y)
        ref2 = layer_norm2(y)

        x = x.to(dpcpp_device)
        linear = linear.to(dpcpp_device)
        layer_norm1 = layer_norm1.to(dpcpp_device)
        layer_norm2 = layer_norm2.to(dpcpp_device)

        y = linear(x)
        real1 = layer_norm1(y)
        real2 = layer_norm2(y)

        self.assertEqual(ref1, real1.cpu(), rtol=10e-5, atol=10e-5)
        self.assertEqual(ref2, real2.cpu(), rtol=10e-5, atol=10e-5)

    def test_layer_norm_bfp16_training(self, dtype=torch.float):
        layernorm = nn.LayerNorm(10)
        x = torch.randn([10, 10]).requires_grad_()
        x.retain_grad()
        gy = torch.randn([10, 10])

        ref = layernorm(x)
        ref.backward(gy)
        ref_gx = x.grad
        print(ref)
        print(ref_gx)

        layernorm.zero_grad()

        layernorm = layernorm.to("xpu")
        x = x.bfloat16().to("xpu").requires_grad_()
        x.retain_grad()
        gy = gy.bfloat16().to("xpu")

        real = layernorm(x)
        real.backward(gy)
        real_gx = x.grad
        print(real.cpu())
        print(real_gx.cpu())

        diff = real.cpu().float() - ref
        print(diff)
        zero = torch.zeros([10, 10])

        self.assertEqual(ref, real.cpu().float(), rtol=10e-4, atol=10e-2)
        self.assertEqual(diff, zero, rtol=10e-4, atol=10e-2)

    def test_layer_norm_half(self, dtype=torch.half):
        x_i = torch.randn([1, 1, 3, 3], device=cpu_device)
        x_dpcpp_i = x_i.to(dpcpp_device).to(dtype)

        layernorm = nn.LayerNorm([1,3,3])
        y_cpu = layernorm(x_i)
        layernorm.to(dpcpp_device).to(dtype)
        y_dpcpp = layernorm(x_dpcpp_i)
        self.assertEqual(y_cpu, y_dpcpp.cpu().float(), atol=1e-2, rtol=0)

    def test_layer_norm_bfloat16(self, dtype=torch.bfloat16):
        x_i = torch.randn([1, 1, 3, 3], device=cpu_device)
        x_dpcpp_i = x_i.to(dpcpp_device).to(dtype)

        layernorm = nn.LayerNorm([1,3,3])
        y_cpu = layernorm(x_i)
        layernorm.to(dpcpp_device).to(dtype)
        y_dpcpp = layernorm(x_dpcpp_i)
        self.assertEqual(y_cpu, y_dpcpp.cpu().float(), atol=1e-1, rtol=0)
