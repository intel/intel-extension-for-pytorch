import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_hardtanh(self, dtype=torch.float):
        grad_output_cpu = torch.randn([1, 1, 8, 8])
        grad_output_xpu = grad_output_cpu.xpu()

        # cpu
        linear = nn.Linear(8, 8)
        tanh = nn.Hardtanh()
        print("linear weight", linear.weight)
        x_cpu = torch.ones([1, 1, 8, 8], device=cpu_device, dtype=dtype)
        print("x_cpu", x_cpu)
        z_cpu = linear(x_cpu)
        print("z_cpu", z_cpu)
        y_cpu = tanh(z_cpu)
        print("y_cpu", y_cpu)
        y_cpu.backward(grad_output_cpu)
        linear_weight_grad_cpu = linear.weight.grad.clone()
        print("linear grad", linear_weight_grad_cpu)
        linear.zero_grad()

        # xpu
        linear_xpu = linear.to("xpu")
        tanh_xpu = tanh.to("xpu")
        print("xpu linear weight", linear_xpu.weight.cpu())
        x_xpu = x_cpu.to("xpu")
        print("x_xpu", x_xpu.cpu())
        z_xpu = linear_xpu(x_xpu)
        print("z_xpu", z_xpu.cpu())
        y_xpu = tanh(z_xpu)
        print("y_xpu", y_xpu.cpu())
        y_xpu.backward(grad_output_xpu)
        linear_weight_grad_xpu = linear.weight.grad.clone()
        print("xpu linear grad", linear_weight_grad_xpu.cpu())
        linear_xpu.zero_grad()

        self.assertEqual(z_cpu, z_xpu)
        self.assertEqual(y_cpu, y_xpu)
        self.assertEqual(linear_weight_grad_cpu, linear_weight_grad_xpu)

    def test_hardtanh_half(self, dtype=torch.float):
        grad_output_cpu = torch.randn([1, 1, 8, 8])
        grad_output_xpu = grad_output_cpu.to("xpu", dtype=torch.float16)

        # cpu
        linear = nn.Linear(8, 8)
        tanh = nn.Hardtanh()
        print("linear weight", linear.weight)
        x_cpu = torch.ones([1, 1, 8, 8], device=cpu_device, dtype=dtype)
        print("x_cpu", x_cpu)
        z_cpu = linear(x_cpu)
        print("z_cpu", z_cpu)
        y_cpu = tanh(z_cpu)
        print("y_cpu", y_cpu)
        y_cpu.backward(grad_output_cpu)
        linear_weight_grad_cpu = linear.weight.grad.clone()
        print("linear grad", linear_weight_grad_cpu)
        linear.zero_grad()

        # xpu
        linear_xpu = linear.to("xpu", dtype=torch.float16)
        tanh_xpu = tanh.to("xpu", dtype=torch.float16)
        print("xpu linear weight", linear_xpu.weight.to("cpu", dtype=torch.float32))
        x_xpu = x_cpu.to("xpu", dtype=torch.float16)
        print("x_xpu", x_xpu.to("cpu", dtype=torch.float32))
        z_xpu = linear_xpu(x_xpu)
        print("z_xpu", z_xpu.to("cpu", dtype=torch.float32))
        y_xpu = tanh(z_xpu)
        print("y_xpu", y_xpu.to("cpu", dtype=torch.float32))
        y_xpu.backward(grad_output_xpu)
        linear_weight_grad_xpu = linear.weight.grad.clone()
        print("xpu linear grad", linear_weight_grad_xpu.to("cpu", dtype=torch.float32))
        linear_xpu.zero_grad()

        self.assertEqual(
            z_cpu, z_xpu.to("cpu", dtype=torch.float32), atol=1e-3, rtol=1e-3
        )
        self.assertEqual(
            y_cpu, y_xpu.to("cpu", dtype=torch.float32), atol=1e-3, rtol=1e-3
        )
        self.assertEqual(
            linear_weight_grad_cpu,
            linear_weight_grad_xpu.to("cpu", dtype=torch.float32),
            atol=1e-3,
            rtol=1e-3,
        )
