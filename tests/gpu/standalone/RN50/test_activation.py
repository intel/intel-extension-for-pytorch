import torch
import torch.nn.functional
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import copy

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

shapes_Relu = [
            (1, 512, 7, 7),
            (1, 2048, 7, 7),
            (1, 64, 56, 56),
            (1, 128, 28, 28),
            (1, 128, 56, 56),
            (1, 256, 14, 14),
            (1, 256, 28, 28),
            (1, 256, 56, 56),
            (1, 512, 14, 14),
            (1, 512, 28, 28),
            (1, 1024, 14, 14),
            (1, 64, 112, 112)
        ]

shapes_Gelu = [
            (2, 384, 4096)
        ]
class TestNNMethod(TestCase):
    def test_activation_relu(self, dtype=torch.float):
        for shape in shapes_Relu:
            #print("\n================== test shape: ", shape, "==================")
            relu_ = torch.nn.functional.relu_
            relu = torch.nn.functional.relu
            x_cpu = torch.randn(shape, dtype=dtype, device=cpu_device)
            x_dpcpp = x_cpu.to("xpu")

            relu_(x_cpu)
            relu_(x_dpcpp)
            self.assertEqual(x_cpu, x_dpcpp.cpu())

            x_cpu.requires_grad_(True)
            x_dpcpp.requires_grad_(True)
            y_cpu = relu(x_cpu)
            y_dpcpp = relu(x_dpcpp)

            self.assertEqual(y_cpu, y_dpcpp.cpu())

            y_cpu.backward(x_cpu)
            y_dpcpp.backward(x_dpcpp)

            self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())
    
    def test_activation_relu_bfloat16(self, dtype=torch.bfloat16):
        for shape in shapes_Relu:
            #print("\n================== test shape: ", shape, "==================")
            relu_ = torch.nn.functional.relu_
            relu = torch.nn.functional.relu
            x_cpu = torch.randn(shape, dtype=dtype, device=cpu_device)
            x_dpcpp = x_cpu.to("xpu")

            relu_(x_cpu)
            relu_(x_dpcpp)
            self.assertEqual(x_cpu, x_dpcpp.cpu())

            x_cpu.requires_grad_(True)
            x_dpcpp.requires_grad_(True)
            y_cpu = relu(x_cpu)
            y_dpcpp = relu(x_dpcpp)

            self.assertEqual(y_cpu, y_dpcpp.cpu())

            y_cpu.backward(x_cpu)
            y_dpcpp.backward(x_dpcpp)

            self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_activation_relu_float16(self, dtype=torch.float16):
        for shape in shapes_Relu:
            #print("\n================== test shape: ", shape, "==================")
            relu_ = torch.nn.functional.relu_
            relu = torch.nn.functional.relu
            x_cpu = torch.randn(shape, dtype=dtype, device=cpu_device)
            x_dpcpp = x_cpu.to("xpu")

            relu_(x_cpu)
            relu_(x_dpcpp)
            self.assertEqual(x_cpu, x_dpcpp.cpu())

            x_cpu.requires_grad_(True)
            x_dpcpp.requires_grad_(True)
            y_cpu = relu(x_cpu)
            y_dpcpp = relu(x_dpcpp)

            self.assertEqual(y_cpu, y_dpcpp.cpu())

            y_cpu.backward(x_cpu)
            y_dpcpp.backward(x_dpcpp)

            self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_activation_gelu(self, dtype=torch.float):
        for shape in shapes_Gelu:
            #print("\n================== test shape: ", shape, "==================")
            C, H, W = shape[0], shape[1], shape[2]
            GELU = torch.nn.GELU()
            GELU_dpcpp = copy.deepcopy(GELU).to("xpu")
            x_cpu = torch.randn([C, H, W], dtype=dtype)
            x_dpcpp = x_cpu.to("xpu")
            x_cpu.requires_grad_(True)
            x_dpcpp.requires_grad_(True)
            y_cpu = GELU(x_cpu)
            y_dpcpp = GELU_dpcpp(x_dpcpp)

            self.assertEqual(y_cpu, y_dpcpp.cpu())

            # y_cpu = torch.tensor([[1, 1],[1, 1],[1, 1],[1, 1]]);
            # y_dpcpp = y_cpu.to("xpu")
            y_cpu.backward(x_cpu)
            y_dpcpp.backward(x_dpcpp)

            self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_activation_gelu_block(self, dtype=torch.float):
         for shape in shapes_Gelu:
            #print("\n================== test shape: ", shape, "==================")
            C, H, W = shape[0], shape[1], shape[2]
            to_block_cpu = torch.nn.Conv2d(C, C, kernel_size=3, padding=1)
            to_block_dpcpp = copy.deepcopy(to_block_cpu).xpu()
            with torch.xpu.onednn_layout():
                GELU = torch.nn.GELU()
                GELU_dpcpp = copy.deepcopy(GELU).to("xpu")
                x_cpu = torch.randn(shape)
                x_dpcpp = x_cpu.to("xpu")
                x_cpu.requires_grad_(True)
                x_dpcpp.requires_grad_(True)
                y_cpu = GELU(to_block_cpu(x_cpu))
                y_dpcpp = GELU_dpcpp(to_block_dpcpp(x_dpcpp))

                self.assertEqual(y_cpu, y_dpcpp.cpu())
                y_cpu.backward(x_cpu)
                y_dpcpp.backward(x_dpcpp)

                self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())
