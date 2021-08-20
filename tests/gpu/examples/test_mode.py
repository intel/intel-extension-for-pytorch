import torch
from torch.testing._internal.common_utils import TestCase
import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_mode(self, dtype=torch.float):

        # mode(input, 0)
        input_cpu = torch.randint(5, (10, 10)) + torch.randn(10, 10).long()
        output_cpu, output_indices_cpu = torch.mode(input_cpu, 0)

        input_xpu = input_cpu.detach().to(dpcpp_device)
        output_xpu, output_indices_xpu = torch.mode(input_xpu, 0)

        print('input = ', input_cpu)
        print('cpu output value = ', output_cpu)
        print('cpu output indices = ', output_indices_cpu)

        print('xpu output value = ', output_xpu.cpu())
        print('xpu output indices = ', output_indices_xpu.cpu())

        self.assertEqual(input_cpu, input_xpu.cpu())
        self.assertEqual(output_cpu, output_xpu.cpu())
        self.assertEqual(output_indices_cpu, output_indices_xpu.cpu())

        # mode(input, 1)
        input_cpu = torch.randint(5, (10, 10)) + torch.randn(10, 10).long()
        output_cpu, output_indices_cpu = torch.mode(input_cpu, 1)

        input_xpu = input_cpu.detach().to(dpcpp_device)
        output_xpu, output_indices_xpu = torch.mode(input_xpu, 1)

        print('input = ', input_cpu)
        print('cpu output value = ', output_cpu)
        print('cpu output indices = ', output_indices_cpu)

        print('xpu output value = ', output_xpu.cpu())
        print('xpu output indices = ', output_indices_xpu.cpu())

        self.assertEqual(input_cpu, input_xpu.cpu())
        self.assertEqual(output_cpu, output_xpu.cpu())
        self.assertEqual(output_indices_cpu, output_indices_xpu.cpu())

        # mode(input), input is one-dimension tensor
        input_cpu = torch.randint(5, (10,)) + torch.randn(10).long()
        output_cpu, output_indices_cpu = torch.mode(input_cpu)

        input_xpu = input_cpu.detach().to(dpcpp_device)
        output_xpu, output_indices_xpu = torch.mode(input_xpu)

        print('input = ', input_cpu)
        print('cpu output value = ', output_cpu)
        print('cpu output indices = ', output_indices_cpu)

        print('xpu output value = ', output_xpu.cpu())
        print('xpu output indices = ', output_indices_xpu.cpu())

        self.assertEqual(input_cpu, input_xpu.cpu())
        self.assertEqual(output_cpu, output_xpu.cpu())
        self.assertEqual(output_indices_cpu, output_indices_xpu.cpu())