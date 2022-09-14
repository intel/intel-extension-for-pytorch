import torch
from torch.testing._internal.common_utils import (TestCase,
                                                  repeat_test_for_types)

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_scatter(self, dtype=torch.float):

        x_cpu = torch.rand(2, 5)
        x_dpcpp = x_cpu.to(dpcpp_device)

        y_cpu = torch.zeros(3, 5).scatter_(0, torch.tensor(
            [[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x_cpu)

        print("y_cpu", y_cpu)

        y_dpcpp = torch.zeros(3, 5, device=dpcpp_device).scatter_(0, torch.tensor(
            [[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], device=dpcpp_device), x_dpcpp)

        print("y_dpcpp", y_dpcpp.cpu())

        z_cpu = torch.zeros(2, 4).scatter_(1, torch.tensor([[2], [3]]), 1.23)

        print("z_cpu", z_cpu)

        z_dpcpp = torch.zeros(2, 4, device=dpcpp_device).scatter_(
            1, torch.tensor([[2], [3]], device=dpcpp_device), 1.23)

        print("z_dpcpp", z_dpcpp.cpu())

        w1_cpu = torch.zeros(3, 5).scatter(0, torch.tensor(
            [[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x_cpu)

        print("w1_cpu", w1_cpu)

        w1_dpcpp = torch.zeros(3, 5, device=dpcpp_device).scatter(0, torch.tensor(
            [[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], device=dpcpp_device), x_dpcpp)

        print("w1_dpcpp", w1_dpcpp)

        w2_cpu = torch.zeros(2, 4).scatter(1, torch.tensor([[2], [3]]), 1.23)

        print("w2_cpu", w2_cpu)

        w2_dpcpp = torch.zeros(2, 4, device=dpcpp_device).scatter(
            1, torch.tensor([[2], [3]], device=dpcpp_device), 1.23)

        print("w2_dpcpp", w2_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(z_cpu, z_dpcpp.cpu())
        self.assertEqual(w1_cpu, w1_dpcpp.cpu())
        self.assertEqual(w2_cpu, w2_dpcpp.cpu())

    @repeat_test_for_types([torch.int, torch.int8, torch.int16,
                            torch.double, torch.float, torch.half, torch.bfloat16])
    def test_scatter_out(self, dtype=torch.float):
        size = 10
        src = torch.randint(0, size, (1, size), dtype=dtype)
        src_xpu = src.clone().to(dpcpp_device)
        index = torch.randperm(size).resize_(1, size)
        index_xpu = index.clone().to(dpcpp_device)
        input = torch.randint(0, size, (1, size), dtype=src.dtype)
        input_xpu = input.clone().to(dpcpp_device)
        # test scatter.reduce_out
        result = input.scatter_(1, index, src, reduce="add")
        result_xpu = input_xpu.scatter_(1, index_xpu, src_xpu, reduce="add")
        print(result)
        print(result_xpu)
        self.assertEqual(result, result_xpu.to(cpu_device), atol=1e-5, rtol=1e-5)
        result = input.scatter_(1, index, src, reduce="multiply")
        result_xpu = input_xpu.scatter_(1, index_xpu, src_xpu, reduce="multiply")
        print(result)
        print(result_xpu)
        self.assertEqual(result, result_xpu.to(cpu_device), atol=1e-5, rtol=1e-5)
        # test scatter.src_out
        input = torch.randint(0, size, (1, size), dtype=src.dtype)
        input_xpu = input.clone().to(dpcpp_device)
        torch.scatter(input, 1, index, src, out=result)
        torch.scatter(input_xpu, 1, index_xpu, src_xpu, out=result_xpu)
        print(result)
        print(result_xpu)
        self.assertEqual(result, result_xpu.to(cpu_device), atol=1e-5, rtol=1e-5)
        # test scatter.value_reduce_out
        input = torch.randint(0, size, (1, size), dtype=src.dtype)
        input_xpu = input.clone().to(dpcpp_device)
        result = input.scatter_(1, index, 2.0, reduce="add")
        result_xpu = input_xpu.scatter_(1, index_xpu, 2.0, reduce="add")
        self.assertEqual(result, result_xpu.to(cpu_device), atol=1e-5, rtol=1e-5)
        result = input.scatter_(1, index, 2.0, reduce="multiply")
        result_xpu = input_xpu.scatter_(1, index_xpu, 2.0, reduce="multiply")
        self.assertEqual(result, result_xpu.to(cpu_device), atol=1e-5, rtol=1e-5)
