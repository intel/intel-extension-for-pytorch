import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

import numpy as np

np.set_printoptions(threshold=np.inf)

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_index(self, dtype=torch.float):
        dtypes = [torch.float, torch.half]
        li = torch.tensor([i for i in range(27) for j in range(i)], device=cpu_device)
        lj = torch.tensor([j for i in range(27) for j in range(i)], device=cpu_device)

        for datatype in dtypes:
            # 2-dims tensor index
            x_cpu = torch.randn([327, 27], dtype=torch.float, device=cpu_device)
            if datatype == torch.half:
                x_cpu = x_cpu.half()

            res_cpu = x_cpu[:, lj]
            x_dpcpp = x_cpu.to(dpcpp_device)
            res_dpcpp = x_dpcpp[:, lj]
            self.assertEqual(res_cpu, res_dpcpp.to(cpu_device))

            # 3-dims tensor index
            x_cpu = torch.randn([327, 27, 27], dtype=torch.float, device=cpu_device)
            if datatype == torch.half:
                x_cpu = x_cpu.half()

            res_cpu = x_cpu[:, li, lj]
            x_dpcpp = x_cpu.to(dpcpp_device)
            res_dpcpp = x_dpcpp[:, li, lj]
            self.assertEqual(res_cpu, res_dpcpp.to(cpu_device))

            res_cpu = x_cpu[:, :, lj]
            x_dpcpp = x_cpu.to(dpcpp_device)
            res_dpcpp = x_dpcpp[:, :, lj]
            self.assertEqual(res_cpu, res_dpcpp.to(cpu_device))

            # 4-dims tensor index
            x_cpu = torch.randn([327, 27, 27, 27], dtype=torch.float, device=cpu_device)
            if datatype == torch.half:
                x_cpu = x_cpu.half()

            res_cpu = x_cpu[:, li, lj, lj]
            x_dpcpp = x_cpu.to(dpcpp_device)
            res_dpcpp = x_dpcpp[:, li, lj, lj]
            self.assertEqual(res_cpu, res_dpcpp.to(cpu_device))

            res_cpu = x_cpu[:, :, li, lj]
            x_dpcpp = x_cpu.to(dpcpp_device)
            res_dpcpp = x_dpcpp[:, :, li, lj]
            self.assertEqual(res_cpu, res_dpcpp.to(cpu_device))

    def test_index_out(self, dtype=torch.float):
        x = torch.rand([5, 20, 10])
        x_dpcpp = x.to(dpcpp_device)
        b = torch.rand([1, 20, 10])
        b_dpcpp = b.to(dpcpp_device)
        for i in range(5):
            torch.ops.aten.index(x, torch.tensor([i]), out=b)
            torch.ops.aten.index(
                x_dpcpp, torch.tensor([i]).to(dpcpp_device), out=b_dpcpp
            )
            self.assertEqual(b, b_dpcpp.to(cpu_device))

        b = torch.rand([1, 10])
        b_dpcpp = b.to(dpcpp_device)
        for i in range(5):
            for j in range(20):
                torch.ops.aten.index(x, [torch.tensor([i]), torch.tensor([j])], out=b)
                torch.ops.aten.index(
                    x_dpcpp,
                    [torch.tensor([i]).to(dpcpp_device), torch.tensor([j])],
                    out=b_dpcpp,
                )
                self.assertEqual(b, b_dpcpp.to(cpu_device))

    def test_index_of_bool_mask(self, dtype=torch.float):
        a_cpu = torch.randn(4, 15000, dtype=dtype)
        b_cpu = torch.randn(4, 15000, dtype=dtype)
        mask_cpu = a_cpu < b_cpu

        a_xpu = a_cpu.xpu()
        mask_xpu = mask_cpu.xpu()

        output_cpu = a_cpu[mask_cpu]
        output_xpu = a_xpu[mask_xpu]

        self.assertEqual(output_cpu, output_xpu)

    def test_unsafe_index(self, dtype=torch.float):
        input_cpu = torch.randn(5, 20, dtype=dtype, device=cpu_device)
        input_dpcpp = input_cpu.to(dpcpp_device)

        for i in range(5):
            for j in range(20):
                output_cpu = torch._unsafe_index(
                    input_cpu, [torch.tensor([i]), torch.tensor([j])])
                output_dpcpp = torch._unsafe_index(
                    input_dpcpp, 
                    [torch.tensor([i]).to(dpcpp_device), 
                     torch.tensor([j]).to(dpcpp_device)])
                self.assertEqual(output_cpu, output_dpcpp.to(cpu_device))
