import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_topk(self, dtype=torch.float):

        # x_cpu = torch.arange(-6., 6.)
        x_cpu = torch.tensor([[-0.2911, -1.3204, -2.6425, -2.4644, -0.6018, -0.0839, -
                               0.1322, -0.4713, -0.3586, -0.8882]], device=torch.device("cpu"), dtype=torch.float)

        # x_cpu = torch.ones([2], dtype=torch.float)
        # x_cpu[0] = 1.0
        # x_cpu[1] = -1.0
        x_dpcpp = x_cpu.to("xpu")

        y_cpu, y_cpu_idx = torch.topk(x_cpu, 2)

        print("x: ", x_cpu)
        print("y: ", y_cpu)
        print("x_dpcpp.dim", x_dpcpp.dim())
        y_dpcpp, y_dpcpp_idx = torch.topk(x_dpcpp, 2)
        print("y_dpcpp: ", y_dpcpp.cpu(), y_dpcpp_idx.cpu())
        self.assertEqual(x_cpu, x_dpcpp.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(y_cpu_idx, y_dpcpp_idx.cpu())

        print("==================================")

        # x_cpu1 = torch.randn([1, 10], device=torch.device("cpu"), dtype=torch.float)
        x_cpu1 = torch.tensor([[-0.2911, -1.3204, -2.6425, -2.4644, -0.6018, -0.0839, -
                                0.1322, -0.4713, -0.3586, -0.8882]], device=torch.device("cpu"), dtype=torch.float)
        x_dpcpp1 = x_cpu1.to("xpu")

        print("x_cpu1=", x_cpu1)
        y_cpu0, y_cpu1 = x_cpu1.topk(5, 1, True, True)

        y_dpcpp0, y_dpcpp1 = x_dpcpp1.topk(5, 1, True, True)

        print("y_cpu0 = ", y_cpu0, "y_cpu1 = ", y_cpu1)
        print("y_dpcpp0 = ", y_dpcpp0.to("cpu"),
              "y_dpcpp1 = ", y_dpcpp1.to("cpu"))
        self.assertEqual(x_cpu1, x_dpcpp1.cpu())
        self.assertEqual(y_cpu0, y_dpcpp0.cpu())
        self.assertEqual(y_cpu1, y_dpcpp1.cpu())

        x_cpu1 = torch.randn(
            [3000, 3000], device=torch.device("cpu"), dtype=torch.float)
        x_dpcpp1 = x_cpu1.to("xpu")

        y_cpu0, y_cpu1 = x_cpu1.topk(5, 1, True, True)

        y_dpcpp0, y_dpcpp1 = x_dpcpp1.topk(5, 1, True, True)

        print("y_cpu0 = ", y_cpu0.to("cpu"))
        print("y_dpcpp0 = ", y_dpcpp0.to("cpu"))
        print("y_cpu1 = ", y_cpu1.to("cpu"))
        print("y_dpcpp1 = ", y_dpcpp1.to("cpu"))
        self.assertEqual(x_cpu1, x_dpcpp1.cpu())
        self.assertEqual(y_cpu0, y_dpcpp0.cpu())
        self.assertEqual(y_cpu1, y_dpcpp1.cpu())

        # case for GPT-J
        a = torch.randn([1, 201600])
        a_xpu = a.to('xpu')
        sort_cpu, index_cpu = torch.topk(a, 5)
        sort_xpu, index_xpu = torch.topk(a_xpu, 5)

        self.assertEqual(sort_cpu, sort_xpu.cpu())
        self.assertEqual(index_cpu, index_xpu.cpu())

        # case for transformer-lt
        a = torch.randn([1, 5, 33712])
        key = torch.Tensor([[ -8.5410,  -8.8709,  -9.3082,  -9.3752,  -9.6210, -10.6507, -10.7050,
                -10.9694, -11.0113, -11.0896],
                [ -9.3133, -10.0262, -13.2868, -13.5677, -13.7332, -16.4816, -16.9183,
                -17.2554, -17.4845, -17.5146]])
        value = torch.Tensor([[11744, 11744,  3701, 11976,     5, 18681, 15307,    39,    93, 17219],
                [    2,     2,     2,     2,     2,   131,   131,   131,   131,    89]]).long()
        a_xpu = a.to("xpu")
        key_xpu = key.to("xpu")
        value_xpu = value.to("xpu")
        torch.topk(a.view(1, -1), 10, out=(key, value))
        torch.topk(a_xpu.view(1, -1), 10, out=(key_xpu, value_xpu))
        self.assertEqual(key, key_xpu.cpu())
        self.assertEqual(value, value_xpu.cpu())
