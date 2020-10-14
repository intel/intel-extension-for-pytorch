import numpy
import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestNNMethod(TestCase):
    def test_topk(self, dtype=torch.float):

        #x_cpu = torch.arange(-6., 6.)
        x_cpu = torch.tensor([[-0.2911, -1.3204,  -2.6425,  -2.4644,  -0.6018, -0.0839, -
                               0.1322, -0.4713, -0.3586, -0.8882]], device=torch.device("cpu"), dtype=torch.float)

        #x_cpu = torch.ones([2], dtype=torch.float)
        #x_cpu[0] = 1.0
        #x_cpu[1] = -1.0
        x_dpcpp = x_cpu.to("dpcpp")

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

        #x_cpu1 = torch.randn([1, 10], device=torch.device("cpu"), dtype=torch.float)
        x_cpu1 = torch.tensor([[-0.2911, -1.3204,  -2.6425,  -2.4644,  -0.6018, -0.0839, -
                                0.1322, -0.4713, -0.3586, -0.8882]], device=torch.device("cpu"), dtype=torch.float)
        x_dpcpp1 = x_cpu1.to("dpcpp")

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
        x_dpcpp1 = x_cpu1.to("dpcpp")

        y_cpu0, y_cpu1 = x_cpu1.topk(5, 1, True, True)

        y_dpcpp0, y_dpcpp1 = x_dpcpp1.topk(5, 1, True, True)

        print("y_dpcpp0 = ", y_dpcpp0.to("cpu"),
              "y_dpcpp1 = ", y_dpcpp1.to("cpu"))
        self.assertEqual(x_cpu1, x_dpcpp1.cpu())
        self.assertEqual(y_cpu0, y_dpcpp0.cpu())
        self.assertEqual(y_cpu1, y_dpcpp1.cpu())
