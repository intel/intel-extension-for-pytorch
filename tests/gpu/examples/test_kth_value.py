import torch
from torch.testing._internal.common_utils import TestCase

import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_kth_value(self, dtype=torch.float):
        x_cpu = torch.tensor([[-0.2911, -1.3204, -2.6425, -2.4644, -0.6018, -0.0839, -
                               0.1322, -0.4713, -0.3586, -0.8882]], device=torch.device("cpu"), dtype=torch.float)
        x_dpcpp = torch.tensor([[-0.2911, -1.3204, -2.6425, -2.4644, -0.6018, -0.0839, -
                                 0.1322, -0.4713, -0.3586, -0.8882]], device=torch.device("xpu"), dtype=torch.float)

        print("y = ", torch.kthvalue(x_cpu, 4))
        y = torch.kthvalue(x_cpu, 4)
        y_dpcpp = torch.kthvalue(x_dpcpp, 4)
        print("y_dpcpp = ", y_dpcpp[0].to("cpu"), y_dpcpp[1].to("cpu"))
        self.assertEqual(y[0], y_dpcpp[0].to(cpu_device))
        self.assertEqual(y[1], y_dpcpp[1].to(cpu_device))

        print("y = ", torch.kthvalue(x_cpu.resize_(2, 5), 1))
        y = torch.kthvalue(x_cpu.resize_(2, 5), 1)
        y_dpcpp = torch.kthvalue(x_dpcpp.resize_(2, 5), 1)
        print("y_dpcpp = ", y_dpcpp[0].to("cpu"), y_dpcpp[1].to("cpu"))
        self.assertEqual(y[0], y_dpcpp[0].to(cpu_device))
        self.assertEqual(y[1], y_dpcpp[1].to(cpu_device))
