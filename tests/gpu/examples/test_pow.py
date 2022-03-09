import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_pow(self, dtype=torch.float):

        x_cpu = torch.tensor(
            ([2.5, 3.1, 1.3]), dtype=torch.float, device=cpu_device)
        x_dpcpp = torch.tensor(
            ([2.5, 3.1, 1.3]), dtype=torch.float, device=dpcpp_device)

        y_cpu = torch.tensor(
            ([3.0, 3.0, 3.0]), dtype=torch.float, device=cpu_device)
        y_dpcpp = torch.tensor(
            ([3.0, 3.0, 3.0]), dtype=torch.float, device=dpcpp_device)

        print("pow x y cpu", torch.pow(x_cpu, y_cpu))
        print("pow x y dpcpp", torch.pow(x_dpcpp, y_dpcpp).cpu())
        self.assertEqual(torch.pow(x_cpu, y_cpu),
                         torch.pow(x_dpcpp, y_dpcpp).cpu())

        print("x.pow y cpu", x_cpu.pow(y_cpu))
        print("x.pow y dpcpp", x_dpcpp.pow(y_dpcpp).cpu())
        self.assertEqual(x_cpu.pow(y_cpu), x_dpcpp.pow(y_dpcpp).cpu())

        print("x.pow_ y cpu", x_cpu.pow_(y_cpu))
        print("x.pow_ y dpcpp", x_dpcpp.pow_(y_dpcpp).cpu())
        self.assertEqual(x_cpu.pow_(y_cpu), x_dpcpp.pow_(y_dpcpp).cpu())
