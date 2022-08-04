import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_trace(self, dtype=torch.float):
        x_cpu = torch.tensor([[-0.2911, -1.3204, -2.6425], [-2.4644, -0.6018, -0.0839],
                              [-0.1322, -0.4713, -0.3586]], device=torch.device("cpu"), dtype=torch.float)
        y = torch.trace(x_cpu)
        print("y = ", y)

        x_dpcpp = torch.tensor([[-0.2911, -1.3204, -2.6425], [-2.4644, -0.6018, -0.0839],
                                [-0.1322, -0.4713, -0.3586]], device=torch.device("xpu"), dtype=torch.float)
        y_dpcpp = torch.trace(x_dpcpp)

        print("y_dpcpp = ", y_dpcpp.to("cpu"))
        self.assertEqual(y, y_dpcpp.cpu())

        print("x_cpu trace = ", x_cpu.trace())
        print("x_dpcpp trace = ", x_dpcpp.trace().cpu())
        self.assertEqual(x_cpu.trace(), x_dpcpp.trace().cpu())
