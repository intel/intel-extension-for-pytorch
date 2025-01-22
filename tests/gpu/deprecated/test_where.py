import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_where(self, dtype=torch.float):
        x = torch.tensor([[0.6580, -1.0969, -0.4614], [-0.1034, -0.5790, 0.1497]])
        x_ones = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        x_dpcpp = torch.tensor(
            [[0.6580, -1.0969, -0.4614], [-0.1034, -0.5790, 0.1497]],
            device=torch.device("xpu"),
        )
        x_ones_dpcpp = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], device=torch.device("xpu")
        )
        self.assertEqual(x, x_dpcpp.to(cpu_device))
        self.assertEqual(x_ones, x_ones_dpcpp.to(cpu_device))
        result = torch.where(x > 0, x, x_ones)
        result_dpcpp_0 = torch.where(x_dpcpp > 0, x_dpcpp, x_ones_dpcpp)
        self.assertEqual(result, result_dpcpp_0.to(cpu_device))
        # test only one tensor on XPU device.
        x = torch.randn(1)[0]
        x_one = torch.tensor(1)
        x_dpcpp = x.to(dpcpp_device)
        x_one_dpcpp = x_one.to(dpcpp_device)
        result = torch.where(x < 0, x, x_one)
        result_dpcpp_0 = torch.where(x_dpcpp < 0, x, x_one)
        result_dpcpp_1 = torch.where(x < 0, x_dpcpp, x_one)
        result_dpcpp_2 = torch.where(x < 0, x, x_one_dpcpp)
        self.assertEqual(result, result_dpcpp_0.to(cpu_device))
        self.assertEqual(result, result_dpcpp_1.to(cpu_device))
        self.assertEqual(result, result_dpcpp_2.to(cpu_device))
