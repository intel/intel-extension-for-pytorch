import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestTorchMethod(TestCase):
    def test_abs(self, dtype=torch.float):
        data = [[-0.2911, -1.3204,  -2.6425,  -2.4644,  -
                 0.6018, -0.0839, -0.1322, -0.4713, -0.3586, -0.8882]]
        x = torch.tensor(data, device=cpu_device)
        x_dpcpp = x.to(dpcpp_device)
        y = torch.abs(x)
        y_dpcpp = torch.abs(x_dpcpp)
        self.assertEqual(y, y_dpcpp.to(cpu_device))
