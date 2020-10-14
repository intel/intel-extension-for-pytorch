import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestTorchMethod(TestCase):
    def test_is_nan(self, dtype=torch.float):
        y_cpu = torch.isnan(torch.tensor([1, float('nan'), 2]))
        y_dpcpp = torch.isnan(torch.tensor(
            [1, float('nan'), 2], device=torch.device("dpcpp")))
        print("cpu isnan", torch.isnan(torch.tensor([1, float('nan'), 2])))
        print("dpcpp isnan", torch.isnan(torch.tensor(
            [1, float('nan'), 2], device=torch.device("dpcpp"))))
        self.assertEqual(y_cpu, y_dpcpp)
