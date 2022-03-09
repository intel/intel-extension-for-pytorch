import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_is_nan(self, dtype=torch.float):
        y_cpu = torch.isnan(torch.tensor([1, float('nan'), 2]))
        y_dpcpp = torch.isnan(torch.tensor(
            [1, float('nan'), 2], device=torch.device("xpu")))
        print("cpu isnan", torch.isnan(torch.tensor([1, float('nan'), 2])))
        print("dpcpp isnan", torch.isnan(torch.tensor(
            [1, float('nan'), 2], device=torch.device("xpu"))))
        self.assertEqual(y_cpu, y_dpcpp)
