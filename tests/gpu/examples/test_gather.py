import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_gather(self, dtype=torch.float):
        t = torch.tensor([[1, 2], [3, 4]], device=cpu_device)
        t_cpu = torch.gather(t, 1, torch.tensor(
            [[0, 0], [1, 0]], device=cpu_device))

        print("cpu")
        print(t_cpu)

        t2 = torch.tensor([[1, 2], [3, 4]], device=dpcpp_device)
        t_dpcpp = torch.gather(t2, 1, torch.tensor(
            [[0, 0], [1, 0]], device=dpcpp_device))

        print("xpu")
        print(t_dpcpp.cpu())
        self.assertEqual(t, t2.to(cpu_device))
        self.assertEqual(t_cpu, t_dpcpp.to(dpcpp_device))
