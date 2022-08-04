import torch
from torch.testing._internal.common_utils import (TestCase,
                                                  repeat_test_for_types)

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @repeat_test_for_types([torch.float, torch.half, torch.bfloat16])
    def test_nan_to_num(self, dtype=torch.float):
        x = torch.randn(3, 3, 3, 3, dtype=torch.float)
        with torch.no_grad():
            x[torch.rand_like(x) < 0.2] = float('nan')
            x[torch.rand_like(x) < 0.2] = float('inf')
            x[torch.rand_like(x) < 0.2] = -float('inf')

        x_cpu = x.clone()
        x_dpcpp = x_cpu.clone().to(dpcpp_device)

        self.assertEqual(x_cpu.nan_to_num(), x_dpcpp.nan_to_num().to(dpcpp_device))
        self.assertEqual(x_cpu.nan_to_num(nan=1.2), x_dpcpp.nan_to_num(nan=1.2).to(dpcpp_device))
        self.assertEqual(x_cpu.nan_to_num(nan=1.2, posinf=2.0),
                         x_dpcpp.nan_to_num(nan=1.2, posinf=2.0).to(dpcpp_device))
        self.assertEqual(x_cpu.nan_to_num(nan=1.2, posinf=2.0, neginf=-2.0),
                         x_dpcpp.nan_to_num(nan=1.2, posinf=2.0, neginf=-2.0).to(dpcpp_device))

        x_cpu = torch.nan_to_num(x_cpu, nan=1.2, posinf=2.0, neginf=-2.0)
        x_dpcpp = torch.nan_to_num(x_dpcpp, nan=1.2, posinf=2.0, neginf=-2.0)
        self.assertEqual(x_cpu, x_dpcpp.to(dpcpp_device))
