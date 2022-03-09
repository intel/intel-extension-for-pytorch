import torch
from torch.testing._internal.common_utils import (TestCase,
                                                  repeat_test_for_types)

import intel_extension_for_pytorch

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @repeat_test_for_types([torch.cfloat, torch.cdouble])
    def test_real(self, dtype=torch.cfloat):
        x = torch.randn(3, 4, 5, dtype=dtype, device=cpu_device)
        x_xpu = x.to(xpu_device)
        self.assertEqual(x.real, x_xpu.real.to(cpu_device))
