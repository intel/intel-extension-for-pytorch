import torch
from torch.testing._internal.common_utils import TestCase, IS_WINDOWS

import intel_extension_for_pytorch  # noqa
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(IS_WINDOWS, reason="This UT needs to be disabled on Windows")
    def test_dlpack(self):
        src = torch.rand((2, 12), device=dpcpp_device)
        dst = src.clone().to(cpu_device)
        dlpack = torch.to_dlpack(src)
        tensor = torch.from_dlpack(dlpack)
        self.assertEqual(tensor.to(cpu_device), dst)
