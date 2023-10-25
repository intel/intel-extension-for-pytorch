import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch
import pytest
from torch.xpu.cpp_extension import IS_WINDOWS


class TestTorchMethod(TestCase):
    def test_device_guard(self):
        target_device = torch.xpu.current_device()
        exchange_device = (target_device + 1) % torch.xpu.device_count()
        with torch.xpu._DeviceGuard(exchange_device):
            self.assertEqual(torch.xpu.current_device(), exchange_device)
        self.assertEqual(torch.xpu.current_device(), target_device)

    @pytest.mark.skipif(IS_WINDOWS, reason="fork is not supported on windows.")
    def test_device_count(self):
        assert (
            intel_extension_for_pytorch._C._getDeviceCount()
            == torch.xpu._raw_device_count()
        )

    @pytest.mark.skipif(IS_WINDOWS, reason="fork is not supported on windows.")
    def test_device_has_fp64_support(self):
        for i in range(torch.xpu.device_count()):
            is_valid, is_supported = torch.xpu.utils._raw_has_fp64_dtype(i)
            assert (
                intel_extension_for_pytorch._C._has_fp64_dtype(i) == is_supported
                and is_valid
            )
