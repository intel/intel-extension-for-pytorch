import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch
import pytest
from torch.xpu.cpp_extension import IS_WINDOWS

import os
import tempfile


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

    def test_device_hierarchy(self):
        fname = tempfile.mkdtemp() + "_device_hierarchy"
        os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "COMPOSITE"
        cmd = f"""python -c "import torch;import intel_extension_for_pytorch;\
        torch.xpu.enable_tile_as_device();print(torch.xpu.device_count())" > {fname}"""
        os.system(cmd)
        del os.environ["ZE_FLAT_DEVICE_HIERARCHY"]
        f = open(fname)
        device_count = int(f.read())
        f.close()
        os.remove(fname)
        self.assertEqual(device_count, torch.xpu.device_count())
