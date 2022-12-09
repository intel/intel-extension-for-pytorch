import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(torch.xpu.has_multi_context(), reason="torch.xpu.has_multi_context()")
    def test_dlpack(self):
        src = torch.rand((2, 12), device=dpcpp_device)
        dst = src.clone().to(cpu_device)
        dlpack = torch.to_dlpack(src)
        tensor = torch.from_dlpack(dlpack)
        self.assertEqual(tensor.to(cpu_device), dst)

    @pytest.mark.skipif(torch.xpu.has_multi_context(), reason="torch.xpu.has_multi_context()")
    def test_usm(self):
        src = torch.rand((2, 12), device=dpcpp_device)
        dst = src.clone().to(cpu_device).reshape(4, 6)
        usm = intel_extension_for_pytorch.xpu.to_usm(src)
        tensor = intel_extension_for_pytorch.xpu.from_usm(usm, dst.dtype, (4, 6))
        self.assertEqual(tensor.to(cpu_device), dst)

        src = torch.rand((2, 12), device=dpcpp_device)
        dst = src.clone().to(cpu_device).reshape(4, 6)
        usm = intel_extension_for_pytorch.xpu.to_usm(src)
        tensor = intel_extension_for_pytorch.xpu.from_usm(usm, dst.dtype, (4, 6))
        self.assertEqual(tensor.to(cpu_device), dst)
