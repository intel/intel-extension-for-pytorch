import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest


class TestTorchMethod(TestCase):
    def test_bincount(self, dtype=torch.float):
        x_cpu = torch.randint(0, 8, (5,), dtype=torch.int64)
        y_cpu = torch.linspace(0, 1, steps=5)
        x_dpcpp = x_cpu.to("xpu")
        y_dpcpp = y_cpu.to("xpu")
        self.assertEqual(x_cpu, x_dpcpp.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(torch.bincount(x_cpu), torch.bincount(x_dpcpp).cpu())

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_histc(self):
        # "histogram_cpu" not implemented for torch.int8, torch.int, torch.long,
        for dtype in [torch.int8, torch.int, torch.long, torch.float, torch.double]:
            cpu_dtype = dtype
            if dtype in [torch.int8, torch.int, torch.long]:
                cpu_dtype = torch.double
            x_dpcpp = torch.randint(0, 127, (5,), dtype=dtype, device="xpu")
            x_cpu = x_dpcpp.to(device="cpu", dtype=cpu_dtype)

            res = torch.histc(x_cpu, bins=4, min=0, max=3)
            res_dpcpp = torch.histc(x_dpcpp, bins=4, min=0, max=3)
            res_tensor = x_cpu.histc(bins=4, min=0, max=3)
            res_tensor_dpcpp = x_dpcpp.histc(bins=4, min=0, max=3)
            self.assertEqual(x_cpu, x_dpcpp.to(device="cpu", dtype=cpu_dtype))
            self.assertEqual(res, res_dpcpp.to(device="cpu", dtype=cpu_dtype))
            self.assertEqual(
                res_tensor, res_tensor_dpcpp.to(device="cpu", dtype=cpu_dtype)
            )
