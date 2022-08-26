import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_logaddexp(self, dtype=torch.float):
        a_cpu = torch.randn([100, 100], dtype=dtype)
        b_cpu = torch.randn([100, 100], dtype=dtype)
        logaddexp_cpu = torch.logaddexp(a_cpu, b_cpu)

        a_xpu = a_cpu.to("xpu")
        b_xpu = b_cpu.to("xpu")
        logaddexp_xpu = torch.logaddexp(a_xpu, b_xpu)
        self.assertEqual(logaddexp_cpu, logaddexp_xpu.to("cpu"))

        torch.logaddexp(a_cpu, b_cpu, out=a_cpu)
        torch.logaddexp(a_xpu, b_xpu, out=a_xpu)
        self.assertEqual(a_cpu, a_xpu.to("cpu"))

    def test_logaddexp2(self, dtype=torch.float):
        a_cpu = torch.randn([100, 100], dtype=dtype)
        a_xpu = a_cpu.to("xpu")

        b_cpu = torch.randn([100, 100], dtype=dtype)
        b_xpu = b_cpu.to("xpu")

        logaddexp_cpu = torch.logaddexp2(a_cpu, b_cpu)
        logaddexp_xpu = torch.logaddexp2(a_xpu, b_xpu)
        self.assertEqual(logaddexp_cpu, logaddexp_xpu.to("cpu"))

        torch.logaddexp2(a_cpu, b_cpu, out=a_cpu)
        torch.logaddexp2(a_xpu, b_xpu, out=a_xpu)
        self.assertEqual(a_cpu, a_xpu.to("cpu"))
