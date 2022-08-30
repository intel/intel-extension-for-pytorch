import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa


class TestTorchMethod(TestCase):
    def test_gcd(self, dtype=torch.int):
        a_cpu = torch.randint(0, 100000, [100, 100])
        b_cpu = torch.randint(0, 1000, [100])
        gcd_cpu = torch.gcd(a_cpu, b_cpu)

        a_xpu = a_cpu.to("xpu")
        b_xpu = b_cpu.to("xpu")
        gcd_xpu = torch.gcd(a_xpu, b_xpu)
        self.assertEqual(gcd_cpu, gcd_xpu.to("cpu"))

        out_cpu = torch.empty((100, 100), dtype=dtype)
        out_xpu = torch.empty((100, 100), dtype=dtype).to("xpu")
        torch.gcd(a_cpu, b_cpu, out=out_cpu)
        torch.gcd(a_xpu, b_xpu, out=out_xpu)
        self.assertEqual(out_cpu, out_xpu.to("cpu"))

        a_cpu.gcd_(b_cpu)
        a_xpu.gcd_(b_xpu)
        self.assertEqual(a_cpu, a_xpu.to("cpu"))

    def test_lcm(self, dtype=torch.int):
        a_cpu = torch.randint(0, 100000, [100, 100])
        b_cpu = torch.randint(0, 1000, [100])
        lcm_cpu = torch.lcm(a_cpu, b_cpu)

        a_xpu = a_cpu.to("xpu")
        b_xpu = b_cpu.to("xpu")
        lcm_xpu = torch.lcm(a_xpu, b_xpu)
        self.assertEqual(lcm_cpu, lcm_xpu.to("cpu"))

        out_cpu = torch.empty((100, 100), dtype=dtype)
        out_xpu = torch.empty((100, 100), dtype=dtype).to("xpu")
        torch.lcm(a_cpu, b_cpu, out=out_cpu)
        torch.lcm(a_xpu, b_xpu, out=out_xpu)
        self.assertEqual(out_cpu, out_xpu.to("cpu"))

        a_cpu.lcm_(b_cpu)
        a_xpu.lcm_(b_xpu)
        self.assertEqual(a_cpu, a_xpu.to("cpu"))
