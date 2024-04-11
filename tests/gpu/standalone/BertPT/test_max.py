import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_max_1(self, dtype=torch.float):
        a_dpcpp = torch.randn(1, 2, dtype=dtype).to("xpu")
        a_cpu = a_dpcpp.to("cpu")
        b_cpu = torch.max(a_cpu, -2)
        b_dpcpp, b_dpcpp_index = a_dpcpp.max(-2)
        self.assertEqual(b_cpu[0], b_dpcpp.cpu())
        self.assertEqual(b_cpu[1], b_dpcpp_index.cpu())

    def test_max_bfloat16_1(self, dtype=torch.bfloat16):
        a_dpcpp = torch.randn(1, 2, dtype=dtype).to("xpu")
        a_cpu = a_dpcpp.to("cpu")
        b_cpu = torch.max(a_cpu, -2)
        b_dpcpp, b_dpcpp_index = a_dpcpp.max(-2)
        self.assertEqual(b_cpu[0], b_dpcpp.cpu())
        self.assertEqual(b_cpu[1], b_dpcpp_index.cpu())

    def test_max_float16_1(self, dtype=torch.float16):
        a_dpcpp = torch.randn(1, 2, dtype=dtype).to("xpu")
        a_cpu = a_dpcpp.to("cpu")
        b_cpu = torch.max(a_cpu, -2)
        b_dpcpp, b_dpcpp_index = a_dpcpp.max(-2)
        self.assertEqual(b_cpu[0], b_dpcpp.cpu())
        self.assertEqual(b_cpu[1], b_dpcpp_index.cpu())

    def test_max_2(self, dtype=torch.float):
        a_dpcpp = torch.randn(20, 30522, dtype=dtype).to("xpu")
        a_cpu = a_dpcpp.to("cpu")
        b_cpu = torch.max(a_cpu, -2)
        b_dpcpp, b_dpcpp_index = a_dpcpp.max(-2)
        self.assertEqual(b_cpu[0], b_dpcpp.cpu())
        self.assertEqual(b_cpu[1], b_dpcpp_index.cpu())

    def test_max_bfloat16_2(self, dtype=torch.bfloat16):
        a_dpcpp = torch.randn(20, 30522, dtype=dtype).to("xpu")
        a_cpu = a_dpcpp.to("cpu")
        b_cpu = torch.max(a_cpu, -2)
        b_dpcpp, b_dpcpp_index = a_dpcpp.max(-2)
        self.assertEqual(b_cpu[0], b_dpcpp.cpu())
        self.assertEqual(b_cpu[1], b_dpcpp_index.cpu())

    def test_max_float16_2(self, dtype=torch.float16):
        a_dpcpp = torch.randn(20, 30522, dtype=dtype).to("xpu")
        a_cpu = a_dpcpp.to("cpu")
        b_cpu = torch.max(a_cpu, -2)
        b_dpcpp, b_dpcpp_index = a_dpcpp.max(-2)
        self.assertEqual(b_cpu[0], b_dpcpp.cpu())
        self.assertEqual(b_cpu[1], b_dpcpp_index.cpu())
