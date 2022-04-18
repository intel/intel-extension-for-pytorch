import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_complex_zeros_like(self):
        a = torch.randn([3, 3], dtype=torch.complex64)
        b = torch.zeros_like(a)
        a_xpu = a.to(dpcpp_device)
        b_xpu = torch.zeros_like(a_xpu)
        self.assertEqual(b, b_xpu.to("cpu"))
