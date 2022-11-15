import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa


class TestAddInt(TestCase):
    def test_add_int(self):
        a = torch.empty([2, 512], dtype=torch.int32, device="xpu")
        b = torch.empty([1, 512], dtype=torch.int32, device="xpu")
        a_cpu = a.cpu()
        b_cpu = b.cpu()
        c = a + b
        c_cpu = a_cpu + b_cpu
        self.assertEqual(c.to("cpu"), c_cpu)
