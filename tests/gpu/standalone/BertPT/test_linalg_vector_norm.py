import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


class TestNNMethod(TestCase):
    def test_linalg_vector_norm(self):
        a = torch.randn(398, dtype=torch.float)
        b = a.reshape((199, 2))
        a_cpu = torch.linalg.vector_norm(a, ord=3.5)

        a_xpu = torch.linalg.vector_norm(a.to("xpu"), ord=3.5)
        self.assertEqual(a_xpu, a_xpu.cpu())
