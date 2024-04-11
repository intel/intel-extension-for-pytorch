import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


class TestNNMethod(TestCase):
    def test_linalg_vector_norm(self, dtype=torch.float):
        a = torch.randn(398, dtype=torch.float)
        b = a.reshape((199, 2))
        a_cpu = torch.linalg.vector_norm(a, ord=3.5)

        a_xpu = torch.linalg.vector_norm(a.to("xpu"), ord=3.5)
        self.assertEqual(a_xpu, a_xpu.cpu())

    def test_linalg_vector_norm_bfloat16(self, dtype=torch.bfloat16):
        a = torch.randn(398, dtype=dtype)
        b = a.reshape((199, 2))
        a_cpu = torch.linalg.vector_norm(a, ord=3.5)

        a_xpu = torch.linalg.vector_norm(a.to("xpu"), ord=3.5)
        self.assertEqual(a_xpu, a_xpu.cpu())

    def test_linalg_vector_norm_float16(self, dtype=torch.float16):
        a = torch.randn(398, dtype=dtype)
        b = a.reshape((199, 2))
        a_cpu = torch.linalg.vector_norm(a, ord=3.5)

        a_xpu = torch.linalg.vector_norm(a.to("xpu"), ord=3.5)
        self.assertEqual(a_xpu, a_xpu.cpu())
