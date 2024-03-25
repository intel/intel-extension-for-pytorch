import torch
import intel_extension_for_pytorch  # noqa

from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):
    def test_deterministic_gemm(self, dtype=torch.float16):
        M = 512
        N = 50257
        K = 126
        r1 = -2  # lower bound of value
        r2 = 2  # upper bound of value
        a = torch.FloatTensor(M, N).uniform_(r1, r2).to(torch.float16).to("xpu")
        b = torch.FloatTensor(N, K).uniform_(r1, r2).to(torch.float16).to("xpu")

        torch.use_deterministic_algorithms(True)

        first_result = torch.matmul(a, b)
        second_result = torch.matmul(a, b)

        torch.use_deterministic_algorithms(False)

        self.assertEqual(first_result, second_result, atol=0, rtol=0)
