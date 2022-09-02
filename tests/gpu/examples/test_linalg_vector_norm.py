import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

class TestNNMethod(TestCase):
    def test_linalg_vector_norm(self):
        a = torch.arange(9, dtype=torch.float) - 4
        b = a.reshape((3, 3))
        a_cpu = torch.linalg.vector_norm(a, ord=3.5)
        b_cpu = torch.linalg.vector_norm(b, ord=3.5)
        self.assertEqual(a_cpu, b_cpu)

        a_xpu = torch.linalg.vector_norm(a.to('xpu'), ord=3.5)
        b_xpu = torch.linalg.vector_norm(b.to('xpu'), ord=3.5)
        self.assertEqual(a_xpu, b_xpu)
        self.assertEqual(a_cpu, b_xpu)
        print('a_cpu', a_cpu)
        print('b_cpu', b_cpu)
        print('a_xpu', a_xpu)
        print('a_xpu', a_xpu)
