import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import platform


class TestTorchMethod(TestCase):
    def test_q_equal(self, dtype=torch.float):
        zp_vec = [0] if platform.system() == 'Windows' else [0, 2]
        for dtype in [torch.quint8, torch.qint8]:
            for zp in zp_vec:
                scale = 0.4

                a = torch.randn(1, 2, 5, 5)
                b = torch.randn(2, 2, 5, 5)

                q_a = torch.quantize_per_tensor(a, scale, zp, dtype)
                q_b = torch.quantize_per_tensor(b, scale, zp, dtype)

                self.assertEqual(q_a.equal(q_b), 0)
                self.assertEqual(q_a.equal(q_a), 1)
