import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest


class TestNNMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_TensorFactories(self, dtype=torch.float):

        x = torch.empty_strided((2, 3), (1, 2))
        x_out = torch.empty_strided((2, 3), (1, 2), device="xpu")
        y = torch.eye(3)
        y_out = torch.eye(3, device="xpu")
        m = torch.tril_indices(3, 3)
        n = torch.triu_indices(3, 3)
        m_out = torch.tril_indices(3, 3, device="xpu")
        n_out = torch.triu_indices(3, 3, device="xpu")

        self.assertEqual(x.size(), x_out.size())
        self.assertEqual(x.stride(), x_out.stride())
        self.assertEqual(y, y_out.cpu())
        self.assertEqual(m, m_out.cpu())
        self.assertEqual(n, n_out.cpu())
