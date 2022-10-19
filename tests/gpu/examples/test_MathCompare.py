import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_math_compare(self, dtype=torch.float):

        x = torch.tensor([[True, True], [True, True]])
        y = torch.tensor([[1, 2], [3, 4]])
        z = torch.tensor([[1, 1], [4, 4]])
        x_dpcpp = x.to("xpu")
        y_dpcpp = y.to("xpu")
        z_dpcpp = z.to("xpu")
        x = torch.lt(y, z)
        x2 = torch.eq(y, z)
        x3 = torch.gt(y, z)
        x4 = torch.le(y, z)
        x5 = torch.ne(y, z)
        x6 = torch.ge(y, z)

        x_out = torch.lt(y_dpcpp, z_dpcpp)
        x2_out = torch.eq(y_dpcpp, z_dpcpp)
        x3_out = torch.gt(y_dpcpp, z_dpcpp)
        x4_out = torch.le(y_dpcpp, z_dpcpp)
        x5_out = torch.ne(y_dpcpp, z_dpcpp)
        x6_out = torch.ge(y_dpcpp, z_dpcpp)

        self.assertEqual(x, x_out.cpu())
        self.assertEqual(x2, x2_out.cpu())
        self.assertEqual(x3, x3_out.cpu())
        self.assertEqual(x4, x4_out.cpu())
        self.assertEqual(x5, x5_out.cpu())
        self.assertEqual(x6, x6_out.cpu())
        self.assertEqual(y.lt_(z), y_dpcpp.lt_(z_dpcpp).cpu())
        self.assertEqual(y.eq_(z), y_dpcpp.eq_(z_dpcpp).cpu())
        self.assertEqual(y.gt_(z), y_dpcpp.gt_(z_dpcpp).cpu())
        self.assertEqual(y.le_(z), y_dpcpp.le_(z_dpcpp).cpu())
        self.assertEqual(y.ne_(z), y_dpcpp.ne_(z_dpcpp).cpu())
        self.assertEqual(y.ge_(z), y_dpcpp.ge_(z_dpcpp).cpu())
