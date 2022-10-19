import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import numpy as np
import pytest


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_igamma(self, dtype=torch.float):
        a = np.array([[1.6835, 1.8474, 1.1929],
                      [1.0475, 1.7162, 1.4180]])
        data = torch.from_numpy(a)
        a_tensor = data.clone().detach()
        a_dpcpp = a_tensor.to("xpu")

        x = np.array([[1.4845, 1.6588, 1.7829],
                      [1.4753, 1.4286, 1.4260]])
        x_tensor = torch.from_numpy(x)
        x_dpcpp = x_tensor.to("xpu")

        y = torch.igamma(a_tensor, x_tensor)
        y_dpcpp = torch.igamma(a_dpcpp, x_dpcpp)

        self.assertEqual(y, y_dpcpp.cpu())

    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_igammac(self, dtype=torch.float):
        a = np.array([[1.6835, 1.8474, 1.1929],
                      [1.0475, 1.7162, 1.4180]])
        data = torch.from_numpy(a)
        a_tensor = data.clone().detach()
        a_dpcpp = a_tensor.to("xpu")

        x = np.array([[1.4845, 1.6588, 1.7829],
                      [1.4753, 1.4286, 1.4260]])
        x_tensor = torch.from_numpy(x)
        x_dpcpp = x_tensor.to("xpu")

        y = torch.igammac(a_tensor, x_tensor)
        y_dpcpp = torch.igammac(a_dpcpp, x_dpcpp)

        self.assertEqual(y, y_dpcpp.cpu())
