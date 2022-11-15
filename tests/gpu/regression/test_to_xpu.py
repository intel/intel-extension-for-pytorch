import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(torch.xpu.device_count() == 1, reason="doesn't support with one device")
    def test_to_xpu1(self):
        x = torch.empty([10])
        y = x.to('xpu:1')
        self.assertEqual(y.device.index, 1)
