import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


class TestTorchMethod(TestCase):
    def test_device_guard(self):
        target_device = torch.xpu.current_device()
        exchange_device = (target_device + 1) % torch.xpu.device_count()
        with torch.xpu._DeviceGuard(exchange_device):
            self.assertEqual(torch.xpu.current_device(), exchange_device)
        self.assertEqual(torch.xpu.current_device(), target_device)
