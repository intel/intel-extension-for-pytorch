import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(
        torch.xpu.device_count() <= 1,
        reason="only one device detected",
    )
    def test_rng_state_offset(self):
        before = torch.xpu.get_rng_state()
        torch.xpu._set_rng_state_offset(100)
        offset = torch.xpu._get_rng_state_offset()
        torch.xpu.set_rng_state(before)
        self.assertEqual(offset, 100)
