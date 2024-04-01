import torch
import intel_extension_for_pytorch  # noqa
from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):
    def test_fork_rng(self):
        try:
            with torch.random.fork_rng(devices=["xpu:0"], device_type='xpu'):
                pass
        except Exception:
            raise AssertionError("false")
