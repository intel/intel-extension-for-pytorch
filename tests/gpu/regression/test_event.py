from torch.testing._internal.common_utils import TestCase


class TestTorchXPUMethod(TestCase):
    def test_event_record(self):
        import torch
        import intel_extension_for_pytorch
        ev = torch.xpu.Event(enable_timing=True)
        # before the fix,
        # AttributeError: module 'intel_extension_for_pytorch' has no attribute 'current_stream'
        ev.record()
        ev.wait()
