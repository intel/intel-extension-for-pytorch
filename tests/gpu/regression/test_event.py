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

    def test_event_elapsed_time(self):
        import torch
        import intel_extension_for_pytorch
        t1 = torch.rand(1024, 1024).to("xpu")
        t2 = torch.rand(1024, 1024).to("xpu")
        torch.xpu.synchronize()
        start_event = torch.xpu.Event(enable_timing=True)
        start_event.record()
        t2 = t1 * t2
        t1 = t1 + t2
        end_event = torch.xpu.Event(enable_timing=True)
        end_event.record()
        end_event.synchronize()
        t = start_event.elapsed_time(end_event)
        print(t)
        self.assertTrue(t > 0)
