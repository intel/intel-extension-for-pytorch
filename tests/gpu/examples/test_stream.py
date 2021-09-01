import torch
import ipex
from torch.testing._internal.common_utils import TestCase
import pytest


class TestStream(TestCase):
    def test_multi_stream_and_dependence(self, dtype=torch.half):
        a = torch.randn(4096, 4096)
        b = torch.randn(4096, 4096)

        stream1 = torch.xpu.current_stream()
        stream2 = torch.xpu.Stream(torch.xpu.current_device())

        a_xpu = a.to("xpu")
        x = torch.tanh(a_xpu)
        with torch.xpu.stream(stream2):
            ''' barrier1 = stream1.record_event()
                stream2.wait_event(barrier1)
            '''
            stream2.wait_stream(stream1)
            b_xpu = b.to("xpu")
            x_xpu = b_xpu + x
            stream2.synchronize()

        x_ref = b + torch.tanh(a)
        x_xpu = x_xpu.cpu()

        self.assertEqual(x_ref, x_xpu)
        print(x_ref - x_xpu)
