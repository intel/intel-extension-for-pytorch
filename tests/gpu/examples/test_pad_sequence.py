import torch
from torch.nn import functional as F
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_pad_sequence(self, dtype=torch.float):
        for batch_first in [True]:
            a = torch.randn((2, 5, 2), device=cpu_device, dtype=dtype)
            b = torch.randn((3, 5, 2), device=cpu_device, dtype=dtype)
            c = torch.randn((4, 5, 2), device=cpu_device, dtype=dtype)
            output = torch.nn.utils.rnn.pad_sequence(
                (a, b, c), batch_first, padding_value=91
            )

            a_xpu = a.to(dpcpp_device)
            b_xpu = b.to(dpcpp_device)
            c_xpu = c.to(dpcpp_device)
            output_xpu = torch.nn.utils.rnn.pad_sequence(
                (a_xpu, b_xpu, c_xpu), batch_first, padding_value=91
            )

        self.assertEqual(output, output_xpu.cpu())
