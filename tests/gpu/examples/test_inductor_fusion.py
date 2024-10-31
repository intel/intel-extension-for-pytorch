import torch
import torch.nn as nn
import torch._dynamo
from torch._inductor import config

from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import numpy as np
import pytest
import platform

np.set_printoptions(threshold=np.inf)

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


# The dict value is match_nodes(computation_op+unary_op)
# See pytorch/test/inductor/test_mkldnn_pattern_matcher.py
unary_list = {
    torch.nn.ReLU(): 2,
    torch.nn.Sigmoid(): 2,
    torch.nn.Tanh(): 2,
    torch.nn.Hardswish(): 6,
    torch.nn.LeakyReLU(0.1, inplace=False): 4,
    torch.nn.Hardtanh(min_val=-0.5, max_val=4, inplace=False): 3,
    torch.nn.Hardtanh(min_val=-0.5, max_val=float("inf"), inplace=False): 3,
    torch.nn.GELU(approximate="none"): 6,
    torch.nn.GELU(approximate="tanh"): 10,
    torch.nn.ReLU6(): 3,
    torch.nn.SiLU(): 3,
    torch.nn.Hardsigmoid(): 5,
}

# The dict value is (match_count, match_nodes, inplace)
# See pytorch/test/inductor/test_mkldnn_pattern_matcher.py
binary_list = {
    lambda x, y: torch.add(x, y): (1, 2, False),  # call_function
    lambda x, y: torch.add(y, x): (1, 2, False),
    lambda x, y: x.add(y): (1, 2, False),  # call_method
    lambda x, y: x.add_(y): (1, 2, False),
    lambda x, y: torch.sub(x, y): (1, 2, False),  # call_function
    lambda x, y: x.sub(y): (1, 2, False),  # call_method
    lambda x, y: x.sub_(y): (1, 2, True),  # call_method
}


class N(nn.Module):
    def __init__(self, in_channels, out_channels, unary_fn, **kwargs):
        super(N, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels, **kwargs)
        self.unary_fn = unary_fn

    def forward(self, x):
        x = self.linear(x)
        x = self.unary_fn(x)
        return x


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(
        platform.system() == "Windows" or "WSL2" in platform.uname().release,
        reason="Windows not yet supported for torch.compile",
    )
    @config.patch({"freezing": True})
    def test_inductor_fusion_linear(self):
        called = False
        device = dpcpp_device
        for unary_fn in unary_list:
            for dynam in [True, False]:
                model = N(3, 4, unary_fn, bias=False).to(device)

                model.eval()
                with torch.no_grad():
                    # with torch.xpu.onednn_verbose(2):
                    run = torch.compile(model, backend="inductor", dynamic=dynam)
                    torch.manual_seed(0)
                    example_input = torch.randn(3, 3).to(device)
                    actual = run(example_input)

                    ref = model(example_input)
                    self.assertEqual(ref, actual)
