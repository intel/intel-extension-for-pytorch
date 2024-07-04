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


class M(nn.Module):
    def __init__(self, in_channels, out_channels, unary_fn, **kwargs):
        super(M, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, **kwargs)
        self.unary_fn = unary_fn

    def forward(self, x):
        x = self.conv(x)
        x = self.unary_fn(x)
        return x


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
    def test_inductor_fusion_conv(self):
        called = False
        device = dpcpp_device
        for unary_fn in unary_list:
            for dynam in [True, False]:
                # device=cpu_device
                model = M(
                    3, 3, unary_fn, kernel_size=3, stride=1, padding=1, bias=False
                ).to(device)

                model.eval()
                with torch.no_grad():
                    # with torch.xpu.onednn_verbose(2):
                    run = torch.compile(model, backend="ipex", dynamic=dynam)
                    torch.manual_seed(0)
                    example_input = torch.randn(1, 3, 72, 72).to(device)

                    actual = run(example_input)
                    ref = model(example_input)
                    self.assertEqual(ref, actual)

    @pytest.mark.skipif(
        platform.system() == "Windows" or "WSL2" in platform.uname().release,
        reason="Windows not yet supported for torch.compile",
    )
    @config.patch({"freezing": True})
    def test_inductor_fusion_linear(self):
        called = False
        device = dpcpp_device
        for unary_fn in unary_list:
            model = N(3, 4, unary_fn, bias=False).to(device)

            model.eval()
            with torch.no_grad():
                # with torch.xpu.onednn_verbose(2):
                run = torch.compile(model, backend="ipex")
                torch.manual_seed(0)
                example_input = torch.randn(3, 3).to(device)
                actual = run(example_input)

                ref = model(example_input)
                self.assertEqual(ref, actual)

    @pytest.mark.skipif(
        platform.system() == "Windows" or "WSL2" in platform.uname().release,
        reason="Windows not yet supported for torch.compile",
    )
    @config.patch({"freezing": True})
    def test_conv_binary_fusion(self):
        class M(torch.nn.Module):
            def __init__(self, binary_fn, has_relu, **kwargs):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                self.binary_fn = binary_fn
                self.has_relu = has_relu

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                if self.has_relu:
                    return self.binary_fn(x1, x2).relu()
                else:
                    return self.binary_fn(x1, x2)

        with torch.no_grad():
            for binary_fn in binary_list:
                model = M(binary_fn, False).to("xpu")
                x = torch.rand([2, 3, 10, 10]).to("xpu")
                run = torch.compile(model, backend="ipex")
                print("Run compiled fn")
                actual = run(x)
                print("Run imperative fn")
                ref = model(x)
                self.assertEqual(actual, ref)

    @pytest.mark.skipif(
        platform.system() == "Windows" or "WSL2" in platform.uname().release,
        reason="Windows not yet supported for torch.compile",
    )
    @config.patch({"freezing": True})
    def test_conv_binary_inplace_fusion(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x, other):
                conv_out = self.conv(x)
                return torch.add(conv_out, other).relu()

        inputs = [
            torch.randn(1, 3, 28, 28).to("xpu"),
            torch.randn(1, 32, 28, 28).to("xpu"),
        ]
        model = M().to("xpu")
        with torch.no_grad():
            run = torch.compile(model, backend="ipex")
            actual = run(inputs[0], inputs[1])
            ref = model(inputs[0], inputs[1])
            self.assertEqual(actual, ref)

    @pytest.mark.skipif(
        platform.system() == "Windows" or "WSL2" in platform.uname().release,
        reason="Windows not yet supported for torch.compile",
    )
    @config.patch({"freezing": True})
    def test_conv_unary_fusion(self):
        class M(torch.nn.Module):
            def __init__(self, unary_fn, **kwargs):
                super().__init__()
                self.conv = torch.nn.Conv2d(9, 8, kernel_size=3, stride=1, padding=1)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.relu(x)
                x = self.conv(x)
                x = self.relu(x)

                return x

        with torch.no_grad():
            for unary_fn in unary_list:
                model = M(unary_fn).to("xpu")
                x = torch.rand([2, 9, 10, 10]).to("xpu")
                run = torch.compile(model, backend="ipex")
                print("Run compiled fn")
                actual = run(x)
                print("Run imperative fn")
                ref = model(x)
                self.assertEqual(actual, ref)

    @pytest.mark.skipif(
        platform.system() == "Windows" or "WSL2" in platform.uname().release,
        reason="Windows not yet supported for torch.compile",
    )
    @config.patch({"freezing": True})
    def test_linear_binary_fusion(self):
        class M(torch.nn.Module):
            def __init__(self, binary_fn, in_channels, out_channels, bias, **kwargs):
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_channels, out_channels, bias=bias, **kwargs
                )
                self.binary_fn = binary_fn

            def forward(self, x, y):
                x = self.linear(x)
                x = self.binary_fn(x, y.clone())
                return x

        out_feature = 30
        in_feature = 14
        for binary_fn in binary_list:
            with torch.no_grad():
                model = M(binary_fn, in_feature, out_feature, False).eval().xpu()
                input = torch.randn((3, in_feature)).xpu()
                other = torch.randn((3, out_feature)).xpu()
                ref = model(input, other)
                model_compiled = torch.compile(model, backend="ipex")
                actual = model_compiled(input, other)
                self.assertEqual(actual, ref)
