import unittest
import itertools
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
    FakeTensorConverter,
)

import intel_extension_for_pytorch as ipex

from common_utils import TestCase

conv_module = {1: torch.nn.Conv1d, 2 : torch.nn.Conv2d, 3 : torch.nn.Conv3d}
convtranspose_module = {2 : torch.nn.ConvTranspose2d, 3 : torch.nn.ConvTranspose3d}

class ConvNd(torch.nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size, stride, padding, dilation, bias, groups):
        super(ConvNd, self).__init__()
        self.conv = conv_module[dim](in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)

    def forward(self, x):
        return self.conv(x)

class Linear(torch.nn.Module):
    def __init__(self, in_f, out_f, bias):
        super(Linear, self).__init__()
        self.linear = torch.nn.Linear(in_f, out_f, bias=bias)

    def forward(self, x):
        return self.linear(x)

class DeconvNd(torch.nn.Module):
    def __init__(self, dim, ic, oc, kernel_size, stride, padding, groups, bias, dilation):
        super(DeconvNd, self).__init__()
        self.deconv = convtranspose_module[dim](ic, oc, kernel_size=kernel_size, stride=stride, \
                                               padding=padding, groups=groups, bias=bias, dilation=dilation)

    def forward(self, x):
        return self.deconv(x)

class TestFakeCases(TestCase):
    def test_conv_inference(self):
        for dim in [1, 2, 3]:
            input_shapes = {1: (224,), 2: (224, 224), 3: (55, 55, 55)}
            if dim == 2:
                channels_last = torch.channels_last
            elif dim == 3:
                channels_last = torch.channels_last_3d
            if dim == 1:
                options = itertools.product([True, False], [1, 2], [1, 4], [True, False], [torch.contiguous_format], [torch.float32, torch.bfloat16])
            else:
                options = itertools.product([True, False], [1, 2], [1, 4], [True, False], [torch.contiguous_format, channels_last], [torch.float32, torch.bfloat16])
            for bias, dilation, groups, feed_sample_input, memory_format, dtype in options:
                N = torch.randint(1, 10, (1,)).item()
                M = torch.randint(1, 3, (1,)).item() * groups
                C = torch.randint(1, 3, (1,)).item() * groups
                x_shape = (N, C) + input_shapes[dim]
                x = torch.randn(x_shape, dtype=torch.float32)
                model = ConvNd(
                    dim=dim,
                    in_channels=C,
                    out_channels=M,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    dilation=dilation,
                    bias=bias,
                    groups=groups).eval()
                model = model.to(memory_format=memory_format)
                x = x.to(memory_format=memory_format)
                if feed_sample_input:
                    ipex_model = ipex.optimize(model, dtype=dtype, level='O1', sample_input=x)
                else:
                    ipex_model = ipex.optimize(model, dtype=dtype, level='O1')
                with torch.cpu.amp.autocast(enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16), torch.no_grad():
                    y = ipex_model(x)
                mode = FakeTensorMode(allow_fallback_kernels=False)
                with torch._subclasses.fake_tensor.FakeCopyMode(mode):
                    ipex_model_fake = copy.deepcopy(ipex_model)
                with mode:
                    x_fake = mode.from_tensor(x)
                    with torch.cpu.amp.autocast(enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16), torch.no_grad():
                        y_fake = ipex_model_fake(x_fake)
                    self.assertTrue(isinstance(x_fake, FakeTensor))
                    self.assertTrue(isinstance(y_fake, FakeTensor))
                    self.assertTrue(y_fake.size() == y.size())
                    self.assertTrue(y_fake.dtype == dtype)

    def test_linear_inference(self):
        out_features = torch.randint(3, 10, (1,)).item()
        in_features = torch.randint(3, 10, (1,)).item()

        input_shapes = [(8, in_features), (2, 4, in_features), (2, 2, 2, in_features)]
        options = itertools.product([True, False], input_shapes, [True, False], [True, False], [torch.float32, torch.bfloat16])
        for bias, x_shape, feed_sample_input, auto_kernel_selection, dtype in options:
            x = torch.randn(x_shape, dtype=torch.float32)
            model = Linear(in_features, out_features, bias).eval()
            if feed_sample_input:
                ipex_model = ipex.optimize(model, dtype=dtype, level='O1', auto_kernel_selection=auto_kernel_selection, sample_input=x)
            else:
                ipex_model = ipex.optimize(model, dtype=dtype, auto_kernel_selection=auto_kernel_selection, level='O1')
            with torch.cpu.amp.autocast(enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16), torch.no_grad():
                y = ipex_model(x)
            mode = FakeTensorMode(allow_fallback_kernels=False)
            with torch._subclasses.fake_tensor.FakeCopyMode(mode):
                ipex_model_fake = copy.deepcopy(ipex_model)
            with mode:
                x_fake = mode.from_tensor(x)
                with torch.cpu.amp.autocast(enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16), torch.no_grad():
                    y_fake = ipex_model_fake(x_fake)
                self.assertTrue(isinstance(x_fake, FakeTensor))
                self.assertTrue(isinstance(y_fake, FakeTensor))
                self.assertTrue(y_fake.size() == y.size())
                self.assertTrue(y_fake.dtype == dtype)

    def test_deconv_inference(self):
        for dim in [2, 3]:
            input_shapes = {2: (12, 12), 3: (12, 12, 12)}
            if dim == 2:
                channels_last = torch.channels_last
            else:
                channels_last = torch.channels_last_3d
            input_channel_per_group = 15
            output_channel_per_group = 3
            kernel_size = 3 
            options = itertools.product([True, False], [1, 2], [1, 2], [1, 2], [1, 2], [True, False], [torch.contiguous_format, channels_last], [torch.float32, torch.bfloat16])
            for bias, stride, padding, groups, dilation, feed_sample_input, memory_format, dtype in options:
                ic = input_channel_per_group * groups
                oc = output_channel_per_group * groups
                x_shape = (2, ic) + input_shapes[dim]
                x = torch.randn(x_shape, dtype=torch.float32)
                model = DeconvNd(dim, ic, oc, kernel_size, stride, padding, groups, bias, dilation).eval()
                model = model.to(memory_format=memory_format)
                x = x.to(memory_format=memory_format)
                if feed_sample_input:
                    ipex_model = ipex.optimize(model, dtype=dtype, level='O1', sample_input=x)
                else:
                    ipex_model = ipex.optimize(model, dtype=dtype, level='O1')
                with torch.cpu.amp.autocast(enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16), torch.no_grad():
                    y = ipex_model(x)
                mode = FakeTensorMode(allow_fallback_kernels=False)
                with torch._subclasses.fake_tensor.FakeCopyMode(mode):
                    ipex_model_fake = copy.deepcopy(ipex_model)
                with mode:
                    x_fake = mode.from_tensor(x)
                    with torch.cpu.amp.autocast(enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16), torch.no_grad():
                        y_fake = ipex_model_fake(x_fake)
                    self.assertTrue(isinstance(x_fake, FakeTensor))
                    self.assertTrue(isinstance(y_fake, FakeTensor))
                    self.assertTrue(y_fake.size() == y.size())
                    self.assertTrue(y_fake.dtype == dtype)
                

if __name__ == '__main__':
    torch.manual_seed(2020)
    test = unittest.main()

