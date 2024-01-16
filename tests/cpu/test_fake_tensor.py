import unittest
import itertools
import copy
import torch

from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
)

import intel_extension_for_pytorch as ipex

from common_utils import TestCase

conv_module = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
convtranspose_module = {2: torch.nn.ConvTranspose2d, 3: torch.nn.ConvTranspose3d}


class ConvNd(torch.nn.Module):
    def __init__(
        self,
        dim,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        bias,
        groups,
    ):
        super(ConvNd, self).__init__()
        self.conv = conv_module[dim](
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )

    def forward(self, x):
        return self.conv(x)


class Linear(torch.nn.Module):
    def __init__(self, in_f, out_f, bias):
        super(Linear, self).__init__()
        self.linear = torch.nn.Linear(in_f, out_f, bias=bias)

    def forward(self, x):
        return self.linear(x)


class DeconvNd(torch.nn.Module):
    def __init__(
        self, dim, ic, oc, kernel_size, stride, padding, groups, bias, dilation
    ):
        super(DeconvNd, self).__init__()
        self.deconv = convtranspose_module[dim](
            ic,
            oc,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )

    def forward(self, x):
        return self.deconv(x)


class Lstm(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        bias,
        dropout,
        batch_first,
    ):
        super(Lstm, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            bias=bias,
            dropout=dropout,
            batch_first=batch_first,
        )

    def forward(self, x, h=None):
        x, h = self.lstm(x, h)
        return x, h


class TestFakeCases(TestCase):
    def test_conv_inference(self):
        for dim in [1, 2, 3]:
            input_shapes = {1: (224,), 2: (224, 224), 3: (55, 55, 55)}
            if dim == 2:
                channels_last = torch.channels_last
            elif dim == 3:
                channels_last = torch.channels_last_3d
            if dim == 1:
                options = itertools.product(
                    [True, False],
                    [1, 2],
                    [1, 4],
                    [True, False],
                    [torch.contiguous_format],
                    [torch.float32, torch.bfloat16],
                )
            else:
                options = itertools.product(
                    [True, False],
                    [1, 2],
                    [1, 4],
                    [True, False],
                    [torch.contiguous_format, channels_last],
                    [torch.float32, torch.bfloat16],
                )
            for (
                bias,
                dilation,
                groups,
                feed_sample_input,
                memory_format,
                dtype,
            ) in options:
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
                    groups=groups,
                ).eval()
                model = model.to(memory_format=memory_format)
                x = x.to(memory_format=memory_format)
                if feed_sample_input:
                    ipex_model = ipex.optimize(
                        model, dtype=dtype, level="O1", sample_input=x
                    )
                else:
                    ipex_model = ipex.optimize(model, dtype=dtype, level="O1")
                with torch.cpu.amp.autocast(
                    enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
                ), torch.no_grad():
                    y = ipex_model(x)
                mode = FakeTensorMode(allow_fallback_kernels=False)
                with torch._subclasses.fake_tensor.FakeCopyMode(mode):
                    ipex_model_fake = copy.deepcopy(ipex_model)
                with mode:
                    x_fake = mode.from_tensor(x)
                    with torch.cpu.amp.autocast(
                        enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
                    ), torch.no_grad():
                        y_fake = ipex_model_fake(x_fake)
                    self.assertTrue(isinstance(x_fake, FakeTensor))
                    self.assertTrue(isinstance(y_fake, FakeTensor))
                    self.assertTrue(y_fake.size() == y.size())
                    self.assertTrue(y_fake.dtype == dtype)

    def test_linear_inference(self):
        out_features = torch.randint(3, 10, (1,)).item()
        in_features = torch.randint(3, 10, (1,)).item()

        input_shapes = [(8, in_features), (2, 4, in_features), (2, 2, 2, in_features)]
        options = itertools.product(
            [True, False],
            input_shapes,
            [True, False],
            [True, False],
            [torch.float32, torch.bfloat16],
        )
        for bias, x_shape, feed_sample_input, auto_kernel_selection, dtype in options:
            x = torch.randn(x_shape, dtype=torch.float32)
            model = Linear(in_features, out_features, bias).eval()
            if feed_sample_input:
                ipex_model = ipex.optimize(
                    model,
                    dtype=dtype,
                    level="O1",
                    auto_kernel_selection=auto_kernel_selection,
                    sample_input=x,
                )
            else:
                ipex_model = ipex.optimize(
                    model,
                    dtype=dtype,
                    auto_kernel_selection=auto_kernel_selection,
                    level="O1",
                )
            with torch.cpu.amp.autocast(
                enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
            ), torch.no_grad():
                y = ipex_model(x)
            mode = FakeTensorMode(allow_fallback_kernels=False)
            with torch._subclasses.fake_tensor.FakeCopyMode(mode):
                ipex_model_fake = copy.deepcopy(ipex_model)
            with mode:
                x_fake = mode.from_tensor(x)
                with torch.cpu.amp.autocast(
                    enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
                ), torch.no_grad():
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
            options = itertools.product(
                [True, False],
                [1, 2],
                [1, 2],
                [1, 2],
                [1, 2],
                [True, False],
                [torch.contiguous_format, channels_last],
                [torch.float32, torch.bfloat16],
            )
            for (
                bias,
                stride,
                padding,
                groups,
                dilation,
                feed_sample_input,
                memory_format,
                dtype,
            ) in options:
                ic = input_channel_per_group * groups
                oc = output_channel_per_group * groups
                x_shape = (2, ic) + input_shapes[dim]
                x = torch.randn(x_shape, dtype=torch.float32)
                model = DeconvNd(
                    dim, ic, oc, kernel_size, stride, padding, groups, bias, dilation
                ).eval()
                model = model.to(memory_format=memory_format)
                x = x.to(memory_format=memory_format)
                if feed_sample_input:
                    ipex_model = ipex.optimize(
                        model, dtype=dtype, level="O1", sample_input=x
                    )
                else:
                    ipex_model = ipex.optimize(model, dtype=dtype, level="O1")
                with torch.cpu.amp.autocast(
                    enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
                ), torch.no_grad():
                    y = ipex_model(x)
                mode = FakeTensorMode(allow_fallback_kernels=False)
                with torch._subclasses.fake_tensor.FakeCopyMode(mode):
                    ipex_model_fake = copy.deepcopy(ipex_model)
                with mode:
                    x_fake = mode.from_tensor(x)
                    with torch.cpu.amp.autocast(
                        enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
                    ), torch.no_grad():
                        y_fake = ipex_model_fake(x_fake)
                    self.assertTrue(isinstance(x_fake, FakeTensor))
                    self.assertTrue(isinstance(y_fake, FakeTensor))
                    self.assertTrue(y_fake.size() == y.size())
                    self.assertTrue(y_fake.dtype == dtype)

    def _lstm_params_list(self):
        params_dict = {
            "input_size": [1, 2],
            "hidden_size": [2],
            "num_layers": [1, 2],
            "bidirectional": [False, True],
            "bias": [False, True],
            "empty_state": [False, True],
            "batch_first": [False, True],
            "dropout": [0, 0.4, 1],
            "batch_size": [1, 2],
            "seq_len": [1, 2],
        }

        params_list = []
        for key, value in params_dict.items():
            params_list.append(value)
        return params_list

    def test_lstm_inference(self):
        params_list = self._lstm_params_list()
        for (
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            bias,
            empty_state,
            batch_first,
            dropout,
            batch_size,
            seq_len,
        ) in itertools.product(*params_list):
            # dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1
            if dropout > 0 and num_layers == 1:
                continue

            num_directions = 2 if bidirectional else 1

            if batch_first:
                x = torch.randn(batch_size, seq_len, input_size)
            else:
                x = torch.randn(seq_len, batch_size, input_size)
            h = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            c = torch.randn(num_layers * num_directions, batch_size, hidden_size)

            model = Lstm(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first,
            ).eval()

            for dtype in [torch.float32, torch.bfloat16]:
                ipex_model = ipex.optimize(model, dtype=dtype, level="O1")
                with torch.cpu.amp.autocast(
                    enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
                ), torch.no_grad():
                    if empty_state:
                        y, hy = ipex_model(x)
                    else:
                        y, hy = ipex_model(x, (h, c))
                mode = FakeTensorMode(allow_fallback_kernels=False)
                with torch._subclasses.fake_tensor.FakeCopyMode(mode):
                    ipex_model_fake = copy.deepcopy(ipex_model)
                with mode:
                    x_fake = mode.from_tensor(x)
                    h_fake = mode.from_tensor(h)
                    c_fake = mode.from_tensor(c)
                    with torch.cpu.amp.autocast(
                        enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
                    ), torch.no_grad():
                        if empty_state:
                            y_fake, hy_fake = ipex_model_fake(x_fake)
                        else:
                            y_fake, hy_fake = ipex_model_fake(x_fake, (h_fake, c_fake))
                    self.assertTrue(isinstance(x_fake, FakeTensor))
                    self.assertTrue(isinstance(y_fake, FakeTensor))
                    self.assertTrue(isinstance(hy_fake[0], FakeTensor))
                    self.assertTrue(isinstance(hy_fake[1], FakeTensor))
                    self.assertTrue(y_fake.size() == y.size())
                    self.assertTrue(hy_fake[0].size() == hy[0].size())
                    self.assertTrue(hy_fake[1].size() == hy[1].size())
                    self.assertTrue(y_fake.dtype == dtype)
                    self.assertTrue(hy_fake[0].dtype == dtype)
                    self.assertTrue(hy_fake[1].dtype == dtype)


if __name__ == "__main__":
    torch.manual_seed(2020)
    test = unittest.main()
