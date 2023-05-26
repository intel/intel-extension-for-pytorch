import unittest
import itertools
import torch

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
    def __init__(self, input_size, hidden_size, num_layers):
        super(Lstm, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
        )

    def forward(self, x, h=None):
        x, h = self.lstm(x, h)
        return x, h


from typing import List


def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("in compiler backend")
    print(gm)
    return gm.forward


class TestCompileCases(TestCase):
    def test_conv_inference(self):
        for dim in [1, 2, 3]:
            input_shapes = {1: (224,), 2: (224, 224), 3: (55, 55, 55)}
            # TODO: add bfloat16 data type tests when 'inductor' backend supports bfloat16.
            options = itertools.product(
                [True, False],
                [1, 2],
                [1, 4],
                [torch.float32],
                ["ipex", "inductor"],
                [True, False],
                [True, False],
            )
            for (
                bias,
                dilation,
                groups,
                dtype,
                backend,
                dynamic,
                ipex_optimize,
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
                with torch.cpu.amp.autocast(
                    enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
                ), torch.no_grad():
                    ori_y = model(x)
                if ipex_optimize:
                    ipex.enable_auto_channels_last()
                    model = ipex.optimize(model, dtype=dtype)
                torch._dynamo.reset()
                compile_model = torch.compile(model, dynamic=dynamic, backend=backend)
                with torch.cpu.amp.autocast(
                    enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
                ), torch.no_grad():
                    y = compile_model(x)
                self.assertEqual(y, ori_y)
                self.assertTrue(y.dtype == dtype)

    def test_deconv_inference(self):
        for dim in [2, 3]:
            input_shapes = {2: (12, 12), 3: (12, 12, 12)}
            input_channel_per_group = 15
            output_channel_per_group = 3
            kernel_size = 3
            if dim == 2:
                channels_last = torch.channels_last
            else:
                channels_last = torch.channels_last_3d
            # TODO: add bfloat16 data type tests when 'inductor' backend supports bfloat16.
            options = itertools.product(
                [True, False],
                [1, 2],
                [1, 2],
                [1, 2],
                [1, 2],
                [torch.contiguous_format, channels_last],
                [torch.float32],
                ["ipex", "inductor"],
                [True, False],
                [True, False],
            )
            for (
                bias,
                stride,
                padding,
                groups,
                dilation,
                memory_format,
                dtype,
                backend,
                dynamic,
                ipex_optimize,
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
                with torch.cpu.amp.autocast(
                    enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
                ), torch.no_grad():
                    ori_y = model(x)
                if ipex_optimize:
                    model = ipex.optimize(model, dtype=dtype)
                torch._dynamo.reset()
                compile_model = torch.compile(model, dynamic=dynamic, backend=backend)
                with torch.cpu.amp.autocast(
                    enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
                ), torch.no_grad():
                    y = compile_model(x)
                self.assertEqual(y, ori_y)
                self.assertTrue(y.dtype == dtype)

    def test_linear_inference(self):
        out_features = torch.randint(3, 10, (1,)).item()
        in_features = torch.randint(3, 10, (1,)).item()

        input_shapes = [(8, in_features), (2, 4, in_features), (2, 2, 2, in_features)]
        # TODO: add bfloat16 data type tests when 'inductor' backend supports bfloat16.
        options = itertools.product(
            [True, False],
            input_shapes,
            [torch.float32],
            ["ipex", "inductor"],
            [True, False],
            [True, False],
        )
        for bias, x_shape, dtype, backend, dynamic, ipex_optimize in options:
            x = torch.randn(x_shape, dtype=torch.float32)
            model = Linear(in_features, out_features, bias).eval()
            with torch.cpu.amp.autocast(
                enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
            ), torch.no_grad():
                ori_y = model(x)
            if ipex_optimize:
                model = ipex.optimize(model, dtype=dtype)
            torch._dynamo.reset()
            compile_model = torch.compile(model, dynamic=dynamic, backend=backend)
            with torch.cpu.amp.autocast(
                enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
            ), torch.no_grad():
                y = compile_model(x)
            self.assertEqual(y, ori_y)
            self.assertTrue(y.dtype == dtype)

    def test_lstm_inference(self):
        # TODO: add bfloat16 data type tests when 'inductor' backend supports bfloat16.
        options = itertools.product(
            [torch.float32], ["ipex", "inductor"], [True, False], [True, False]
        )
        for dtype, backend, dynamic, ipex_optimize in options:
            input = torch.randn(5, 3, 10)
            h0 = torch.randn(2, 3, 20)
            c0 = torch.randn(2, 3, 20)
            model = Lstm(10, 20, 2).eval()
            with torch.cpu.amp.autocast(
                enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
            ), torch.no_grad():
                ori_output, (ori_hn, ori_cn) = model(input, (h0, c0))
            if ipex_optimize:
                model = ipex.optimize(model, dtype=dtype)
            torch._dynamo.reset()
            compile_model = torch.compile(model, dynamic=dynamic, backend=backend)
            with torch.cpu.amp.autocast(
                enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
            ), torch.no_grad():
                output, (hn, cn) = compile_model(input, (h0, c0))
            self.assertEqual(ori_output, output)
            self.assertEqual(ori_hn, hn)
            self.assertEqual(ori_cn, cn)
            self.assertTrue(output.dtype == dtype)
            self.assertTrue(hn.dtype == dtype)
            self.assertTrue(cn.dtype == dtype)


if __name__ == "__main__":
    torch.manual_seed(2020)
    test = unittest.main()
