import unittest
import itertools
import copy
import torch
import torch.nn.functional as F
from torch.optim import SGD
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.nn import FrozenBatchNorm2d

from common_utils import TestCase
from test_emb import Embeddingbag
from test_rmsnorm import RMSNorm
from test_masked_mha import MaskedMHATest
from test_roialign import skipIfNoTorchVision, torchvision_fn
from intel_extension_for_pytorch.cpu._auto_kernel_selection import (
    _enable_tpp,
    _disable_tpp,
)
from test_tpp_linear import (
    Linear_with_bias,
    Linear_without_bias,
    Linear_gelu,
    Linear_silu,
    Linear_relu,
    Linear_mul,
    Linear_add,
    Linear_add_add,
)

conv_module = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
convtranspose_module = {
    1: torch.nn.ConvTranspose1d,
    2: torch.nn.ConvTranspose2d,
    3: torch.nn.ConvTranspose3d,
}


class ConvNd_Relu(torch.nn.Module):
    def __init__(
        self,
        dim,
        in_channels,
        out_channels,
        kernel_size,
    ):
        super(ConvNd_Relu, self).__init__()
        self.conv = conv_module[dim](
            in_channels,
            out_channels,
            kernel_size=kernel_size,
        )

    def forward(self, x):
        return F.relu(self.conv(x))


class Linear_Relu(torch.nn.Module):
    def __init__(self, in_f, out_f):
        super(Linear_Relu, self).__init__()
        self.linear = torch.nn.Linear(in_f, out_f)

    def forward(self, x):
        return F.relu(self.linear(x))


class DeconvNd_Relu(torch.nn.Module):
    def __init__(
        self,
        dim,
        ic,
        oc,
        kernel_size,
    ):
        super(DeconvNd_Relu, self).__init__()
        self.deconv = convtranspose_module[dim](
            ic,
            oc,
            kernel_size=kernel_size,
        )

    def forward(self, x):
        return F.relu(self.deconv(x))


class Lstm(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Lstm, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
        )

    def forward(self, x, h=None):
        x, h = self.lstm(x, h)
        return x, h


class BmmAdd(torch.nn.Module):
    def __init__(self):
        super(BmmAdd, self).__init__()

    def forward(self, input, batch1, batch2):
        bmm_res = torch.bmm(batch1, batch2)
        res = torch.add(bmm_res, input)
        return res


class AddSoftmax(torch.nn.Module):
    def __init__(self):
        super(AddSoftmax, self).__init__()

    def forward(self, x1, x2):
        return torch.ops.torch_ipex.add_softmax_(x1, x2)


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = torch.nn.ModuleList()
        self.mlp.append(torch.nn.Linear(10, 10))
        self.mlp.append(torch.nn.ReLU())

    def forward(self, x):
        return self.mlp[1](self.mlp[0](x))


class TestCompileCases(TestCase):
    def test_conv_relu_inference(self):
        for dim in [1, 2, 3]:
            input_shapes = {1: (4,), 2: (4, 4), 3: (4, 4, 4)}
            options = itertools.product(
                [torch.float32, torch.bfloat16],
                ["torchscript", "inductor"],
                [True, False],
                [True, False],
                [True, False],
                [True, False],
            )
            for (
                dtype,
                compiler_backend,
                dynamic,
                ipex_optimize,
                weight_prepack,
                feed_sample_input,
            ) in options:
                if compiler_backend == "torchscript" and dynamic is True:
                    continue
                if weight_prepack is True and ipex_optimize is False:
                    continue
                if feed_sample_input is True and weight_prepack is False:
                    continue
                N = 2
                M = 2
                C = 3
                x_shape = (N, C) + input_shapes[dim]
                x = torch.randn(x_shape, dtype=torch.float32)
                model = ConvNd_Relu(
                    dim=dim,
                    in_channels=C,
                    out_channels=M,
                    kernel_size=3,
                ).eval()
                if ipex_optimize:
                    # TODO: support channels_last_1d.
                    if dim == 1:
                        ipex.disable_auto_channels_last()
                    else:
                        ipex.enable_auto_channels_last()
                    if feed_sample_input:
                        model = ipex.optimize(
                            model,
                            weights_prepack=weight_prepack,
                            dtype=dtype,
                            sample_input=x,
                        )
                    else:
                        model = ipex.optimize(
                            model, weights_prepack=weight_prepack, dtype=dtype
                        )
                torch._dynamo.reset()
                ipex._set_compiler_backend(compiler_backend)
                compile_model = torch.compile(model, dynamic=dynamic, backend="ipex")
                with torch.cpu.amp.autocast(
                    enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
                ), torch.no_grad():
                    ori_y = model(x)
                    for _ in range(3):
                        y = compile_model(x)
                self.assertEqual(y, ori_y)
                self.assertTrue(y.dtype == dtype)

    def test_conv_relu_train(self):
        for dim in [1, 2, 3]:
            input_shapes = {1: (4,), 2: (4, 4), 3: (4, 4, 4)}
            options = itertools.product(
                [torch.float32, torch.bfloat16],
                ["inductor"],
                [True, False],
                [True, False],
                [True, False],
                [True, False],
            )
            for (
                dtype,
                compiler_backend,
                dynamic,
                ipex_optimize,
                weight_prepack,
                feed_sample_input,
            ) in options:
                if weight_prepack is True and ipex_optimize is False:
                    continue
                if feed_sample_input is True and weight_prepack is False:
                    continue
                N = 2
                M = 2
                C = 3
                x_shape = (N, C) + input_shapes[dim]
                input = torch.randn(x_shape, dtype=torch.float32)
                ori_x = input.clone().requires_grad_()
                x = input.clone().requires_grad_()
                sample_x = input.clone().requires_grad_()
                conv = ConvNd_Relu(
                    dim=dim,
                    in_channels=C,
                    out_channels=M,
                    kernel_size=3,
                )
                ori_model = copy.deepcopy(conv).train()
                model = copy.deepcopy(conv).train()
                optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
                if ipex_optimize:
                    # TODO: support channels_last_1d.
                    if dim == 1:
                        ipex.disable_auto_channels_last()
                    else:
                        ipex.enable_auto_channels_last()
                    if feed_sample_input:
                        ori_model, _ = ipex.optimize(
                            ori_model,
                            weights_prepack=weight_prepack,
                            dtype=dtype,
                            optimizer=optimizer,
                            sample_input=sample_x,
                        )
                        model, _ = ipex.optimize(
                            model,
                            weights_prepack=weight_prepack,
                            dtype=dtype,
                            optimizer=optimizer,
                            sample_input=sample_x,
                        )
                    else:
                        ori_model, _ = ipex.optimize(
                            ori_model,
                            weights_prepack=weight_prepack,
                            dtype=dtype,
                            optimizer=optimizer,
                        )
                        model, _ = ipex.optimize(
                            model,
                            weights_prepack=weight_prepack,
                            dtype=dtype,
                            optimizer=optimizer,
                        )
                torch._dynamo.reset()
                ipex._set_compiler_backend(compiler_backend)
                compile_model = torch.compile(model, dynamic=dynamic, backend="ipex")
                with torch.cpu.amp.autocast(
                    enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
                ):
                    ori_y = ori_model(ori_x)
                    y = compile_model(x)
                    grad_x = torch.randn(y.shape, dtype=torch.float32)
                    ori_y.backward(grad_x)
                    y.backward(grad_x)
                    self.assertEqual(y, ori_y)
                    self.assertTrue(y.dtype == dtype)
                    self.assertEqual(x.grad, ori_x.grad)

    def test_deconv_relu_inference(self):
        for dim in [1, 2, 3]:
            input_shapes = {1: (4,), 2: (4, 4), 3: (4, 4, 4)}
            input_channel_per_group = 6
            output_channel_per_group = 3
            kernel_size = 3
            options = itertools.product(
                [torch.float32, torch.bfloat16],
                ["torchscript", "inductor"],
                [True, False],
                [True, False],
                [True, False],
                [True, False],
            )
            for (
                dtype,
                compiler_backend,
                dynamic,
                ipex_optimize,
                weight_prepack,
                feed_sample_input,
            ) in options:
                if compiler_backend == "torchscript" and dynamic is True:
                    continue
                if weight_prepack is True and ipex_optimize is False:
                    continue
                if feed_sample_input is True and weight_prepack is False:
                    continue
                ic = input_channel_per_group
                oc = output_channel_per_group
                x_shape = (2, ic) + input_shapes[dim]
                x = torch.randn(x_shape, dtype=torch.float32)
                model = DeconvNd_Relu(dim, ic, oc, kernel_size).eval()
                if ipex_optimize:
                    # TODO: support channels_last_1d.
                    if dim == 1:
                        ipex.disable_auto_channels_last()
                    else:
                        ipex.enable_auto_channels_last()
                    if feed_sample_input:
                        model = ipex.optimize(
                            model,
                            weights_prepack=weight_prepack,
                            dtype=dtype,
                            sample_input=x,
                        )
                    else:
                        model = ipex.optimize(
                            model, weights_prepack=weight_prepack, dtype=dtype
                        )
                torch._dynamo.reset()
                ipex._set_compiler_backend(compiler_backend)
                compile_model = torch.compile(model, dynamic=dynamic, backend="ipex")
                with torch.cpu.amp.autocast(
                    enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
                ), torch.no_grad():
                    ori_y = model(x)
                    for _ in range(3):
                        y = compile_model(x)
                self.assertEqual(y, ori_y)
                self.assertTrue(y.dtype == dtype)

    def test_deconv_relu_train(self):
        for dim in [1, 2, 3]:
            input_shapes = {1: (4,), 2: (4, 4), 3: (4, 4, 4)}
            input_channel_per_group = 6
            output_channel_per_group = 3
            kernel_size = 3
            options = itertools.product(
                [torch.float32, torch.bfloat16],
                ["inductor"],
                [True, False],
                [True, False],
                [True, False],
                [True, False],
            )
            for (
                dtype,
                compiler_backend,
                dynamic,
                ipex_optimize,
                weight_prepack,
                feed_sample_input,
            ) in options:
                if weight_prepack is True and ipex_optimize is False:
                    continue
                if feed_sample_input is True and weight_prepack is False:
                    continue
                ic = input_channel_per_group
                oc = output_channel_per_group
                x_shape = (2, ic) + input_shapes[dim]
                input = torch.randn(x_shape, dtype=torch.float32)
                ori_x = input.clone().requires_grad_()
                x = input.clone().requires_grad_()
                sample_x = input.clone().requires_grad_()
                deconv = DeconvNd_Relu(dim, ic, oc, kernel_size)
                ori_model = copy.deepcopy(deconv).train()
                model = copy.deepcopy(deconv).train()
                optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
                if ipex_optimize:
                    # TODO: support channels_last_1d.
                    if dim == 1:
                        ipex.disable_auto_channels_last()
                    else:
                        ipex.enable_auto_channels_last()
                    if feed_sample_input:
                        ori_model, _ = ipex.optimize(
                            ori_model,
                            weights_prepack=weight_prepack,
                            dtype=dtype,
                            optimizer=optimizer,
                            sample_input=sample_x,
                        )
                        model, _ = ipex.optimize(
                            model,
                            weights_prepack=weight_prepack,
                            dtype=dtype,
                            optimizer=optimizer,
                            sample_input=sample_x,
                        )
                    else:
                        ori_model, _ = ipex.optimize(
                            ori_model,
                            weights_prepack=weight_prepack,
                            dtype=dtype,
                            optimizer=optimizer,
                        )
                        model, _ = ipex.optimize(
                            model,
                            weights_prepack=weight_prepack,
                            dtype=dtype,
                            optimizer=optimizer,
                        )
                torch._dynamo.reset()
                ipex._set_compiler_backend(compiler_backend)
                compile_model = torch.compile(model, dynamic=dynamic, backend="ipex")
                with torch.cpu.amp.autocast(
                    enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
                ):
                    ori_y = ori_model(ori_x)
                    y = compile_model(x)
                    grad_x = torch.randn(ori_y.shape, dtype=torch.float32)
                    ori_y.backward(grad_x)
                    y.backward(grad_x)
                    self.assertEqual(y, ori_y)
                    self.assertTrue(y.dtype == dtype)
                    self.assertEqual(x.grad, ori_x.grad)

    def test_linear_relu_inference(self):
        out_features = 4
        in_features = 3
        input_shapes = [(2, in_features), (2, 2, in_features), (2, 2, 2, in_features)]
        options = itertools.product(
            input_shapes,
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
            [True, False],
            [True, False],
            [True, False],
        )
        for (
            x_shape,
            dtype,
            compiler_backend,
            dynamic,
            ipex_optimize,
            weight_prepack,
            feed_sample_input,
        ) in options:
            if compiler_backend == "torchscript" and dynamic is True:
                continue
            if weight_prepack is True and ipex_optimize is False:
                continue
            if feed_sample_input is True and weight_prepack is False:
                continue
            x = torch.randn(x_shape, dtype=torch.float32)
            model = Linear_Relu(in_features, out_features).eval()
            if ipex_optimize:
                if feed_sample_input:
                    model = ipex.optimize(
                        model,
                        weights_prepack=weight_prepack,
                        dtype=dtype,
                        sample_input=x,
                    )
                else:
                    model = ipex.optimize(
                        model, weights_prepack=weight_prepack, dtype=dtype
                    )
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            compile_model = torch.compile(model, dynamic=dynamic, backend="ipex")
            with torch.cpu.amp.autocast(
                enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
            ), torch.no_grad():
                ori_y = model(x)
                for _ in range(3):
                    y = compile_model(x)
            self.assertEqual(y, ori_y, prec=0.01)
            self.assertTrue(y.dtype == dtype)

    def test_linear_relu_train(self):
        out_features = 4
        in_features = 3

        input_shapes = [(2, in_features), (2, 2, in_features), (2, 2, 2, in_features)]
        options = itertools.product(
            input_shapes,
            [torch.float32, torch.bfloat16],
            ["inductor"],
            [True, False],
            [True, False],
            [True, False],
            [True, False],
        )
        for (
            x_shape,
            dtype,
            compiler_backend,
            dynamic,
            ipex_optimize,
            weight_prepack,
            feed_sample_input,
        ) in options:
            if weight_prepack is True and ipex_optimize is False:
                continue
            if feed_sample_input is True and weight_prepack is False:
                continue
            input = torch.randn(x_shape, dtype=torch.float32)
            ori_x = input.clone().requires_grad_()
            x = input.clone().requires_grad_()
            sample_x = input.clone().requires_grad_()
            linear = Linear_Relu(in_features, out_features)
            ori_model = copy.deepcopy(linear).train()
            model = copy.deepcopy(linear).train()
            optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
            if ipex_optimize:
                if feed_sample_input:
                    ori_model, _ = ipex.optimize(
                        ori_model,
                        weights_prepack=weight_prepack,
                        dtype=dtype,
                        optimizer=optimizer,
                        sample_input=sample_x,
                    )
                    model, _ = ipex.optimize(
                        model,
                        weights_prepack=weight_prepack,
                        dtype=dtype,
                        optimizer=optimizer,
                        sample_input=sample_x,
                    )
                else:
                    ori_model, _ = ipex.optimize(
                        ori_model,
                        weights_prepack=weight_prepack,
                        dtype=dtype,
                        optimizer=optimizer,
                    )
                    model, _ = ipex.optimize(
                        model,
                        weights_prepack=weight_prepack,
                        dtype=dtype,
                        optimizer=optimizer,
                    )
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            compile_model = torch.compile(model, dynamic=dynamic, backend="ipex")
            with torch.cpu.amp.autocast(
                enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
            ):
                ori_y = ori_model(ori_x)
                y = compile_model(x)
                grad_x = torch.randn(ori_y.shape, dtype=torch.float32)
                ori_y.backward(grad_x)
                y.backward(grad_x)
                self.assertEqual(y, ori_y, prec=0.01)
                self.assertTrue(y.dtype == dtype)
                self.assertEqual(x.grad, ori_x.grad, prec=0.01)

    def test_lstm_inference(self):
        options = itertools.product(
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
            [True, False],
        )
        for dtype, compiler_backend, dynamic, ipex_optimize in options:
            if compiler_backend == "torchscript" and dynamic is True:
                continue
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
            ipex._set_compiler_backend(compiler_backend)
            compile_model = torch.compile(model, dynamic=dynamic, backend="ipex")
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

    def test_bmm_add_inference(self):
        options = itertools.product(
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
        )
        for dtype, compiler_backend, dynamic in options:
            if compiler_backend == "torchscript" and dynamic is True:
                continue
            x = torch.randn(6, 3, 5).to(dtype=dtype)
            b1 = torch.randn(6, 3, 4).to(dtype=dtype)
            b2 = torch.randn(6, 4, 5).to(dtype=dtype)
            model = BmmAdd().eval()
            with torch.cpu.amp.autocast(
                enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
            ), torch.no_grad():
                ori_y = model(x, b1, b2)
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            compile_model = torch.compile(model, dynamic=dynamic, backend="ipex")
            with torch.cpu.amp.autocast(
                enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
            ), torch.no_grad():
                y = compile_model(x, b1, b2)
            self.assertEqual(ori_y, y, prec=0.1)
            self.assertTrue(y.dtype == dtype)

    def test_add_softmax_inference(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        model = AddSoftmax()
        ori_y = model(a, b)
        # TODO: support custom inplace operators in inductor path.
        for compiler_backend in [
            "torchscript",
        ]:
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            compile_model = torch.compile(model, backend="ipex")

            y = compile_model(a, b)
            self.assertEqual(ori_y, y)

    def test_frozen_batch_norm_inference(self):
        torch._dynamo.allow_in_graph(FrozenBatchNorm2d)
        options = itertools.product(
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
        )
        for dtype, compiler_backend, dynamic in options:
            if compiler_backend == "torchscript" and dynamic is True:
                continue
            x = (
                torch.randn(20, 100, 35, 45)
                .to(dtype=dtype)
                .to(memory_format=torch.channels_last)
            )
            model = FrozenBatchNorm2d(100).eval()
            with torch.cpu.amp.autocast(
                enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
            ), torch.no_grad():
                ori_y = model(x)
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            compile_model = torch.compile(model, dynamic=dynamic, backend="ipex")
            with torch.cpu.amp.autocast(
                enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
            ), torch.no_grad():
                y = compile_model(x)
            self.assertEqual(ori_y, y)
            self.assertTrue(y.dtype == dtype)

    def test_frozen_batch_norm_train(self):
        torch._dynamo.allow_in_graph(FrozenBatchNorm2d)
        options = itertools.product(
            [torch.float32, torch.bfloat16],
            [
                "inductor",
            ],
            [True, False],
        )
        for dtype, compiler_backend, dynamic in options:
            input = (
                torch.randn(20, 100, 35, 45)
                .to(dtype=dtype)
                .to(memory_format=torch.channels_last)
            )
            ori_x = input.clone().requires_grad_()
            x = input.clone().requires_grad_()
            FrozenBatchNorm = FrozenBatchNorm2d(100)
            ori_model = copy.deepcopy(FrozenBatchNorm).train()
            model = copy.deepcopy(FrozenBatchNorm).train()
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            compile_model = torch.compile(model, dynamic=dynamic, backend="ipex")
            with torch.cpu.amp.autocast(
                enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
            ):
                ori_y = ori_model(ori_x)
                y = compile_model(x)
                ori_y.mean().backward()
                y.mean().backward()
            self.assertEqual(ori_y, y)
            self.assertEqual(ori_x.grad, x.grad)
            self.assertTrue(y.dtype == dtype)
            self.assertTrue(x.grad.dtype == dtype)

    def test_cumsum(self):
        def func(x):
            return torch.ops.torch_ipex.cumsum(x, 1)

        options = itertools.product(
            ["torchscript", "inductor"],
            [True, False],
        )
        x = torch.randn(17, 47)
        for compiler_backend, dynamic in options:
            if compiler_backend == "torchscript" and dynamic is True:
                continue
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            compile_fn = torch.compile(func, dynamic=dynamic, backend="ipex")
            with torch.no_grad():
                ori_y = func(x)
                y = compile_fn(x)
                self.assertEqual(ori_y, y)

    def test_linear_eltwise(self):
        torch._dynamo.allow_in_graph(ipex.nn.modules.IPEXLinearEltwise)
        input = torch.rand(5, 10).requires_grad_()

        options = itertools.product(
            [torch.float32, torch.bfloat16],
            [
                "inductor",
            ],
            [True, False],
        )
        for dtype, compiler_backend, dynamic in options:
            fused_model = MLP()
            opt = torch.optim.SGD(fused_model.parameters(), lr=0.01)
            fused_model, opt = ipex.optimize(
                fused_model, dtype=dtype, optimizer=opt, auto_kernel_selection=True
            )
            fused_model.mlp[0] = ipex.nn.modules.IPEXLinearEltwise(
                fused_model.mlp[0], "relu"
            )
            fused_model.mlp[1] = torch.nn.Identity()

            ori_x = input.to(dtype=dtype).clone().detach().requires_grad_()
            x = input.to(dtype=dtype).clone().detach().requires_grad_()

            ori_model = copy.deepcopy(fused_model).train()
            model = copy.deepcopy(fused_model).train()

            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            compile_model = torch.compile(model, dynamic=dynamic, backend="ipex")

            with torch.cpu.amp.autocast(enabled=(dtype == torch.bfloat16)):
                ori_y = ori_model(ori_x)
                ori_y.sum().backward()
                y = compile_model(x)
                y.sum().backward()

            self.assertEqual(ori_y, y)
            self.assertTrue(y.dtype == dtype)
            self.assertEqual(ori_x.grad, x.grad)
            self.assertTrue(x.grad.dtype == dtype)

    def test_emb_torch_compile(self):
        emb = Embeddingbag().eval()
        input = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
        offsets = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7])
        for dtype, compiler_backend, dynamic in itertools.product(
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
        ):
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            emb_torchcompile = torch.compile(emb, dynamic=dynamic, backend="ipex")
            with torch.cpu.amp.autocast(
                enabled=(dtype == torch.bfloat16)
            ), torch.no_grad():
                y0 = emb(input, offsets)
                y1 = emb_torchcompile(input, offsets)
            self.assertEqual(y0, y1)
            self.assertEqual(y1.dtype, dtype)

    def test_RMSNorm_torchcompile(self):
        for dim in [2, 3, 4, 5]:
            with torch.cpu.amp.autocast(), torch.no_grad():
                input_size = [
                    3,
                ]
                for _ in range(dim - 1):
                    input_size.append(10)
                x = torch.randn(input_size)
                # RMSNorm input is fp32
                model = RMSNorm(input_size).eval()
                torch._dynamo.reset()
                compiled_model = torch.compile(model, backend="ipex")
                y1_fp32 = model(x, fused_rmsnorm=True)
                y2_fp32 = compiled_model(x, fused_rmsnorm=True)
                self.assertEqual(y1_fp32, y2_fp32)
                x_bf16 = x.to(torch.bfloat16)
                y1_bf16 = model(x_bf16, fused_rmsnorm=True)
                y2_bf16 = compiled_model(x_bf16, fused_rmsnorm=True)
                self.assertEqual(y1_bf16, y2_bf16)

    def test_mha_torchcompile(self):
        MaskedMHATest._test_mha(self, torchcompile=True)
        MaskedMHATest._test_mha_fp16(self, torchcompile=True)

    @skipIfNoTorchVision
    def test_torchvision_roialign_inference_torchcompile(self):
        pool_size = 5
        n_channels = 2 * (pool_size**2)
        x = torch.rand(2, n_channels, 10, 10).to(memory_format=torch.channels_last)
        rois = torch.tensor(
            [
                [0, 0, 0, 9, 9],  # format is (xyxy)
                [0, 0, 5, 4, 9],
                [0, 5, 5, 9, 9],
                [1, 0, 0, 9, 9],
            ]
        )
        pool_h, pool_w = pool_size, pool_size

        for dtype, compiler_backend, dynamic in itertools.product(
            [torch.float32, torch.bfloat16], ["torchscript", "inductor"], [True, False]
        ):
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            torchcompile_torchvision_fn = torch.compile(
                torchvision_fn, dynamic=dynamic, backend="ipex"
            )
            x = x.to(dtype=dtype)
            rois = rois.to(dtype=dtype)
            # forward
            with torch.cpu.amp.autocast(
                enabled=(dtype == torch.bfloat16)
            ), torch.no_grad():
                y0 = torchvision_fn(
                    x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1
                )
                y1 = torchcompile_torchvision_fn(
                    x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1
                )
                self.assertEqual(y0, y1)
                self.assertTrue(y1.dtype == dtype)

    @skipIfNoTorchVision
    def test_torchvision_roialign_train_torchcompile(self):
        pool_size = 5
        n_channels = 2 * (pool_size**2)
        input = torch.rand(2, n_channels, 10, 10).to(memory_format=torch.channels_last)
        rois = torch.tensor(
            [
                [0, 0, 0, 9, 9],  # format is (xyxy)
                [0, 0, 5, 4, 9],
                [0, 5, 5, 9, 9],
                [1, 0, 0, 9, 9],
            ]
        )
        pool_h, pool_w = pool_size, pool_size

        for dtype, compiler_backend, dynamic in itertools.product(
            [torch.float32, torch.bfloat16], ["inductor"], [True, False]
        ):
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            torchcompile_torchvision_fn = torch.compile(
                copy.deepcopy(torchvision_fn), dynamic=dynamic, backend="ipex"
            )
            input = input.to(dtype=dtype)
            rois = rois.to(dtype=dtype)
            ori_x = input.clone().requires_grad_()
            x = input.clone().requires_grad_()

            # forward
            with torch.cpu.amp.autocast(enabled=(dtype == torch.bfloat16)):
                ori_y = torchvision_fn(
                    ori_x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1
                )
                y = torchcompile_torchvision_fn(
                    x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1
                )
                grad_y = torch.randn(ori_y.shape, dtype=torch.float32)
                ori_y.backward(grad_y)
                y.backward(grad_y)
                self.assertEqual(y, ori_y)
                self.assertTrue(y.dtype == dtype)
                self.assertEqual(x.grad, ori_x.grad)

    def test_tpp_linear_torchcompile(self):
        x = torch.rand(2, 2, 4096)

        options = itertools.product(
            [Linear_with_bias, Linear_without_bias],
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
            [True, False],
        )
        for (
            Model,
            dtype,
            compiler_backend,
            dynamic,
            cpp_wrapper,
        ) in options:
            if compiler_backend == "torchscript" and cpp_wrapper:
                continue
            model = Model().to(dtype=dtype).eval()
            x = x.to(dtype=dtype)

            _enable_tpp()
            model = ipex.optimize(model, dtype=dtype)

            with torch.no_grad():
                ref_out = model(x)
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            torch._inductor.config.cpp_wrapper = cpp_wrapper
            compile_model = torch.compile(model, dynamic=dynamic, backend="ipex")
            with torch.no_grad():
                out = compile_model(x)
            self.assertEqual(out, ref_out)
            self.assertTrue(out.dtype == dtype)
            _disable_tpp()

    def test_tpp_linear_gelu_torchcompile(self):
        x = torch.rand(2, 2, 4096)

        options = itertools.product(
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
            [True, False],
        )
        for (
            dtype,
            compiler_backend,
            dynamic,
            cpp_wrapper,
        ) in options:
            if compiler_backend == "torchscript" and cpp_wrapper:
                continue
            model = Linear_gelu().to(dtype=dtype).eval()
            x = x.to(dtype=dtype)

            _enable_tpp()
            model = ipex.optimize(model, dtype=dtype)

            def fn(x):
                return torch.ops.torch_ipex.tpp_linear_gelu(
                    x, model.mlp.weight, model.mlp.bias, model.mlp.out_features
                )

            with torch.no_grad():
                ref_out = fn(x)
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            torch._inductor.config.cpp_wrapper = cpp_wrapper
            compile_fn = torch.compile(fn, dynamic=dynamic, backend="ipex")
            with torch.no_grad():
                out = compile_fn(x)
            self.assertEqual(out, ref_out)
            self.assertTrue(out.dtype == dtype)
            _disable_tpp()

    def test_tpp_linear_silu_torchcompile(self):
        x = torch.rand(2, 2, 4096)

        options = itertools.product(
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
            [True, False],
        )
        for (
            dtype,
            compiler_backend,
            dynamic,
            cpp_wrapper,
        ) in options:
            if compiler_backend == "torchscript" and cpp_wrapper:
                continue
            model = Linear_silu().to(dtype=dtype).eval()
            x = x.to(dtype=dtype)

            _enable_tpp()
            model = ipex.optimize(model, dtype=dtype)

            def fn(x):
                return torch.ops.torch_ipex.tpp_linear_silu(
                    x, model.mlp.weight, x.new_empty(0), model.mlp.out_features
                )

            with torch.no_grad():
                ref_out = fn(x)
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            torch._inductor.config.cpp_wrapper = cpp_wrapper
            compile_fn = torch.compile(fn, dynamic=dynamic, backend="ipex")
            with torch.no_grad():
                out = compile_fn(x)
            self.assertEqual(out, ref_out)
            self.assertTrue(out.dtype == dtype)
            _disable_tpp()

    def test_tpp_linear_relu_torchcompile(self):
        x = torch.rand(2, 2, 4096)

        options = itertools.product(
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
            [True, False],
        )
        for (
            dtype,
            compiler_backend,
            dynamic,
            cpp_wrapper,
        ) in options:
            if compiler_backend == "torchscript" and cpp_wrapper:
                continue
            model = Linear_relu().to(dtype=dtype).eval()
            x = x.to(dtype=dtype)

            _enable_tpp()
            model = ipex.optimize(model, dtype=dtype)

            def fn(x):
                return torch.ops.torch_ipex.tpp_linear_relu(
                    x, model.mlp.weight, x.new_empty(0), model.mlp.out_features
                )

            with torch.no_grad():
                ref_out = fn(x)
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            torch._inductor.config.cpp_wrapper = cpp_wrapper
            compile_fn = torch.compile(fn, dynamic=dynamic, backend="ipex")
            with torch.no_grad():
                out = compile_fn(x)
            self.assertEqual(out, ref_out)
            self.assertTrue(out.dtype == dtype)
            _disable_tpp()

    def test_tpp_linear_mul_torchcompile(self):
        x = torch.rand(2, 2, 4096)

        options = itertools.product(
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
            [True, False],
        )
        for (
            dtype,
            compiler_backend,
            dynamic,
            cpp_wrapper,
        ) in options:
            if compiler_backend == "torchscript" and cpp_wrapper:
                continue
            model = Linear_mul().to(dtype=dtype).eval()
            x = x.to(dtype=dtype)

            _enable_tpp()
            model = ipex.optimize(model, dtype=dtype)

            def fn(x):
                return torch.ops.torch_ipex.tpp_linear_mul(
                    x, x, model.mlp.weight, x.new_empty(0), model.mlp.out_features
                )

            with torch.no_grad():
                ref_out = fn(x)
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            torch._inductor.config.cpp_wrapper = cpp_wrapper
            compile_fn = torch.compile(fn, dynamic=dynamic, backend="ipex")
            with torch.no_grad():
                out = compile_fn(x)
            self.assertEqual(out, ref_out)
            self.assertTrue(out.dtype == dtype)
            _disable_tpp()

    def test_tpp_linear_add_torchcompile(self):
        x = torch.rand(2, 2, 4096)

        options = itertools.product(
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
            [True, False],
        )
        for (
            dtype,
            compiler_backend,
            dynamic,
            cpp_wrapper,
        ) in options:
            if compiler_backend == "torchscript" and cpp_wrapper:
                continue
            model = Linear_add().to(dtype=dtype).eval()
            x = x.to(dtype=dtype)

            _enable_tpp()
            model = ipex.optimize(model, dtype=dtype)

            def fn(x):
                return torch.ops.torch_ipex.tpp_linear_add(
                    x, x, model.mlp.weight, x.new_empty(0), 1.0, model.mlp.out_features
                )

            with torch.no_grad():
                ref_out = fn(x)
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            torch._inductor.config.cpp_wrapper = cpp_wrapper
            compile_fn = torch.compile(fn, dynamic=dynamic, backend="ipex")
            with torch.no_grad():
                out = compile_fn(x)
            self.assertEqual(out, ref_out)
            self.assertTrue(out.dtype == dtype)
            _disable_tpp()

    def test_tpp_linear_add2_torchcompile(self):
        x = torch.rand(2, 2, 4096)

        options = itertools.product(
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
            [True, False],
        )
        for (
            dtype,
            compiler_backend,
            dynamic,
            cpp_wrapper,
        ) in options:
            if compiler_backend == "torchscript" and cpp_wrapper:
                continue
            model = Linear_add_add().to(dtype=dtype).eval()
            x = x.to(dtype=dtype)

            _enable_tpp()
            model = ipex.optimize(model, dtype=dtype)

            def fn(x):
                return torch.ops.torch_ipex.tpp_linear_add_add(
                    x,
                    x,
                    x,
                    model.mlp.weight,
                    model.mlp.bias,
                    1.0,
                    model.mlp.out_features,
                )

            with torch.no_grad():
                ref_out = fn(x)
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            torch._inductor.config.cpp_wrapper = cpp_wrapper
            compile_fn = torch.compile(fn, dynamic=dynamic, backend="ipex")
            with torch.no_grad():
                out = compile_fn(x)
            self.assertEqual(out, ref_out)
            self.assertTrue(out.dtype == dtype)
            _disable_tpp()


if __name__ == "__main__":
    torch.manual_seed(2020)
    test = unittest.main()
