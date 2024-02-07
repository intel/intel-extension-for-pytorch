import unittest
import itertools
import copy
import torch
import torch.nn.functional as F
from torch.optim import SGD
import intel_extension_for_pytorch as ipex

from common_utils import TestCase

conv_module = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}


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


class TestCompile(TestCase):
    def test_conv_relu_inference(self):
        for dim in [
            2,
        ]:
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
        for dim in [
            2,
        ]:
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


if __name__ == "__main__":
    test = unittest.main()
