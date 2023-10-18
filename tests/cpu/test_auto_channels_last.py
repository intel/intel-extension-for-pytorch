import unittest
from common_utils import TestCase
import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.utils.channels_last_1d import (
    is_contiguous_channels_last_1d,
)

try:
    import torchvision

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")


class TestAutoChannelsLast(TestCase):
    def _get_covnNd(self, dim):
        class ConvNd(nn.Module):
            def __init__(self, dim):
                super(ConvNd, self).__init__()
                if dim == 1:
                    self.conv = nn.Conv1d(16, 33, 3)
                elif dim == 2:
                    self.conv = nn.Conv2d(16, 33, 3)
                elif dim == 3:
                    self.conv = nn.Conv3d(16, 33, 3)

            def forward(self, x):
                x = self.conv(x)
                return x

        model = ConvNd(dim=dim)
        return model

    def _get_sequential_conv2d(self):
        class Conv2d(nn.Module):
            def __init__(self):
                super(Conv2d, self).__init__()
                self.conv1 = nn.Conv2d(16, 33, 3)
                self.conv2 = nn.Conv2d(33, 33, 3)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        model = Conv2d()
        return model

    def _get_covnNd_relu(self, dim):
        class ConvNdReLU(nn.Module):
            def __init__(self, dim):
                super(ConvNdReLU, self).__init__()
                if dim == 1:
                    self.conv = nn.Conv1d(16, 33, 3)
                elif dim == 2:
                    self.conv = nn.Conv2d(16, 33, 3)
                elif dim == 3:
                    self.conv = nn.Conv3d(16, 33, 3)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.relu(x)
                return x

        model = ConvNdReLU(dim=dim)
        return model

    def _get_covnNd_linear(self, dim):
        class ConvNdLinear(nn.Module):
            def __init__(self, dim):
                super(ConvNdLinear, self).__init__()
                if dim == 1:
                    self.conv = nn.Conv1d(16, 33, 3)
                elif dim == 2:
                    self.conv = nn.Conv2d(16, 33, 3)
                elif dim == 3:
                    self.conv = nn.Conv3d(16, 33, 3)
                self.linear = nn.Linear(48, 48)

            def forward(self, x):
                x = self.conv(x)
                x = self.linear(x)
                return x

        model = ConvNdLinear(dim=dim)
        return model

    def _get_ipex_optimized_model_and_output_tensor(
        self, model, dim, disable_auto_channels_last=False
    ):
        model.eval()

        if dim == 1:
            x = torch.randn(20, 16, 50)
        elif dim == 2:
            x = torch.randn(20, 16, 50, 50)
        elif dim == 3:
            x = torch.randn(20, 16, 50, 50, 50)

        if disable_auto_channels_last:
            ipex.disable_auto_channels_last()

        model = ipex.optimize(model, weights_prepack=False)
        output = model(x)
        return model, output

    def get_channels_last_modules(self, module):
        channels_last_modules = []
        for name, param in module.named_parameters():
            if param.is_contiguous(memory_format=torch.channels_last):
                channels_last_modules.append(name)
        return channels_last_modules

    def test_auto_channels_last(self):
        model = self._get_covnNd(dim=1)
        model, output = self._get_ipex_optimized_model_and_output_tensor(model, dim=1)
        self.assertTrue(is_contiguous_channels_last_1d(model.conv.weight))
        self.assertTrue(is_contiguous_channels_last_1d(output))

        model = self._get_covnNd(dim=2)
        model, output = self._get_ipex_optimized_model_and_output_tensor(model, dim=2)
        self.assertTrue(
            model.conv.weight.is_contiguous(memory_format=torch.channels_last)
        )
        self.assertTrue(output.is_contiguous(memory_format=torch.channels_last))

        model = self._get_covnNd(dim=3)
        model, output = self._get_ipex_optimized_model_and_output_tensor(model, dim=3)
        self.assertTrue(
            model.conv.weight.is_contiguous(memory_format=torch.channels_last_3d)
        )

    def test_disable_auto_channels_last(self):
        model = self._get_covnNd(dim=1)
        model, output = self._get_ipex_optimized_model_and_output_tensor(
            model, dim=1, disable_auto_channels_last=True
        )
        self.assertTrue(
            model.conv.weight.is_contiguous(memory_format=torch.contiguous_format)
        )
        self.assertTrue(output.is_contiguous(memory_format=torch.contiguous_format))

        model = self._get_covnNd(dim=2)
        model, output = self._get_ipex_optimized_model_and_output_tensor(
            model, dim=2, disable_auto_channels_last=True
        )
        self.assertTrue(
            model.conv.weight.is_contiguous(memory_format=torch.contiguous_format)
        )
        self.assertTrue(output.is_contiguous(memory_format=torch.contiguous_format))

        model = self._get_covnNd(dim=3)
        model, output = self._get_ipex_optimized_model_and_output_tensor(
            model, dim=3, disable_auto_channels_last=True
        )
        self.assertTrue(
            model.conv.weight.is_contiguous(memory_format=torch.contiguous_format)
        )
        self.assertTrue(output.is_contiguous(memory_format=torch.contiguous_format))

    def test_auto_channels_last_recursion(self):
        model = self._get_sequential_conv2d()
        model, output = self._get_ipex_optimized_model_and_output_tensor(model, dim=2)

        self.assertTrue(
            model.conv1.weight.is_contiguous(memory_format=torch.channels_last)
        )
        self.assertTrue(
            model.conv2.weight.is_contiguous(memory_format=torch.channels_last)
        )
        self.assertTrue(output.is_contiguous(memory_format=torch.channels_last))

    def test_auto_channels_last_memory_format_propagation(self):
        # memory format propagates through channels_last compatible layers
        model = self._get_covnNd_relu(dim=1)
        model, output = self._get_ipex_optimized_model_and_output_tensor(model, dim=1)
        self.assertTrue(is_contiguous_channels_last_1d(model.conv.weight))
        self.assertTrue(is_contiguous_channels_last_1d(output))

        model = self._get_covnNd_relu(dim=2)
        model, output = self._get_ipex_optimized_model_and_output_tensor(model, dim=2)
        self.assertTrue(
            model.conv.weight.is_contiguous(memory_format=torch.channels_last)
        )
        self.assertTrue(output.is_contiguous(memory_format=torch.channels_last))

        model = self._get_covnNd_relu(dim=3)
        model, output = self._get_ipex_optimized_model_and_output_tensor(model, dim=3)
        self.assertTrue(
            model.conv.weight.is_contiguous(memory_format=torch.channels_last_3d)
        )

        # memory format reverts back to contiguous_format as linear is channels_last incompatible
        model = self._get_covnNd_linear(dim=1)
        model, output = self._get_ipex_optimized_model_and_output_tensor(model, dim=1)
        self.assertTrue(is_contiguous_channels_last_1d(model.conv.weight))
        self.assertTrue(output.is_contiguous(memory_format=torch.contiguous_format))

        model = self._get_covnNd_linear(dim=2)
        model, output = self._get_ipex_optimized_model_and_output_tensor(model, dim=2)
        self.assertTrue(
            model.conv.weight.is_contiguous(memory_format=torch.channels_last)
        )
        self.assertTrue(output.is_contiguous(memory_format=torch.contiguous_format))

        model = self._get_covnNd_linear(dim=3)
        model, output = self._get_ipex_optimized_model_and_output_tensor(model, dim=3)
        self.assertTrue(
            model.conv.weight.is_contiguous(memory_format=torch.channels_last_3d)
        )
        self.assertTrue(output.is_contiguous(memory_format=torch.contiguous_format))

    @skipIfNoTorchVision
    def test_auto_channels_last_resnet50(self):
        model = torchvision.models.resnet.resnet50(pretrained=False)
        model.eval()

        # manual
        model_channels_last = model.to(memory_format=torch.channels_last)
        model_channels_last = self.get_channels_last_modules(model_channels_last)

        # auto
        model_ipex = ipex.optimize(model, weights_prepack=False)
        model_ipex_channels_last_modules = self.get_channels_last_modules(model_ipex)

        self.assertEqual(model_channels_last, model_ipex_channels_last_modules)

    def test_auto_channels_last_for_int8(self):
        conv_module = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        class ConvNd(torch.nn.Module):
            def __init__(self, dim, in_channels, out_channels, kernel_size, stride):
                super(ConvNd, self).__init__()
                self.conv = conv_module[dim](
                    in_channels, out_channels, kernel_size=kernel_size, stride=stride
                )

            def forward(self, x):
                return self.conv(x)

        def _test_conv(dim):
            input_shapes = {1: (224,), 2: (224, 224), 3: (55, 55, 55)}
            x_shape = (2, 3) + input_shapes[dim]
            x = torch.randn(x_shape, dtype=torch.float32)
            model = ConvNd(dim, 3, 4, 3, 2).eval()
            qconfig = ipex.quantization.default_static_qconfig
            prepared_model = ipex.quantization.prepare(model, qconfig, x)
            # do calibration
            y = prepared_model(x)
            convert_model = ipex.quantization.convert(prepared_model)
            with torch.no_grad():
                traced_model = torch.jit.trace(convert_model, x)
                traced_model = torch.jit.freeze(traced_model)
                for _ in range(3):
                    y = traced_model(x)
            return y

        # disable auto channels_last
        ipex.disable_auto_channels_last()
        self.assertTrue(
            _test_conv(2).is_contiguous(memory_format=torch.contiguous_format)
        )
        self.assertTrue(
            _test_conv(3).is_contiguous(memory_format=torch.contiguous_format)
        )

        # enable auto channels_last
        ipex.enable_auto_channels_last()

        self.assertTrue(_test_conv(2).is_contiguous(memory_format=torch.channels_last))
        # temporary disable before https://github.com/pytorch/pytorch/pull/74023 merged
        # self.assertTrue(_test_conv(3).is_contiguous(memory_format = torch.channels_last_3d))


if __name__ == "__main__":
    test = unittest.main()
