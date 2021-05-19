import torch
import unittest
import itertools
import torch.nn as nn
import torch.nn.functional as F
from test_jit_llga_utils import JitLlgaTestCase, run_tests, LLGA_FUSION_GROUP
from torch.testing._internal.common_utils import TEST_SCIPY

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
except RuntimeError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, 'no torchvision')

def get_eltwise_fn(name):
    if hasattr(torch, name):
        return getattr(torch, name)
    elif hasattr(F, name):
        return getattr(F, name)
    else:
        raise NameError('Eltwise function %s not found' % name)

class TestOp(JitLlgaTestCase):
    def test_conv2d(self):
        for [
                spatial,
                in_channels,
                out_channels,
                kernel,
                padding,
                stride,
                dilation,
                g,
                bias
            ] in itertools.product(
                [7, 8],
                [8, 15],
                [7, 16],
                [3, 4],
                [0, 2],
                [1, 2],
                [1, 2],
                [1, 2],
                [True, False]):

            m = nn.Conv2d(in_channels=in_channels * g,
                          out_channels=out_channels * g,
                          kernel_size=kernel,
                          padding=padding,
                          stride=stride,
                          dilation=dilation,
                          groups=g,
                          bias=bias)
            x = torch.rand(1, in_channels * g, spatial, spatial)
            
            graph = self.checkQuantizeTrace(m, [x], atol=2e-1, config_name="conv2d")
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 4)
            self.assertFused(graph, ['aten::_convolution', 'aten::quantize_per_tensor', 'aten::quantize_per_channel', 'aten::dequantize'])
            
            patterns = [
                ["aten::quantize_per_tensor"],
                ["aten::quantize_per_channel"],
                ["aten::dequantize", "aten::_convolution", "aten::quantize_per_tensor"],
                ["aten::dequantize"]
            ]
            self.checkPatterns(graph, patterns)

    def test_linear(self):
        for bias in [True, False]:
            x = torch.rand(32, 28)
            m = torch.nn.Linear(in_features=28, out_features=64, bias=bias)
            
            graph = self.checkQuantizeTrace(m, [x], atol=1e-1, config_name="linear")
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 4)
            self.assertFused(graph, ['aten::linear', 'aten::quantize_per_tensor', 'aten::quantize_per_channel', 'aten::dequantize'])

            patterns = [
                ["aten::quantize_per_tensor"],
                ["aten::quantize_per_channel"],
                ["aten::dequantize", "aten::linear", "aten::quantize_per_tensor"],
                ["aten::dequantize"]
            ]
            self.checkPatterns(graph, patterns)

class TestFusionPattern(JitLlgaTestCase):
    def test_conv2d_eltwise(self):
        class M(nn.Module):
            def __init__(self, eltwise_fn):
                super(M, self).__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.eltwise = eltwise_fn

            def forward(self, x):
                x = self.conv1(x)
                x = self.eltwise(x)
                x = self.conv2(x)
                return x

        for eltwise in ['relu']: # TODO: ['sigmoid', 'sqrt', 'abs', 'square', 'hardtanh']
            for inplace in [False, True]:
                eltwise_fn_name = eltwise + '_' if inplace else eltwise
                eltwise_fn = get_eltwise_fn(eltwise_fn_name)

                m = M(eltwise_fn)
                x = torch.rand(1, 32, 28, 28)
                
                graph = self.checkQuantizeTrace(m, [x], atol=2e-1, config_name="conv2d_eltwise")
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 6)
                self.assertFused(graph, ['aten::_convolution', 'aten::' + eltwise, 'aten::quantize_per_tensor', 'aten::quantize_per_channel', 'aten::dequantize'])

                patterns = [
                    ["aten::quantize_per_tensor"],
                    ["aten::quantize_per_channel"],
                    ["aten::dequantize", "aten::_convolution", 'aten::' + eltwise, "aten::quantize_per_tensor"], # inplace op will become outplace op on the JIT graph
                    ["aten::quantize_per_channel"],
                    ["aten::dequantize", "aten::_convolution", "aten::quantize_per_tensor"],
                    ["aten::dequantize"]
                ]
                self.checkPatterns(graph, patterns)

    def test_conv2d_bn(self):
        class M(nn.Module):
            def __init__(self, bias):
                super(M, self).__init__()
                self.conv1 = nn.Conv2d(32, 5, 3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(5)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                return x
        for bias in [False, True]:
            m = M(bias).eval()
            x = torch.rand(1, 32, 16, 16)
            # TODO: This shape will fail
            # x = torch.rand(1, 32, 28, 28)
            
            graph = self.checkQuantizeTrace(m, [x], atol=1e-1, folding=True, config_name="conv2d_bn")
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 4)
            self.assertFused(graph, ['aten::_convolution', 'aten::quantize_per_tensor', 'aten::quantize_per_channel', 'aten::dequantize'])

            patterns = [
                ["aten::quantize_per_tensor"],
                ["aten::quantize_per_channel"],
                ["aten::dequantize", "aten::_convolution", "aten::quantize_per_tensor"],
                ["aten::dequantize"]
            ]
            self.checkPatterns(graph, patterns)

    def test_conv2d_bn_relu(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.bn1 = nn.BatchNorm2d(32)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = F.relu(x)
                return x

        m = M().eval()
        x = torch.rand(1, 32, 28, 28)
        graph = self.checkQuantizeTrace(m, [x], atol=1e-1, folding=True, config_name="conv2d_bn_relu")
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 4)
        self.assertFused(graph, ['aten::_convolution', 'aten::relu',
                                 'aten::quantize_per_tensor', 'aten::quantize_per_channel', 'aten::dequantize'])

        patterns = [
            ["aten::quantize_per_tensor"],
            ["aten::quantize_per_channel"],
            ["aten::dequantize", "aten::_convolution", "aten::relu", "aten::quantize_per_tensor"],
            ["aten::dequantize"]
        ]
        self.checkPatterns(graph, patterns)

    def test_linear_eltwise(self):
        class M(nn.Module):
            def __init__(self, eltwise_fn, bias):
                super(M, self).__init__()
                self.linear = nn.Linear(28, 64, bias)
                self.eltwise = eltwise_fn

            def forward(self, x):
                x = self.linear(x)
                x = self.eltwise(x)
                return x

        # TODO: use itertools.product once all combinations is supported
        for [has_bias, eltwise] in [
            [True, 'relu'],
            [False, 'relu'],
            # [True, 'gelu'], # TODO: enable it once linear_gelu default recipe is fixed
            # [False, 'gelu'], # TODO: enable it once linear_gelu default recipe is fixed
            [True, 'sigmoid'],
            [False, 'sigmoid'],
        ]:
            eltwise_fn = get_eltwise_fn(eltwise)
            m = M(eltwise_fn, has_bias)
            x = torch.rand(32, 28, requires_grad=False)
            
            graph = self.checkQuantizeTrace(m, [x], atol=1e-1, config_name="linear_eltwise")
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 4)
            self.assertFused(graph, ['aten::' + eltwise])

            patterns = [
                ["aten::quantize_per_tensor"],
                ["aten::quantize_per_channel"],
                ["aten::dequantize", "aten::linear", "aten::" + eltwise, "aten::quantize_per_tensor"],
                ["aten::dequantize"]
            ]
            self.checkPatterns(graph, patterns)

    def test_conv2d_sum(self):
        class M(nn.Module):
            def __init__(self, bias=False):
                super(M, self).__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=bias)
                self.bn1 = nn.BatchNorm2d(32)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=bias)
                self.bn2 = nn.BatchNorm2d(32)
                self.relu = nn.ReLU()
                self.conv3 = nn.Conv2d(32, 32, 3, padding=1, bias=bias)
                self.bn3 = nn.BatchNorm2d(32)

            def forward(self, x, y):
                x = self.conv1(x)
                x = self.bn1(x)
                y = self.conv2(y)
                y = self.bn2(y)
                z = self.relu(x + y)
                z = self.conv3(z)
                z = self.bn3(z)
                return z

        for bias in [True, False]:
            m = M(bias).eval()
            x = torch.rand(1, 32, 16, 16, requires_grad=False)
            y = torch.rand(1, 32, 16, 16, requires_grad=False)
            graph = self.checkQuantizeTrace(m, [x, y], folding=True, atol=1e-1, config_name="conv2d_sum")
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 13) # TODO: nb FUSION_GROUP=10 when oneDNN support sum post_ops with zps

            # TODO: check patterns when oneDNN support sum post_ops with zps
            # patterns = [
            #     ["aten::quantize_per_tensor"],
            #     ["aten::quantize_per_channel"],
            #     ["aten::dequantize", "aten::_convolution", "aten::quantize_per_tensor"],
            #     ["aten::quantize_per_channel"],
            #     ["aten::dequantize", "aten::_convolution", "aten::quantize_per_tensor"],
            #     ["aten::quantize_per_channel"],
            #     ["aten::dequantize", "aten::_convolution", "aten::relu", "aten::add", "aten::quantize_per_tensor"],
            #     ["aten::quantize_per_channel"],
            #     ["aten::dequantize", "aten::_convolution", "aten::quantize_per_tensor"],
            #     ["aten::dequantize"]
            # ]
            # self.checkPatterns(graph, patterns)

class TestModel(JitLlgaTestCase):
    @skipIfNoTorchVision
    def _test_vision(self, model_name):
        m = getattr(torchvision.models, model_name)().eval()
        x = torch.rand(1, 3, 224, 224) / 10

        graph = self.checkQuantizeTrace(m, [x], atol=2e-1, folding=True, config_name=model_name)
        
        # TODO: aten::adaptive_avg_pool2d also need to be fused once backend supported it
        self.assertFused(graph, ['aten::_convolution', 'aten::relu', 
                                 'aten::max_pool2d', 'aten::linear'
                                 'aten::quantize_per_tensor', 'aten::quantize_per_channel',
                                 'aten::dequantize'])


for model_name, enabled in [
    ['resnet50', True],
]:
    def wrapper(mname):
        @unittest.skipIf(not enabled, 'Disabled')
        def test(self):
            return self._test_vision(mname)
        return test

    setattr(TestModel, 'test_vision_%s' % model_name, wrapper(model_name))

if __name__ == '__main__':
    run_tests()
