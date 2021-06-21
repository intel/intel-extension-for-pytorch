import unittest
import itertools
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F
from test_jit_llga_utils import JitLlgaTestCase, run_tests, LLGA_FUSION_GROUP
from torch.testing._internal.common_utils import TEST_SCIPY

import intel_pytorch_extension as ipex

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

# For LLGA UT, disable the PyTorch profiling executor and the IPEX JIT opt
def llga_test_env(func):
    @wraps(func)
    def wrapTheFunction(*args):
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        ipex.core.disable_jit_opt()
        func(*args)
        ipex.core.enable_jit_opt()
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_set_profiling_executor(True)
    return wrapTheFunction

class TestOp(JitLlgaTestCase):
    @llga_test_env
    def test_conv2d_int8_in_f32_out(self):
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
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 2)
            self.assertFused(graph, ['aten::_convolution', 'aten::quantize_per_tensor', 'aten::quantize_per_channel'])
            
            patterns = [
                ["aten::quantize_per_tensor"],
                ["aten::quantize_per_channel", "aten::dequantize", "aten::_convolution"]
            ]
            self.checkPatterns(graph, patterns)

    @llga_test_env
    def test_linear_int8_in_f32_out(self):
        for bias in [True, False]:
            x = torch.rand(32, 28)
            m = torch.nn.Linear(in_features=28, out_features=64, bias=bias)
            
            graph = self.checkQuantizeTrace(m, [x], atol=1e-1, config_name="linear")
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 2)
            self.assertFused(graph, ['aten::linear', 'aten::quantize_per_tensor', 'aten::quantize_per_channel', 'aten::dequantize'])

            patterns = [
                ["aten::quantize_per_tensor"],
                ["aten::quantize_per_channel", "aten::dequantize", "aten::linear"],
            ]
            self.checkPatterns(graph, patterns)

    @llga_test_env
    def test_linear_int8_in_int8_out(self):
        class M(nn.Module):
            def __init__(self, bias):
                super(M, self).__init__()
                self.linear1 = nn.Linear(15, 20, bias=bias)
                self.linear2 = nn.Linear(20, 3, bias=bias)

            def forward(self, x, y):
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        for bias in [True, False]:
            x = torch.randn(2, 15)
            y = torch.randn(2, 20)
            m = M(bias)
            graph = self.checkQuantizeTrace(m, [x, y], atol=2e-1, config_name="linear_int8")
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)
            self.assertFused(graph, ['aten::linear',
                                    'aten::quantize_per_tensor', 'aten::quantize_per_channel', 'aten::dequantize'])

            patterns = [
                ["aten::quantize_per_tensor"],
                ["aten::quantize_per_channel", "aten::dequantize", "aten::linear", "aten::quantize_per_tensor"],
                ["aten::quantize_per_channel", "aten::dequantize", "aten::linear"]
            ]
            self.checkPatterns(graph, patterns)

    @llga_test_env
    def test_max_pool2d(self):
        for [
                spatial,
                kernel,
                padding,
                stride,
                dilation,
                ceil_mode
            ] in itertools.product(
                [15], # [15, 16], TODO: check backend
                [3, 5], # [3, 4, 5], TODO: check backend
                [0, 1],
                [1, 2], # [1, 2, 4], TODO: fix issue in pad calculation
                [1, 2],
                [True, False]):

            m = nn.MaxPool2d(kernel_size=kernel,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             ceil_mode=ceil_mode)
            x = torch.rand(1, 3, spatial, spatial)

            graph = self.checkQuantizeTrace(m, [x], atol=1e-1, config_name="max_pool2d")
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)
            self.assertFused(graph, ['aten::max_pool2d', 'aten::quantize_per_tensor', 'aten::dequantize'])

            patterns = [
                ["aten::quantize_per_tensor"],
                ["aten::dequantize", "aten::max_pool2d", "aten::quantize_per_tensor"],
                ["aten::dequantize"]
            ]
            self.checkPatterns(graph, patterns)

    @llga_test_env
    @unittest.skipIf(True, 'int8 adaptive_avg_pool2d is not supported in the backend')
    def test_adaptive_avg_pool2d(self):
        m = nn.AdaptiveAvgPool2d((1, 1))
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 224, 224, dtype=torch.float32) * 100

        graph = self.checkQuantizeTrace(m, [x], atol=1e-1, config_name="adaptive_avg_pool2d")
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)
        self.assertFused(graph, ['aten::adaptive_avg_pool2d', 'aten::quantize_per_tensor', 'aten::dequantize'])

        patterns = [
            ["aten::quantize_per_tensor"],
            ["aten::dequantize", "aten::adaptive_avg_pool2d", "aten::quantize_per_tensor"],
            ["aten::dequantize"]
        ]
        self.checkPatterns(graph, patterns)

class TestFusionPattern(JitLlgaTestCase):
    @llga_test_env
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
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)
                self.assertFused(graph, ['aten::_convolution', 'aten::' + eltwise, 'aten::quantize_per_tensor', 'aten::quantize_per_channel', 'aten::dequantize'])

                patterns = [
                    ["aten::quantize_per_tensor"],
                    ["aten::quantize_per_channel", "aten::dequantize", "aten::_convolution", 'aten::' + eltwise, "aten::quantize_per_tensor"], # inplace op will become outplace op on the JIT graph
                    ["aten::quantize_per_channel", "aten::dequantize", "aten::_convolution"]
                ]
                self.checkPatterns(graph, patterns)

    @llga_test_env
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
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 2)
            self.assertFused(graph, ['aten::_convolution', 'aten::quantize_per_tensor', 'aten::quantize_per_channel'])

            patterns = [
                ["aten::quantize_per_tensor"],
                ["aten::quantize_per_channel", "aten::dequantize", "aten::_convolution"]
            ]
            self.checkPatterns(graph, patterns)

    @llga_test_env
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
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)
        self.assertFused(graph, ['aten::_convolution', 'aten::relu',
                                 'aten::quantize_per_tensor', 'aten::quantize_per_channel', 'aten::dequantize'])

        patterns = [
            ["aten::quantize_per_tensor"],
            ["aten::quantize_per_channel", "aten::dequantize", "aten::_convolution", "aten::relu", "aten::quantize_per_tensor"],
            ["aten::dequantize"]
        ]
        self.checkPatterns(graph, patterns)

    @llga_test_env
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
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)
            self.assertFused(graph, ['aten::' + eltwise])

            patterns = [
                ["aten::quantize_per_tensor"],
                ["aten::quantize_per_channel", "aten::dequantize", "aten::linear", "aten::" + eltwise, "aten::quantize_per_tensor"],
                ["aten::dequantize"]
            ]
            self.checkPatterns(graph, patterns)

    @llga_test_env
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
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 5)

            patterns = [
                ["aten::quantize_per_tensor"],
                ["aten::quantize_per_tensor"],
                ["aten::quantize_per_channel", "aten::dequantize", "aten::_convolution", "aten::quantize_per_tensor"],
                ["aten::quantize_per_channel", "aten::dequantize", "aten::_convolution", "aten::relu", "aten::add", "aten::quantize_per_tensor"],
                ["aten::quantize_per_channel", "aten::dequantize", "aten::_convolution"]
            ]
            self.checkPatterns(graph, patterns)

    @llga_test_env
    def test_linear_dropout_sum(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.linear1 = nn.Linear(15, 20)
                self.dropout = nn.Dropout()
                self.linear2 = nn.Linear(20, 3)

            def forward(self, x, y):
                x = self.linear1(x)
                x = self.dropout(x)
                z = self.linear2(x + y)
                return z

        x = torch.randn(2, 15)
        y = torch.randn(2, 20)
        m = M()
        graph = self.checkQuantizeTrace(m, [x, y], atol=2e-1, remove_dropout=True, config_name="linear_dropout_sum")
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 4)
        self.assertFused(graph, ['aten::linear', 'aten::add',
                                 'aten::quantize_per_tensor', 'aten::quantize_per_channel', 'aten::dequantize'])

        patterns = [
            ["aten::quantize_per_tensor"],
            ["aten::quantize_per_tensor"],
            ["aten::quantize_per_channel", "aten::dequantize", "aten::linear", "aten::add", "aten::quantize_per_tensor"],
            ["aten::quantize_per_channel", "aten::dequantize", "aten::linear"]
        ]
        self.checkPatterns(graph, patterns)

        # TODO: check patterns when oneDNN support sum post_ops with zps
        # patterns = [
        #     ["aten::quantize_per_tensor"],
        #     ["aten::quantize_per_channel"],
        #     ["aten::dequantize", "aten::linear", "aten::add", "aten::quantize_per_tensor"],
        #     ["aten::quantize_per_channel"],
        #     ["aten::dequantize", "aten::linear", "aten::quantize_per_tensor"],
        #     ["aten::dequantize"]
        # ]
        # self.checkPatterns(graph, patterns)

    @llga_test_env
    def test_defer_size(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.eltwise = nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.eltwise(x)
                y = self.conv2(x)
                y = y.reshape(x.size(0), -1)
                return y
        
        m = M()
        x = torch.rand(1, 32, 28, 28)
        
        graph = self.checkQuantizeTrace(m, [x], atol=2e-1, config_name="defer_size")
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)
        self.assertFused(graph, ['aten::_convolution', 'aten::relu', 'aten::quantize_per_tensor', 'aten::quantize_per_channel', 'aten::dequantize'])

        patterns = [
            ["aten::quantize_per_tensor"],
            ["aten::quantize_per_channel", "aten::dequantize", "aten::_convolution", 'aten::relu', "aten::quantize_per_tensor"],
            ["aten::quantize_per_channel", "aten::dequantize", "aten::_convolution"]
        ]
        self.checkPatterns(graph, patterns)

class TestModel(JitLlgaTestCase):
    @skipIfNoTorchVision
    @llga_test_env
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
