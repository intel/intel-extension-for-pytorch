import unittest
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from test_jit_llga_utils import JitLlgaTestCase, run_tests, LLGA_FUSION_GROUP
from torch.testing._internal.common_utils import TEST_SCIPY

import intel_extension_for_pytorch as ipex

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
                bias,
                memory_format
            ] in itertools.product(
                [7, 8],
                [8, 15],
                [7, 16],
                [3, 4],
                [0, 2],
                [1, 2],
                [1, 2],
                [1, 2],
                [True, False],
                [torch.contiguous_format, torch.channels_last]):

            m = nn.Conv2d(in_channels=in_channels * g,
                          out_channels=out_channels * g,
                          kernel_size=kernel,
                          padding=padding,
                          stride=stride,
                          dilation=dilation,
                          groups=g,
                          bias=bias)
            x = torch.rand(1, in_channels * g, spatial, spatial).to(memory_format=memory_format)
            patterns = [
                ["aten::dequantize", "aten::_convolution"]
            ]
            #TODO: enable torch.per_tensor_symmetric case.
            for qscheme in [torch.per_tensor_affine]:
                graph = self.checkQuantizeTrace(m, [x], x_var=[torch.rand(5, in_channels * g, spatial, spatial, requires_grad=False)], atol=2e-1, config_name="conv2d", qscheme=qscheme)
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
                self.assertFused(graph, ['aten::_convolution', 'aten::dequantize'])
                self.checkPatterns(graph, patterns)

    def test_linear_int8_in_f32_out(self):
        for bias in [True, False]:
            x = torch.rand(32, 28)
            m = torch.nn.Linear(in_features=28, out_features=64, bias=bias)

            patterns = [
                ["aten::dequantize", "aten::linear"],
            ]
            for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
                graph = self.checkQuantizeTrace(m, [x], atol=1e-1, config_name="linear", qscheme=qscheme)
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
                self.assertFused(graph, ['aten::linear', 'aten::dequantize'])
                self.checkPatterns(graph, patterns)

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

            patterns = [
                ["aten::dequantize", "aten::linear", "aten::quantize_per_tensor"],
                ["aten::dequantize", "aten::linear"]
            ]

            for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
                graph = self.checkQuantizeTrace(m, [x, y], atol=2e-1, config_name="linear_int8", qscheme=qscheme)
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 2)
                self.assertFused(graph, ['aten::linear', 'aten::quantize_per_channel', 'aten::dequantize'])
                self.checkPatterns(graph, patterns)

    def test_linear_int8_in_bf16_out(self):
        class M(nn.Module):
            def __init__(self, bias):
                super(M, self).__init__()
                self.linear1 = nn.Linear(15, 20, bias=bias)

            def forward(self, x):
                x = self.linear1(x)
                return x

        for bias in [True]: # TODO：[True, False] when supported in backend
            x = torch.randn(2, 15)
            m = M(bias)

            patterns = [
                ["aten::to", "aten::quantize_per_tensor"],
                ["aten::dequantize", "aten::to", "aten::linear"],
            ]

            for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
                graph = self.checkQuantizeTrace(m, [x], atol=2e-1, config_name="linear_int8_bf16", qscheme=qscheme, int8_bf16=True)
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 2)
                # single aten::to won't be rewritten by llga backend
                self.assertFused(graph, ['aten::dequantize', 'aten::linear'])
                self.checkPatterns(graph, patterns)                

    def test_max_pool2d(self):
        for [
                spatial,
                kernel,
                padding,
                stride,
                dilation,
                ceil_mode,
                memory_format
            ] in itertools.product(
                [15], # [15, 16], TODO: check backend
                [3, 5], # [3, 4, 5], TODO: check backend
                [0, 1],
                [1, 2], # [1, 2, 4], TODO: fix issue in pad calculation
                [1, 2],
                [True, False],
                [torch.contiguous_format, torch.channels_last]):

            m = nn.MaxPool2d(kernel_size=kernel,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             ceil_mode=ceil_mode)
            x = torch.rand(1, 3, spatial, spatial).to(memory_format=memory_format)

            patterns = [
                ["aten::dequantize", "aten::max_pool2d", "aten::quantize_per_tensor"],
            ]
            for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
                graph = self.checkQuantizeTrace(m, [x], atol=1e-1, config_name="max_pool2d", qscheme=qscheme)
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
                self.assertFused(graph, ['aten::max_pool2d'])
                self.checkPatterns(graph, patterns)

    def test_bmm_int8_in_f32_out(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()

            def forward(self, x, y):
                mm_res = torch.matmul(x, y)
                return mm_res

        x = torch.randn(128, 16, 384, 64) * 0.1
        y = torch.randn(128, 1, 64, 384) * 0.1
        patterns = [
            ["aten::dequantize", "aten::matmul"],
        ]
        m = M()
        graph = self.checkQuantizeTrace(m, [x, y], atol=2e-1, config_name="bmm", qscheme=torch.per_tensor_affine)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        self.assertFused(graph, ['aten::matmul'])
        self.checkPatterns(graph, patterns)

    def test_strided_bmm_int8_in_bf16_out(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.num_attention_heads = 16
                self.attention_head_size = 4

            def forward(self, x, y):
                new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
                x = x.view(*new_x_shape)
                z1 = x.permute(0, 2, 1, 3)

                new_y_shape2 = y.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
                y = y.view(*new_y_shape2)
                z2 = y.permute(0, 2, 1, 3)
                
                # inputs to matmul has been permuted or transposed, thus are strided tensor
                return torch.matmul(z1, z2.transpose(-1, -2))
        m = M()
        x = torch.randn(2, 3, 64)
        y = torch.randn(2, 3, 64)

        patterns = [
                ["aten::to", "aten::quantize_per_tensor"],
                ["aten::to", "aten::quantize_per_tensor"],
                ["aten::dequantize", "aten::to", "aten::matmul"],
        ]

        graph = self.checkQuantizeTrace(m, [x, y], atol=2e-1, config_name="strided_bmm_int8_in_bf16_out", qscheme=torch.per_tensor_affine, int8_bf16=True)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)
        self.assertFused(graph, ['aten::matmul', 'aten::dequantize', 'aten::quantize_per_tensor'])
        self.checkPatterns(graph, patterns)

    def test_add_scalar_input(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()

            def forward(self, x):
                x_shape = x.size()[0]
                y = x_shape + 2
                return y
        
        # input[0] to add being scalar is unsupported
        x = torch.randn(3, 3)
        m = M()
        graph = self.checkQuantizeTrace(m, [x], atol=2e-1, config_name="add_scalar_input", qscheme=torch.per_tensor_affine)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)
        self.assertGraphContainsExactly(graph, "aten::add", 1)

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
                for memory_format in [torch.contiguous_format, torch.channels_last]:
                    eltwise_fn_name = eltwise + '_' if inplace else eltwise
                    eltwise_fn = get_eltwise_fn(eltwise_fn_name)

                    m = M(eltwise_fn)
                    x = torch.rand(1, 32, 28, 28).to(memory_format=memory_format)

                    patterns = [
                        ["aten::dequantize", "aten::_convolution", 'aten::' + eltwise, "aten::quantize_per_tensor"], # inplace op will become outplace op on the JIT graph
                        ["aten::dequantize", "aten::_convolution"]
                    ]
                    for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
                        graph = self.checkQuantizeTrace(m, [x], atol=2e-1, config_name="conv2d_eltwise", qscheme=qscheme)
                        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 2)
                        self.assertFused(graph, ['aten::_convolution', 'aten::' + eltwise, 'aten::quantize_per_channel', 'aten::dequantize'])
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
            for memory_format in [torch.contiguous_format, torch.channels_last]:
                m = M(bias).eval()
                x = torch.rand(1, 32, 16, 16).to(memory_format=memory_format)
                # TODO: This shape will fail
                # x = torch.rand(1, 32, 28, 28)

                patterns = [
                    ["aten::dequantize", "aten::_convolution"]
                ]
                # TODO: add torch.per_tensor_symmetric case.
                for qscheme in [torch.per_tensor_affine]:
                    graph = self.checkQuantizeTrace(m, [x], atol=1e-1, folding=True, config_name="conv2d_bn", qscheme=qscheme)
                    self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
                    self.assertFused(graph, ['aten::_convolution', 'aten::quantize_per_channel', 'aten::dequantize'])
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
        for memory_format in [torch.contiguous_format, torch.channels_last]:
            m = M().eval()
            x = torch.rand(1, 32, 28, 28).to(memory_format=memory_format)
            patterns = [
                ["aten::dequantize", "aten::_convolution", "aten::relu", "aten::quantize_per_tensor"],
            ]
            for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
                graph = self.checkQuantizeTrace(m, [x], atol=1e-1, folding=True, config_name="conv2d_bn_relu", qscheme=qscheme)
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
                self.assertFused(graph, ['aten::_convolution', 'aten::relu', 'aten::quantize_per_channel'])
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
            patterns = [
                ["aten::dequantize", "aten::linear", "aten::" + eltwise, "aten::quantize_per_tensor"],
            ]
            for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
                graph = self.checkQuantizeTrace(m, [x], x_var=[torch.rand(2, 28, requires_grad=False)], atol=1e-1, config_name="linear_eltwise", qscheme=qscheme)
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
                self.assertFused(graph, ['aten::' + eltwise])
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
            for memory_format in [torch.contiguous_format, torch.channels_last]:
                m = M(bias).eval()
                x = torch.rand(1, 32, 16, 16, requires_grad=False).to(memory_format=memory_format)
                y = torch.rand(1, 32, 16, 16, requires_grad=False).to(memory_format=memory_format)
                patterns = [
                    ["aten::dequantize", "aten::_convolution", "aten::quantize_per_tensor"],
                    ["aten::dequantize", "aten::_convolution", "aten::relu", "aten::add", "aten::quantize_per_tensor"],
                    ["aten::dequantize", "aten::_convolution"]
                ]
                for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
                    graph = self.checkQuantizeTrace(m, [x, y], folding=True, atol=1e-1, config_name="conv2d_sum", qscheme=qscheme)
                    self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)
                    self.assertFused(graph, ['aten::_convolution', 'aten::relu', 'aten::add', 'aten::quantize_per_channel', 'aten::dequantize'])
                    self.checkPatterns(graph, patterns)

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
        patterns = [
            ["aten::dequantize", "aten::linear", "aten::add", "aten::quantize_per_tensor"],
            ["aten::dequantize", "aten::linear"]
        ]
        for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
            graph = self.checkQuantizeTrace(m, [x, y], atol=2e-1, remove_dropout=True, config_name="linear_dropout_sum", qscheme=qscheme)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 2)
            self.assertFused(graph, ['aten::linear', 'aten::add', 'aten::quantize_per_channel', 'aten::dequantize'])
        self.checkPatterns(graph, patterns)

    def test_linear_dropout_sum_bf16(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.linear1 = nn.Linear(15, 20, bias=True)
                self.dropout = nn.Dropout()
                self.linear2 = nn.Linear(15, 20, bias=True)

            def forward(self, x, y):
                x = self.linear1(x)
                x = self.dropout(x)
                z = self.linear2(y) + x
                return z
        x = torch.randn(2, 15)
        y = torch.randn(2, 15)
        m = M()
        patterns = [
            ["aten::to", "aten::quantize_per_tensor"],
            ["aten::to", "aten::quantize_per_tensor"],
            ["aten::dequantize", "aten::to", "aten::linear", "aten::to", "aten::quantize_per_tensor"],
            ["aten::dequantize", "aten::to", "aten::linear", "aten::add"]
        ]
        graph = self.checkQuantizeTrace(m, [x, y], atol=2e-1, remove_dropout=True,
                                        config_name="linear_dropout_sum", qscheme=torch.per_tensor_affine, int8_bf16=True)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 4)
        self.assertFused(graph, ['aten::linear', 'aten::add', 'aten::dequantize'])
        self.checkPatterns(graph, patterns)

    def test_linear_gelu_bf16(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.linear = nn.Linear(28, 64, bias=True)
                self.eltwise = nn.GELU()
                self.linear2 = nn.Linear(64, 1, bias=True)

            def forward(self, x):
                x = self.linear(x)
                x = self.eltwise(x)
                x = self.linear2(x)
                return x
        patterns = [
            ["aten::to", "aten::quantize_per_tensor"],
            ["aten::dequantize", "aten::to", "aten::linear", "aten::gelu", "aten::to", "aten::quantize_per_tensor"],
            ["aten::dequantize", "aten::to", "aten::linear"]
        ]        
        m = M()
        x = torch.rand(32, 28, requires_grad=False)
        for qscheme in [torch.per_tensor_affine]:
            graph = self.checkQuantizeTrace(m, [x], x_var=[torch.rand(2, 28, requires_grad=False)], atol=1e-1, config_name="linear_eltwise", qscheme=qscheme, int8_bf16=True)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)
            self.assertFused(graph, ['aten::dequantize', 'aten::linear', 'aten::gelu'])
            self.checkPatterns(graph, patterns)

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
        for memory_format in [torch.contiguous_format, torch.channels_last]:
            m = M()
            x = torch.rand(1, 32, 28, 28).to(memory_format=memory_format)
            patterns = [
                ["aten::dequantize", "aten::_convolution", 'aten::relu', "aten::quantize_per_tensor"],
                ["aten::dequantize", "aten::_convolution"]
            ]
            for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
                graph = self.checkQuantizeTrace(m, [x], atol=2e-1, config_name="defer_size", qscheme=qscheme)
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 2)
                self.assertFused(graph, ['aten::_convolution', 'aten::relu', 'aten::quantize_per_channel', 'aten::dequantize'])
                self.checkPatterns(graph, patterns)

    def test_lift_up_quant(self):
        class M(nn.Module):
            def __init__(self, bias):
                super(M, self).__init__()
                self.linear = nn.Linear(28, 64, bias)
                self.linear2 = nn.Linear(28, 64, bias=True)
                self.num_attention_heads = 16
                self.attention_head_size = 4

            def forward(self, x, y):
                x = self.linear(x)
                new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
                x = x.view(*new_x_shape)
                z1 = x.permute(0, 2, 1, 3)

                y = self.linear2(y)
                new_y_shape2 = y.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
                y = y.view(*new_y_shape2)
                z2 = y.permute(0, 2, 1, 3)

                return torch.matmul(z1, z2.transpose(-1, -2))

        m = M(bias=True)
        x = torch.randn(2, 3, 28)
        y = torch.randn(2, 3, 28)

        patterns = [
            ["aten::dequantize", "aten::linear", "aten::quantize_per_tensor"],
            ["aten::dequantize","aten::linear", "aten::quantize_per_tensor"],
            ["aten::dequantize", "aten::matmul"]
        ]

        # TODO: test shape fallback
        graph = self.checkQuantizeTrace(m, [x, y], atol=1e-1, config_name="lift_up_to_quant", qscheme=torch.per_tensor_affine)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)
        self.assertFused(graph, ['aten::dequantize', 'aten::linear', 'aten::matmul'])
        self.checkPatterns(graph, patterns)

    def test_lift_up_to_quant(self):
        class M(nn.Module):
            def __init__(self, bias):
                super(M, self).__init__()
                self.linear = nn.Linear(28, 64, bias)
                self.linear2 = nn.Linear(28, 64, bias=True)
                self.num_attention_heads = 16
                self.attention_head_size = 4

            def forward(self, x, y):
                x = self.linear(x)
                new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
                x = x.view(*new_x_shape)
                z1 = x.permute(0, 2, 1, 3)

                y = self.linear2(y)
                new_y_shape2 = y.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
                y = y.view(*new_y_shape2)
                z2 = y.permute(0, 2, 1, 3)

                return torch.matmul(z1, z2.transpose(-1, -2))

        m = M(bias=True)
        x = torch.randn(2, 3, 28)
        y = torch.randn(2, 3, 28)

        patterns = [
            ["aten::to", "aten::quantize_per_tensor"],
            ["aten::dequantize", "aten::to", "aten::linear", "aten::to", "aten::quantize_per_tensor"],
            ["aten::to", "aten::quantize_per_tensor"],
            ["aten::dequantize", "aten::to", "aten::linear", "aten::to", "aten::quantize_per_tensor"],
            ["aten::dequantize", "aten::to", "aten::matmul"]
        ]

        # TODO: test shape fallback
        graph = self.checkQuantizeTrace(m, [x, y], atol=1e-1, config_name="lift_up_to_quant", qscheme=torch.per_tensor_affine, int8_bf16=True)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 5)
        self.assertFused(graph, ['aten::dequantize', 'aten::linear', 'aten::matmul'])
        self.checkPatterns(graph, patterns)    

    def test_wildcard(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.eltwise = nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                y = self.eltwise(x)
                return [x, y]

        # The pattern is as the following:
        #      conv
        #     |    \
        # eltwise   \
        #    |       \
        #  ListConstruct
        #
        # The output of conv is used by a wildcard op: ListConstruct.
        # Thus conv-eltwise cannot be selected into the same Partition.
        m = M()
        x = torch.rand(1, 32, 28, 28)
        patterns = [
                ["aten::dequantize", "aten::_convolution"],
        ]
        graph = self.checkQuantizeTrace(m, [x], atol=2e-1, config_name="defer_size", qscheme=torch.per_tensor_affine)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        self.assertGraphContainsExactly(graph, "aten::relu", 1)
        self.assertFused(graph, ['aten::_convolution', 'aten::quantize_per_channel'])
        self.checkPatterns(graph, patterns)

    def test_bmm_div_scalar(self):
        class M(nn.Module):
            def __init__(self, div_value):
                super(M, self).__init__()
                self.div_value = div_value

            def forward(self, x, y):
                mm_res = torch.matmul(x, y)
                return mm_res.div(self.div_value)

        x = torch.randn(1, 16, 384, 64)
        y = torch.randn(1, 1, 64, 384)
        patterns = [
                ["aten::dequantize", "aten::matmul", "aten::div"],
        ]
        m = M(8.)
        graph = self.checkQuantizeTrace(m, [x, y], atol=2e-1, config_name="bmm_div_scalar", qscheme=torch.per_tensor_affine)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        self.assertFused(graph, ['aten::matmul', 'aten::div'])
        self.checkPatterns(graph, patterns)

    def test_bmm_div_identity(self):
        class M(nn.Module):
            def __init__(self, div_value):
                super(M, self).__init__()
                self.div_value = div_value

            def forward(self, x, y):
                mm_res = torch.matmul(x, y)
                return mm_res.div(self.div_value)

        x = torch.randn(1, 16, 384, 64) * 0.1
        y = torch.randn(1, 1, 64, 384) * 0.1
        patterns = [
                ["aten::dequantize", "aten::matmul"],
        ]
        m = M(1.)
        graph = self.checkQuantizeTrace(m, [x, y], atol=2e-1, config_name="bmm_div_identity", qscheme=torch.per_tensor_affine)
        # divide by 1 should be removed by Constant Propagation
        self.assertGraphContainsExactly(graph, "aten::div", 0, consider_subgraphs=True)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        self.assertFused(graph, ['aten::matmul'])
        self.checkPatterns(graph, patterns)

    def test_bmm_div_tensor(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()

            def forward(self, x, y, z):
                mm_res = torch.matmul(x, y)
                return mm_res.div(z)

        x = torch.randn(1, 16, 384, 64) * 0.1
        y = torch.randn(1, 1, 64, 384) * 0.1
        z = torch.randn(1) # TODO: enable torch.randn(20) and torch.randn(1, 1, 20, 20) once backend supported them
        patterns = [
                ["aten::dequantize", "aten::matmul", "aten::div"],
        ]
        m = M()
        graph = self.checkQuantizeTrace(m, [x, y, z], atol=2e-1, config_name="bmm_div_tensor", qscheme=torch.per_tensor_affine)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        self.assertFused(graph, ['aten::matmul', 'aten::div'])
        self.checkPatterns(graph, patterns)

    def test_bmm_div_int8_in_bf16_out(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()

            def forward(self, x, y):
                mm_res = torch.matmul(x, y) / 2
                return mm_res

        x = torch.randn(1, 16, 384, 64) * 0.1
        y = torch.randn(1, 1, 64, 384) * 0.1
        patterns = [
            ["aten::to", "aten::quantize_per_tensor"],
            ["aten::to", "aten::quantize_per_tensor"],
            ["aten::dequantize", "aten::to", "aten::matmul", "aten::div"],
        ]
        m = M()
        graph = self.checkQuantizeTrace(m, [x, y], atol=2e-1, config_name="bmm", qscheme=torch.per_tensor_affine, int8_bf16=True)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)
        # single aten::to won't be rewritten by llga backend
        self.assertFused(graph, ['aten::dequantize', 'aten::matmul', 'aten::div'])
        self.checkPatterns(graph, patterns)  

    def test_strided_bmm_div_int8_in_bf16_out(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.num_attention_heads = 16
                self.attention_head_size = 4

            def forward(self, x, y):
                new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
                x = x.view(*new_x_shape)
                z1 = x.permute(0, 2, 1, 3)

                new_y_shape2 = y.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
                y = y.view(*new_y_shape2)
                z2 = y.permute(0, 2, 1, 3)
                
                # inputs to matmul has been permuted or transposed, thus are strided tensor
                return torch.matmul(z1, z2.transpose(-1, -2)) / 0.4
        m = M()
        x = torch.randn(2, 3, 64)
        y = torch.randn(2, 3, 64)

        patterns = [
                ["aten::to", "aten::quantize_per_tensor"],
                ["aten::to", "aten::quantize_per_tensor"],
                ["aten::dequantize", "aten::to", "aten::matmul", "aten::div"],
        ]

        graph = self.checkQuantizeTrace(m, [x, y], atol=2e-1, config_name="strided_bmm_int8_in_bf16_out", qscheme=torch.per_tensor_affine, int8_bf16=True)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)
        self.assertFused(graph, ['aten::matmul', 'aten::dequantize', 'aten::quantize_per_tensor'])
        self.checkPatterns(graph, patterns)

    def test_split_dequant_to(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.linear1 = nn.Linear(2, 1, bias=True)
                self.linear2 = nn.Linear(2, 1, bias=True)
                self.linear3 = nn.Linear(2, 1, bias=True)

            def forward(self, x):
                a = self.linear1(x)
                b = self.linear2(x)
                c = self.linear3(x)
                return torch.cat([a, b, c])

        # The below pattern:
        #           to
        #           |
        #         quant
        #           |
        #        dequant
        #           |
        #          to
        #     /    |    \
        # linear linear linear
        #    |     |      |
        #
        # should be transformed to:
        #               to
        #               |
        #             quant
        #        /      |     \
        #   dequant dequant  dequant
        #      |       |       |
        #     to       to     to
        #      |       |       |
        #  linear   linear  linear
        #      |       |       |

        patterns = [
            ["aten::to", "aten::quantize_per_tensor"],
            ["aten::dequantize", "aten::to", "aten::linear"],
            ["aten::dequantize", "aten::to", "aten::linear"],
            ["aten::dequantize", "aten::to", "aten::linear"],
        ]
        m = M()
        x = torch.randn(2, 2)
        graph = self.checkQuantizeTrace(m, [x], atol=2e-1, config_name="split_dequant_to",
                                        qscheme=torch.per_tensor_affine, int8_bf16=True)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 4)
        # single aten::to won't be rewritten by llga backend
        self.assertFused(graph, ['aten::dequantize', 'aten::linear'])
        self.checkPatterns(graph, patterns)


class TestShapeFallback(JitLlgaTestCase):
    @unittest.skipIf(True, 'Size peephole optimization not enabled yet')
    def test_view_permute(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()

            def forward(self, x):
                new_x_shape = x.size()[:-1] + (3, 5)
                x = x.view(*new_x_shape)
                return x.permute(0, 2, 1, 3)

        x = torch.randn(5, 10, 15)
        m = M()

        for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
            graph = self.checkQuantizeTrace(m, [x], config_name="view_permute", qscheme=qscheme)
            self.assertGraphContainsExactly(graph, "aten::size", 0)
            self.assertGraphContainsExactly(graph, "prim::ListConstruct", 0)

            # change the size of the input
            x2 = torch.randn(6, 4, 15)
            # Bailout get triggered here
            y2 = m(x2)

    def test_conv_reshape(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv1 = nn.Conv2d(4, 4, 3, padding=1, bias=True)
                self.conv2 = nn.Conv2d(4, 32, 3, padding=1, bias=True)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x).reshape(x.size(0), 4, -1)
                return x
        for memory_format in [torch.contiguous_format, torch.channels_last]:
            x = torch.randn(15, 4, 28, 28).to(memory_format=memory_format)
            # change the size of the input, check the fallback
            x_var = torch.randn(7, 4, 16, 16).to(memory_format=memory_format)
            m = M()
            for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
                graph = self.checkQuantizeTrace(m, [x], x_var = [x_var], atol=2e-1, config_name="conv_reshape", qscheme=qscheme)

                # TODO: enable this check when size peephole optimization is enabled
                # self.assertGraphContainsExactly(graph, "aten::size", 0)

class TestModel(JitLlgaTestCase):
    @skipIfNoTorchVision
    def _test_vision(self, model_name):
        for memory_format in [torch.contiguous_format, torch.channels_last]:
            m = getattr(torchvision.models, model_name)().eval()
            x = (torch.rand(1, 3, 224, 224) / 10).to(memory_format=memory_format)

            for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
                graph = self.checkQuantizeTrace(m, [x], atol=2e-1, folding=True, config_name=model_name, qscheme=qscheme)

                # TODO: aten::adaptive_avg_pool2d also need to be fused once backend supported it
                self.assertFused(graph, ['aten::_convolution', 'aten::relu',
                                        'aten::max_pool2d', 'aten::linear', 'aten::quantize_per_channel'])


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
