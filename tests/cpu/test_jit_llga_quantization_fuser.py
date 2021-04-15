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
                [7],
                [8],
                [7],
                [3],
                [0],
                [1],
                [1],
                [1],
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
            
            graph = self.checkQuantizeTrace(m, x, atol=2e-1)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 4)
            self.assertFused(graph, ['aten::conv2d', 'aten::quantize_per_tensor', 'aten::quantize_per_channel', 'aten::dequantize'])
            
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
                x = self.eltwise(x)
                return x

        for eltwise in ['relu']: # TODO: ['sigmoid', 'sqrt', 'abs', 'square', 'hardtanh']
            for inplace in [False]:
                eltwise_fn_name = eltwise + '_' if inplace else eltwise
                eltwise_fn = get_eltwise_fn(eltwise_fn_name)

                m = M(eltwise_fn)
                x = torch.rand(1, 32, 28, 28)
                
                graph = self.checkQuantizeTrace(m, x, atol=2e-1)
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 6)
                self.assertFused(graph, ['aten::conv2d', 'aten::' + eltwise, 'aten::quantize_per_tensor', 'aten::quantize_per_channel', 'aten::dequantize'])

    def test_conv2d_bn(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=False) # TODO: bias=True
                self.bn1 = nn.BatchNorm2d(32)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                return x
        
        m = M().eval()
        x = torch.rand(1, 32, 28, 28)
        
        graph = self.checkQuantizeTrace(m, x, atol=1e-1, folding=True)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 4)
        self.assertFused(graph, ['aten::conv2d', 'aten::quantize_per_tensor', 'aten::quantize_per_channel', 'aten::dequantize'])


class TestModel(JitLlgaTestCase):
    @skipIfNoTorchVision
    def _test_vision(self, model_name):
        m = getattr(torchvision.models, model_name)().eval()
        x = torch.rand(1, 3, 224, 224) / 10

        graph = self.checkQuantizeTrace(m, x, atol=2e-1, folding=True)
        # self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 116)
        self.assertFused(graph, ['aten::conv2d', 'aten::batch_norm',
                                 'aten::relu', 'aten::mm', 'aten::add',
                                 'aten::avg_pool2d', 'aten::max_pool2d', 
                                 'aten::linear'
                                 'aten::quantize_per_tensor', 'aten::quantize_per_channel',
                                 'aten::dequantize'])
        # self.assertFused(graph, ['aten::conv2d', 'aten::batch_norm',
        #                          'aten::relu', 'aten::mm', 'aten::add',
        #                          'aten::avg_pool2d', 'aten::max_pool2d',
        #                          'aten::quantize_per_tensor', 'aten::quantize_per_channel',
        #                          'aten::dequantize'])


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
