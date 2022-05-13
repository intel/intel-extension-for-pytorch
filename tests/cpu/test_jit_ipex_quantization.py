import sys
import unittest
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import FileCheck
import copy
from test_jit_llga_utils import JitLlgaTestCase, run_tests, LLGA_FUSION_GROUP
from test_autocast import get_rand_seed

import intel_extension_for_pytorch as ipex


class TestIpexOps(JitLlgaTestCase):
    def test_adaptive_avg_pool2d(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((5,7))

            def forward(self, x):
                x = self.adaptive_avg_pool2d(x)
                return x

        m = M()
        x = torch.rand(1, 32, 28, 28)
        for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
            graph = self.checkQuantizeTrace(m, [x], atol=2e-1, config_name="adaptive_avg_pool2d", qscheme=qscheme)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)


    def test_flatten_int8(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv1 = nn.Conv2d(3, 3, 2, padding=1, bias=True)
                self.pool = nn.MaxPool2d(2)
                self.flatten = nn.Flatten(1)
                self.linear = nn.Linear(147, 32)

            def forward(self, x):
                x = self.conv1(x)
                x = self.pool(x)
                x = self.flatten(x)
                x = self.linear(x)
                return x

        m = M()
        x = torch.rand(1, 3, 14, 14)
        patterns = [
            ["aten::dequantize", "aten::_convolution"],
            ["aten::dequantize", "aten::max_pool2d", "aten::quantize_per_tensor"],
            ["aten::dequantize", "aten::linear"],
        ]
        for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
            graph = self.checkQuantizeTrace(m, [x], atol=2e-1, config_name="flatten", qscheme=qscheme)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)
            self.checkPatterns(graph, patterns)

    def test_flatten_fp32(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.flatten = nn.Flatten(1)

            def forward(self, x):
                x = self.flatten(x)
                return x

        m = M()
        x = torch.rand(1, 3, 14, 14)
        for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
            graph = self.checkQuantizeTrace(m, [x], config_name="flatten", qscheme=qscheme)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)
            FileCheck().check_not("aten::quantize_per_tensor") \
                .check_not("at::dequantize") \
                .check("aten::flatten") \
                .run(graph)

    def test_embeddingbag_int8(self):
        m = nn.EmbeddingBag(10, 3, mode='sum', sparse=True)
        input = torch.LongTensor([1,2,4,5,4,3,2,9])
        offsets = torch.LongTensor([0,1,2,3,4,5,6,7])
        for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
            graph = self.checkQuantizeTrace(m, [input, offsets], atol=1e-2, config_name="emb", qscheme=qscheme)
            self.assertGraphContainsExactly(graph, 'ipex::qembedding_bag', 1)

    def test_interaction_int8(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.f = ipex.nn.functional.interaction

            def forward(self, *x):
                x = self.f(*x)
                return x

        m = M()
        for ninputs in [2, 27]:
            inputs = []
            for i in range(0, ninputs):
                inputs.append(torch.randn([128, 128]) * 0.1)
            for qscheme in [torch.per_tensor_symmetric]:
                graph = self.checkQuantizeTrace(m, inputs, atol=1e-2, config_name="interaction", qscheme=qscheme)
                self.assertGraphContainsExactly(graph, 'ipex::qinteraction', 1)

    def test_lstm(self):
        class M(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, bidirectional=False, bias=False, dropout=0, batch_first=False):
                super(M, self).__init__()
                self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, bias=bias, dropout=dropout, batch_first=batch_first)

            def forward(self, x, h = None):
                x, h = self.lstm(x, h)
                return x, h

        def _lstm_params_list():
            params_dict = {
                "input_size": [1, 32],
                "hidden_size": [16],
                "num_layers": [3],
                "bidirectional": [False, True],
                "bias": [False, True],
                "empty_state": [False, True],
                "batch_first": [False, True],
                "dropout": [0, 0.4, 1],
                "batch_size": [1, 2],
                "seq_len": [48]
            }

            params_list = []
            for key, value in params_dict.items():
                params_list.append(value)
            return params_list


        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)

        params_list = _lstm_params_list()
        for input_size, hidden_size, num_layers, bidirectional, bias, empty_state, batch_first, dropout, batch_size, seq_len in itertools.product(*params_list):
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

            m = M(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, bias=bias, dropout=dropout, batch_first=batch_first)

            graph = self.checkQuantizeTrace(m, [x], atol=3e-2, rtol=1e-1, config_name="lstm")
            self.assertGraphContainsExactly(graph, 'ipex::quantized_lstm', 1)

class TestIpexQuantizationConvertAPI(JitLlgaTestCase):
    def test_inplace_convert(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.linear = nn.Linear(128,1)

            def forward(self, x):
                x = self.linear(x)
                return x

        m = M()
        x = torch.rand(1,128)
        for int8_bf16 in [False, True]:
            m_ = copy.deepcopy(m)
            for inplace in [False, True]:
                orgin_model_weight_dtype = m_.linear.weight.dtype
                orgin_model_bias_dtype = m_.linear.bias.dtype
                _, _, ori_model = self.prepareModel(m_, x, folding=False, remove_dropout=False, config_name="inplace_convert", qscheme=torch.per_tensor_affine, int8_bf16=int8_bf16, inplace=inplace)
                if inplace and int8_bf16:
                    if m_.linear.weight.dtype == orgin_model_weight_dtype or m_.linear.bias.dtype == orgin_model_bias_dtype:
                        print("model should have changed")
                        assert(0)
                else:
                    if m_.linear.weight.dtype != orgin_model_weight_dtype or m_.linear.bias.dtype != orgin_model_bias_dtype:
                        print("model should not change")
                        assert(0)
if __name__ == '__main__':
    run_tests()
