import sys
import os
import unittest
import itertools
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import FileCheck
import copy
from test_autocast import get_rand_seed

import intel_extension_for_pytorch as ipex
from test_ao_jit_llga_utils import JitLlgaTestCase, run_tests, LLGA_FUSION_GROUP
from torch.testing._internal.common_utils import TEST_SCIPY, TemporaryFileName

import intel_extension_for_pytorch as ipex
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, HistogramObserver, QConfig

default_weight_observer = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)

static_qconfig = [
        QConfig(
            activation = MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
            weight = default_weight_observer),
        QConfig(
            activation = MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
            weight = default_weight_observer),
        QConfig(
            activation = HistogramObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8, reduce_range=True),
            weight = default_weight_observer),
        QConfig(
            activation = HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8, reduce_range=True),
            weight = default_weight_observer),
        ]


class TestIpexOps(JitLlgaTestCase):
    def test_adaptive_avg_pool2d(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = nn.Conv2d(3, 3, 2, padding=1, bias=True)
                self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((5,7))

            def forward(self, x):
                x = self.conv(x)
                x = self.adaptive_avg_pool2d(x)
                x = x.relu()
                return x

        m = M()
        x = torch.rand(1, 3, 28, 28)
        patterns = [
            ["aten::dequantize", "aten::_convolution", "aten::quantize_per_tensor"],
        ]
        for qconfig in static_qconfig:
            graph = self.checkQuantizeTrace(m, [x], atol=2e-1, qconfig=qconfig)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
            self.checkPatterns(graph, patterns)

    # single none gemm ops will not be quantized if pre and post don't has
    # quantizable op.
    def test_adaptive_avg_pool2d_fp32(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((5,7))

            def forward(self, x):
                x = self.adaptive_avg_pool2d(x)
                return x

        m = M()
        x = torch.rand(1, 3, 28, 28)
        patterns = [
            ["aten::dequantize", "aten::_convolution", "aten::quantize_per_tensor"],
        ]
        for qconfig in static_qconfig:
            graph = self.checkQuantizeTrace(m, [x], qconfig=qconfig)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)
            FileCheck().check_not("aten::quantize_per_tensor") \
                .check_not("at::dequantize") \
                .check("aten::adaptive_avg_pool2d") \
                .run(graph)

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
            ["aten::dequantize", "aten::_convolution", "aten::quantize_per_tensor"],
            ["aten::dequantize", "aten::max_pool2d", "aten::quantize_per_tensor"],
            ["aten::dequantize", "aten::linear"],
        ]
        for qconfig in static_qconfig:
            graph = self.checkQuantizeTrace(m, [x], atol=2e-1, qconfig=qconfig)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)
            self.checkPatterns(graph, patterns)

    
    # single none gemm ops will not be quantized if pre and post don't has
    # quantizable op.
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
        for qconfig in static_qconfig:
            graph = self.checkQuantizeTrace(m, [x], qconfig=qconfig)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)
            FileCheck().check_not("aten::quantize_per_tensor") \
                .check_not("at::dequantize") \
                .check("aten::flatten") \
                .run(graph)

    def test_embeddingbag_int8(self):
        m = nn.EmbeddingBag(10, 3, mode='sum', sparse=True)
        input = torch.LongTensor([1,2,4,5,4,3,2,9])
        offsets = torch.LongTensor([0,1,2,3,4,5,6,7])

        graph = self.checkQuantizeTrace(m, [input, offsets], atol=1e-2, qconfig=static_qconfig[1])
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
        inputs = []
        for i in range(0, 27):
            inputs.append(torch.randn([128, 128]) * 0.1)
        graph = self.checkQuantizeTrace(m, inputs, atol=1e-2, qconfig=static_qconfig[1])
        self.assertGraphContainsExactly(graph, 'ipex::qinteraction', 1)

    '''
    # This test case will be enabled after LSTM int8->fp32 works
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
            graph = self.checkQuantizeTrace(m, [x], atol=3e-2, rtol=1e-1)
            self.assertGraphContainsExactly(graph, 'ipex::quantized_lstm', 1)
    '''
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
        for int8_bf16 in [False]:
            m_ = copy.deepcopy(m)
            for inplace in [False, True]:
                orgin_model_weight_dtype = m_.linear.weight.dtype
                orgin_model_bias_dtype = m_.linear.bias.dtype
                _, _, ori_model = self.prepareModel(m_, x, qconfig=static_qconfig[1], int8_bf16=int8_bf16, inplace=inplace)
                if inplace and int8_bf16:
                    if m_.linear.weight.dtype == orgin_model_weight_dtype or m_.linear.bias.dtype == orgin_model_bias_dtype:
                        print("model should have changed")
                        assert(0)
                else:
                    if m_.linear.weight.dtype != orgin_model_weight_dtype or m_.linear.bias.dtype != orgin_model_bias_dtype:
                        print("model should not change")
                        assert(0)

    def test_qconf_summary_save_load(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.linear = nn.Linear(128,1)

            def forward(self, x):
                x = self.linear(x)
                x = torch.relu(x)
                return x

        m = M()
        x = torch.rand(1,128)
        prepared_model = ipex.quantization.prepare(m, static_qconfig[0], example_inputs=x, inplace=False)
        prepared_model(x)
        with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "configure.json")
                prepared_model.save_qconf_summary(path)
                convert_model = ipex.quantization.convert(prepared_model, example_inputs=x)
                y_before = convert_model(x)
                # load the saved qconf
                prepared_model = ipex.quantization.prepare(m, static_qconfig[0], example_inputs=x, inplace=False)
                prepared_model.load_qconf_summary(path)
                convert_model = ipex.quantization.convert(prepared_model, example_inputs=x)
                y_after = convert_model(x)
                self.assertEqual(y_before, y_after)

class TestRemoveMutate(JitLlgaTestCase):
    def test_mutated_value_alive_after_inplace_op(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, 224)

            def forward(self, x):
                a = self.conv(x)
                b = torch.sigmoid(a)
                c = a[0]
                a.mul_(b)
                c += 2
                return c

        m = M()
        x = torch.randn(1, 3, 224, 224)
        graph, _, _ = self.prepareModel(m, [x])
        FileCheck().check_not("aten::mul").check("aten::mul_").run(graph)

    def test_mutated_value_inalive_after_inplace_op(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, 224)

            def forward(self, x):
                a = self.conv(x)
                b = torch.sigmoid(a)
                res = a.mul_(b)
                return res

        m = M()
        x = torch.randn(1, 3, 224, 224)
        graph, _, _ = self.prepareModel(m, [x])
        FileCheck().check_not("aten::mul_").check("aten::mul").run(graph)


if __name__ == '__main__':
    run_tests()
