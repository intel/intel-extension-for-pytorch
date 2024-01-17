import sys
import os
import itertools
import tempfile
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.testing import FileCheck
import copy
import json
from test_autocast import get_rand_seed

import intel_extension_for_pytorch as ipex
from test_ao_jit_llga_utils import (
    JitLlgaTestCase,
    LLGA_FUSION_GROUP,
    llga_fp32_bf16_test_env,
)
from torch.testing._internal.jit_utils import freeze_rng_state

from torch.ao.quantization import (
    MinMaxObserver,
    PerChannelMinMaxObserver,
    HistogramObserver,
    QConfig,
    PlaceholderObserver,
)
from torch.testing._internal.common_utils import run_tests

default_weight_observer = PerChannelMinMaxObserver.with_args(
    dtype=torch.qint8, qscheme=torch.per_channel_symmetric
)

static_qconfig = [
    QConfig(
        activation=MinMaxObserver.with_args(
            qscheme=torch.per_tensor_affine, dtype=torch.quint8
        ),
        weight=default_weight_observer,
    ),
    QConfig(
        activation=MinMaxObserver.with_args(
            qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
        ),
        weight=default_weight_observer,
    ),
    QConfig(
        activation=HistogramObserver.with_args(
            qscheme=torch.per_tensor_symmetric, dtype=torch.qint8, reduce_range=True
        ),
        weight=default_weight_observer,
    ),
    ipex.quantization.default_static_qconfig,
]

dynamic_qconfig = [
    QConfig(
        activation=PlaceholderObserver.with_args(dtype=torch.float, is_dynamic=True),
        weight=MinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        ),
    ),
    ipex.quantization.default_dynamic_qconfig,
]


class TestIpexOps(JitLlgaTestCase):
    def test_adaptive_avg_pool2d(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = nn.Conv2d(3, 3, 2, padding=1, bias=True)
                self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((5, 7))

            def forward(self, x):
                x = self.conv(x)
                x = self.adaptive_avg_pool2d(x)
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
                self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((5, 7))

            def forward(self, x):
                x = self.adaptive_avg_pool2d(x)
                return x

        m = M()
        x = torch.rand(1, 3, 28, 28)
        for qconfig in static_qconfig:
            graph = self.checkQuantizeTrace(m, [x], qconfig=qconfig)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)
            FileCheck().check_not("aten::quantize_per_tensor").check_not(
                "at::dequantize"
            ).check("aten::adaptive_avg_pool2d").run(graph)

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

        class M2(nn.Module):
            def __init__(self):
                super(M2, self).__init__()
                self.conv1 = nn.Conv2d(3, 3, 2, padding=1, bias=True)
                self.pool = nn.MaxPool2d(2)
                self.linear = nn.Linear(147, 32)

            def forward(self, x):
                x = self.conv1(x)
                x = self.pool(x)
                x = x.flatten(1)
                x = self.linear(x)
                return x

        m = M()
        m2 = M2()
        for test_m in [m, m2]:
            x = torch.rand(1, 3, 14, 14)
            patterns = [
                ["aten::dequantize", "aten::_convolution", "aten::quantize_per_tensor"],
                ["aten::dequantize", "aten::max_pool2d", "aten::quantize_per_tensor"],
                ["aten::dequantize", "aten::linear"],
            ]
            for qconfig in static_qconfig:
                graph = self.checkQuantizeTrace(test_m, [x], atol=2e-1, qconfig=qconfig)
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
            FileCheck().check_not("aten::quantize_per_tensor").check_not(
                "at::dequantize"
            ).check("aten::flatten").run(graph)

    def test_embeddingbag_int8(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.m = nn.EmbeddingBag(10, 110, mode="sum", sparse=True)

            def forward(self, input, offset):
                x = self.m(input, offset)
                return x

        def get_input(bag_size_1):
            if bag_size_1:
                return torch.LongTensor(
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                ), torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            else:
                return torch.LongTensor(
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                ), torch.LongTensor([0])

        def fake_quant(tensor, scale, zp):
            qtensor = torch.quantize_per_tensor(tensor, scale, zp, torch.qint8)
            return qtensor.dequantize()

        def get_expect(module, input, offsets):
            def _calculate_scale(max_val, min_val):
                min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
                max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
                max_val_pos = torch.max(-min_val_neg, max_val_pos)
                scale = max_val_pos / 127.5
                scale = max(scale.item(), torch.finfo(torch.float32).eps)
                return scale

            _module = copy.deepcopy(module)
            y = _module(input, offsets)
            o_scale = _calculate_scale(y.max(), y.min())
            if isinstance(_module, nn.EmbeddingBag):
                w_scale = _calculate_scale(_module.weight.max(), _module.weight.min())
                _module.weight.data = fake_quant(_module.weight, w_scale, 0)
            else:
                w_scale = _calculate_scale(
                    _module.m.weight.max(), _module.m.weight.min()
                )
                _module.m.weight.data = fake_quant(_module.m.weight, w_scale, 0)
            expect = _module(input, offsets)
            return fake_quant(expect, o_scale, 0)

        # This will call in F.embeddingbag
        with torch.no_grad():
            for bag_size_1 in [True, False]:
                input, offsets = get_input(bag_size_1)
                m = nn.EmbeddingBag(10, 110, mode="sum", sparse=True)
                y = get_expect(m, input, offsets)
                tol = 1e-2 if bag_size_1 else 5e-2
                graph = self.checkQuantizeTrace(
                    m, [input, offsets], qconfig=static_qconfig[1], expect_result=y
                )
                self.assertGraphContainsExactly(graph, "ipex::qembedding_bag", 1)
                # test nn.EmbeddingBag
                m = M().eval()
                y = get_expect(m, input, offsets)
                graph = self.checkQuantizeTrace(
                    m, [input, offsets], qconfig=static_qconfig[1], expect_result=y
                )
                self.assertGraphContainsExactly(graph, "ipex::qembedding_bag", 1)

    def test_mergedembcat_int8(self):
        # test with module
        class ModuleCall(torch.nn.Module):
            def __init__(self, NUM_TABLE, NUM_DIM):
                super(ModuleCall, self).__init__()
                emblist = torch.nn.ModuleList()
                for _ in range(NUM_TABLE):
                    emblist.append(torch.nn.EmbeddingBag(1000, NUM_DIM, mode="sum"))
                self.merged_emb = (
                    ipex.nn.modules.MergedEmbeddingBagWithCat.from_embeddingbag_list(
                        emblist
                    )
                )

            def forward(self, indices, offsets, to_cat):
                return self.merged_emb(indices, offsets, to_cat)

        # test with function
        class FunctionCall(torch.nn.Module):
            def __init__(self, NUM_TABLE, NUM_DIM):
                super(FunctionCall, self).__init__()
                self.weights = [(torch.randn(1000, NUM_DIM)) for _ in range(NUM_TABLE)]

            def forward(self, indices, offsets, to_cat):
                return torch.ops.torch_ipex.merged_embeddingbag_cat_forward(
                    self.weights, indices, offsets, to_cat
                )

        def get_input(emb_dim, num_table, batch_size):
            multi_hot = [
                2,
                3,
                1,
                2,
                5,
            ]
            indices = tuple(
                [
                    torch.randint(1000, (batch_size * multi_hot[i],))
                    for i in range(num_table)
                ]
            )
            offsets = tuple(
                [
                    torch.arange(0, batch_size * multi_hot[i], multi_hot[i])
                    for i in range(num_table)
                ]
            )
            dense = torch.randn(batch_size, emb_dim)
            return indices, offsets, dense

        def fake_quant(tensor, scale, zp):
            qtensor = torch.quantize_per_tensor(tensor, scale, zp, torch.qint8)
            return qtensor.dequantize()

        def get_expect(module, input, offsets, dense):
            def _calculate_scale(max_val, min_val):
                min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
                max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
                max_val_pos = torch.max(-min_val_neg, max_val_pos)
                scale = max_val_pos / 127.5
                scale = max(scale.item(), torch.finfo(torch.float32).eps)
                return scale

            _module = copy.deepcopy(module)
            y = _module(input, offsets, dense)
            o_scale = _calculate_scale(y.max(), y.min())
            d_scale = _calculate_scale(dense.max(), dense.min())
            dense = fake_quant(dense, d_scale, 0)

            # fake quant weight
            if hasattr(_module, "weights"):
                weights = _module.weights
            else:
                weights = _module.merged_emb.weights
            for i in range(len(weights)):
                w_scale = _calculate_scale(weights[i].max(), weights[i].min())
                weights[i].data = fake_quant(weights[i], w_scale, 0)

            expect = _module(input, offsets, dense)
            return fake_quant(expect, o_scale, 0)

        with torch.no_grad():
            for emb_dim in [128, 129]:  # fast path for 128 and general path for 129
                NUM_TABLE = 5
                BATCH_SIZE = 16
                input, offsets, dense = get_input(emb_dim, NUM_TABLE, BATCH_SIZE)
                m = ModuleCall(NUM_TABLE, emb_dim)
                y = get_expect(m, input, offsets, dense)
                graph = self.checkQuantizeTrace(
                    m,
                    [input, offsets, dense],
                    qconfig=static_qconfig[1],
                    rtol=2e-1,
                    expect_result=y,
                )
                self.assertGraphContainsExactly(
                    graph, "ipex::qmerged_embeddingbag_cat", 1
                )
                # test function call
                m = FunctionCall(NUM_TABLE, emb_dim).eval()
                y = get_expect(m, input, offsets, dense)
                graph = self.checkQuantizeTrace(
                    m,
                    [input, offsets, dense],
                    qconfig=static_qconfig[1],
                    rtol=2e-1,
                    expect_result=y,
                )
                self.assertGraphContainsExactly(
                    graph, "ipex::qmerged_embeddingbag_cat", 1
                )

    def test_interaction_int8(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.f = ipex.nn.functional.interaction

            def forward(self, x1, x2, x3):
                x = self.f(x1.relu(), x2.relu(), x3.relu())
                return x

        m = M()
        inputs = []
        for i in range(0, 3):
            inputs.append(torch.randn([128, 128]) * 0.1)
        graph = self.checkQuantizeTrace(m, inputs, atol=1e-2, qconfig=static_qconfig[1])
        self.assertGraphContainsExactly(graph, "ipex::qinteraction", 1)

    # Besides its primary objective, this UT also implicitly tests if mayRevertDtypeAttributeInsertion
    # in csrc/jit/codegen/onednn/prepare_binary.cpp works well.
    def test_add_int8(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()

            def forward(self, x1, x2):
                out = torch.add(torch.dequantize(x1), torch.dequantize(x2))
                return torch.quantize_per_tensor(out, 0.1, 10, torch.quint8)

        m = M().eval()
        inputs = [
            torch.quantize_per_tensor(torch.randn(12, 12), 0.1, 10, torch.quint8),
            torch.quantize_per_tensor(torch.randn(12, 12), 0.1, 10, torch.quint8),
        ]
        with torch.no_grad():
            traced_model = torch.jit.trace(m, inputs)
            traced_model = torch.jit.freeze(traced_model)
            traced_model(*inputs)
            graph = traced_model.graph_for(*inputs)

            ori_out = m(*inputs)
            out = traced_model(*inputs)

        self.assertEqual(ori_out, out)
        self.assertGraphContainsExactly(graph, "quantized::add", 1)

    # This test case will be enabled after LSTM int8->fp32 works
    def test_lstm(self):
        class M(nn.Module):
            def __init__(
                self,
                input_size,
                hidden_size,
                num_layers,
                bidirectional=False,
                bias=False,
                dropout=0,
                batch_first=False,
            ):
                super(M, self).__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bidirectional=bidirectional,
                    bias=bias,
                    dropout=dropout,
                    batch_first=batch_first,
                )

            def forward(self, x, h=None):
                x, h = self.lstm(x, h)
                return x, h

        def _lstm_params_list():
            params_dict = {
                "input_size": [1],
                "hidden_size": [16],
                "num_layers": [3],
                "bidirectional": [False, True],
                "bias": [False, True],
                "empty_state": [False, True],
                "batch_first": [False, True],
                "dropout": [0, 0.4, 1],
                "batch_size": [1, 2],
                "seq_len": [48],
            }
            params_list = []
            for key, value in params_dict.items():
                params_list.append(value)
            return params_list

        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        params_list = _lstm_params_list()
        for (
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            bias,
            empty_state,
            batch_first,
            dropout,
            batch_size,
            seq_len,
        ) in itertools.product(*params_list):
            # dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1
            if dropout > 0 and num_layers == 1:
                continue
            num_directions = 2 if bidirectional else 1

            if batch_first:
                x = torch.randn(batch_size, seq_len, input_size)
            else:
                x = torch.randn(seq_len, batch_size, input_size)
            m = M(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first,
            )
            graph = self.checkQuantizeTrace(m, [x], atol=3e-2, rtol=1e-1)
            self.assertGraphContainsExactly(graph, "ipex::quantized_lstm", 1)

    def test_lstm_PackedSequence(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.lstm = nn.LSTM(
                    input_size=288,
                    hidden_size=1024,
                    num_layers=6,
                    batch_first=True,
                    bidirectional=True,
                    bias=True,
                    dropout=0.2,
                )

            def forward(self, input, hid, mask=None):
                if mask is not None:
                    lengths = mask.sum(-1)
                    seq = pack_padded_sequence(input, lengths.cpu(), batch_first=True)
                    seq, hid = self.lstm(seq, hid)
                    seq = pad_packed_sequence(seq, batch_first=True)[0]
                    return seq, hid
                else:
                    return self.lstm(input, hid)

        model = M().eval()
        seq = torch.randn(size=(1, 211, 288), dtype=torch.float32)
        # initialize hidden states
        h0 = torch.zeros((12, 1, 1024), dtype=seq.dtype)
        hid = (h0, h0)
        mask = torch.ones(size=(1, 211), dtype=torch.uint8)

        graph = self.checkQuantizeTrace(model, [seq, hid, mask])
        self.assertGraphContainsExactly(graph, "aten::lstm", 1)

    def test_linear_lstm(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.linear = nn.Linear(512, 64)
                self.lstm = nn.LSTM(input_size=64, hidden_size=256, num_layers=2)

            def forward(self, input, hid=None):
                x = self.linear(input)
                x = self.lstm(x, hid)
                return x

        model = M().eval()
        seq = torch.randn(24, 1, 512)
        h0 = torch.zeros((2, 1, 256), dtype=seq.dtype)
        hid = (h0, h0)

        graph = self.checkQuantizeTrace(model, [seq, hid], atol=3e-2, rtol=1e-1)
        self.assertGraphContainsExactly(graph, "ipex::quantized_lstm", 1)
        self.assertGraphContainsExactly(graph, "aten::lstm", 0)

    def test_conv2d_with_padding(self):
        class M(nn.Module):
            def __init__(self, padding_mode):
                super(M, self).__init__()
                self.conv = nn.Conv2d(
                    3, 3, 2, padding=1, bias=True, padding_mode=padding_mode
                )

            def forward(self, x):
                x = self.conv(x)
                return x

        x = torch.rand(1, 3, 14, 14)
        patterns = [
            ["aten::dequantize", "aten::_convolution"],
        ]
        for padding_mode in ["circular", "replicate", "reflect"]:
            m = M(padding_mode=padding_mode).eval()
            graph = self.checkQuantizeTrace(m, [x], atol=2e-1)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
            self.checkPatterns(graph, patterns)


class TestIpexQuantizationConvertAPI(JitLlgaTestCase):
    def test_inplace_preapre(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.linear = nn.Linear(128, 1)

            def forward(self, x):
                x = self.linear(x)
                return x

        x = torch.rand(1, 128)
        for inplace in [False, True]:
            m = M()
            prepared_model = ipex.quantization.prepare(
                m, static_qconfig[0], example_inputs=x, inplace=inplace
            )
            if inplace:
                self.assertEqual(
                    m.linear.weight.data_ptr(), prepared_model.linear.weight.data_ptr()
                )
            else:
                self.assertNotEqual(
                    m.linear.weight.data_ptr(), prepared_model.linear.weight.data_ptr()
                )

    def test_inplace_convert(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.linear = nn.Linear(128, 1)

            def forward(self, x):
                x = self.linear(x)
                return x

        m = M()
        x = torch.rand(1, 128)
        for int8_bf16 in [False]:
            m_ = copy.deepcopy(m)
            for inplace in [False, True]:
                orgin_model_weight_dtype = m_.linear.weight.dtype
                orgin_model_bias_dtype = m_.linear.bias.dtype
                _, _, ori_model = self.prepareModel(
                    m_,
                    x,
                    qconfig=static_qconfig[1],
                    int8_bf16=int8_bf16,
                    prepare_inplace=True,
                    convert_inplace=inplace,
                )
                if inplace and int8_bf16:
                    if (
                        m_.linear.weight.dtype == orgin_model_weight_dtype
                        or m_.linear.bias.dtype == orgin_model_bias_dtype
                    ):
                        print("model should have changed")
                        assert 0
                else:
                    if (
                        m_.linear.weight.dtype != orgin_model_weight_dtype
                        or m_.linear.bias.dtype != orgin_model_bias_dtype
                    ):
                        print("model should not change")
                        assert 0

    def test_qconf_summary_save_load(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = nn.Conv2d(3, 64, 1, 1)
                self.linear = nn.Linear(256, 1)

            def forward(self, x):
                x = self.conv(x)
                x = torch.flatten(x, 1)
                x = self.linear(x)
                y = torch.relu(x)
                x = torch.add(x, y)
                return x

        m = M()
        x = torch.rand(1, 3, 2, 2)
        prepared_model = ipex.quantization.prepare(
            m, static_qconfig[0], example_inputs=x, inplace=False
        )
        prepared_model(x)
        with tempfile.TemporaryDirectory() as tmp:
            # case1: save qconf and load qconf.
            path = os.path.join(tmp, "configure.json")
            prepared_model.save_qconf_summary(path)
            convert_model = ipex.quantization.convert(prepared_model)
            traced_model_ref = torch.jit.trace(convert_model, x).eval()
            traced_model_ref = torch.jit.freeze(traced_model_ref)
            # load the saved qconf
            prepared_model = ipex.quantization.prepare(
                m, static_qconfig[0], example_inputs=x, inplace=False
            )
            prepared_model.load_qconf_summary(path)
            convert_model = ipex.quantization.convert(prepared_model)
            traced_model = torch.jit.trace(convert_model, x).eval()
            traced_model = torch.jit.freeze(traced_model)
            for i in range(2):
                y_before = traced_model_ref(x)
                y_after = traced_model(x)
            self.assertEqual(y_before, y_after)
            # save and load qconf again to make sure we didn't lost something
            path2 = os.path.join(tmp, "configure_new.json")
            prepared_model.save_qconf_summary(path2)
            prepared_model = ipex.quantization.prepare(
                m, static_qconfig[0], example_inputs=x, inplace=False
            )
            prepared_model.load_qconf_summary(path2)
            convert_model = ipex.quantization.convert(prepared_model)
            traced_model = torch.jit.trace(convert_model, x).eval()
            traced_model = torch.jit.freeze(traced_model)
            for i in range(2):
                y_after = traced_model(x)
            self.assertEqual(y_before, y_after)
            # make sure the new saved json is same as old one.
            with open(path, "r") as f:
                old_json = json.load(f)
            with open(path2, "r") as f:
                new_json = json.load(f)
            self.assertTrue(old_json == new_json)

            # case2: load qconf and re-do calibration, make sure the scales/zps is updated.
            x_new = torch.rand(1, 3, 2, 2) * 10
            # do ref quantization
            prepared_model = ipex.quantization.prepare(
                m, static_qconfig[0], example_inputs=x_new, inplace=False
            )
            prepared_model(x_new)
            ref_path = os.path.join(tmp, "configure_ref.json")
            prepared_model.save_qconf_summary(ref_path)
            convert_model = ipex.quantization.convert(prepared_model)
            traced_model_ref = torch.jit.trace(convert_model, x_new).eval()
            traced_model_ref = torch.jit.freeze(traced_model_ref)
            # load qconf, and re-do calibration
            prepared_model = ipex.quantization.prepare(
                m, static_qconfig[0], example_inputs=x_new, inplace=False
            )
            prepared_model.load_qconf_summary(path2)
            prepared_model(x_new)
            new_path = os.path.join(tmp, "configure_new.json")
            prepared_model.save_qconf_summary(new_path)
            traced_model_new = torch.jit.trace(convert_model, x_new).eval()
            traced_model_new = torch.jit.freeze(traced_model_new)
            for i in range(2):
                y_ref = traced_model_ref(x_new)
                y_new = traced_model_new(x_new)
            self.assertEqual(y_ref, y_new)
            # make sure the new saved json is same as ref one.
            with open(ref_path, "r") as f:
                old_json = json.load(f)
            with open(new_path, "r") as f:
                new_json = json.load(f)
            self.assertTrue(old_json == new_json)

    def test_observer_dtype_update(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        m = M()
        x = torch.rand(4, 4)
        prepared_model = ipex.quantization.prepare(
            m, static_qconfig[0], example_inputs=x, inplace=False
        )
        prepared_model(x)
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = os.path.join(tmp, "configure.json")
            prepared_model.save_qconf_summary(ref_path)
            with open(ref_path, "r") as f:
                old_json = json.load(f)
                # change observe's dtype.
                old_json[" "]["q_op_infos"]["0"]["activation_observer"][
                    "dtype"
                ] = "torch.qint8"
                old_json[" "]["q_op_infos"]["0"]["activation_observer"][
                    "quant_min"
                ] = -128
                old_json[" "]["q_op_infos"]["0"]["activation_observer"][
                    "quant_max"
                ] = 127
                new_path = os.path.join(tmp, "configure_new.json")
                with open(new_path, "w") as fp:
                    json.dump(old_json, fp, indent=4)
                prepared_model.load_qconf_summary(new_path)
                prepared_model(x)
                convert_model = ipex.quantization.convert(prepared_model)
                traced_model = torch.jit.trace(convert_model, x).eval()
                traced_model = torch.jit.freeze(traced_model)
                for _ in range(2):
                    y_new = traced_model(x)

                ref_qconfig = QConfig(
                    activation=MinMaxObserver.with_args(
                        qscheme=torch.per_tensor_affine, dtype=torch.qint8
                    ),
                    weight=default_weight_observer,
                )
                prepared_model = ipex.quantization.prepare(
                    m, ref_qconfig, example_inputs=x, inplace=False
                )
                prepared_model(x)
                convert_model = ipex.quantization.convert(prepared_model)
                traced_model = torch.jit.trace(convert_model, x).eval()
                traced_model = torch.jit.freeze(traced_model)
                for _ in range(2):
                    y_ref = traced_model(x)
                self.assertEqual(y_ref, y_new)

    def test_subclass_format(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.linear = nn.Linear(128, 1)

            def forward(self, x):
                x = self.linear(x)
                return x.sum()

        x = torch.rand(1, 128)
        prepared_model = ipex.quantization.prepare(
            M().eval(), static_qconfig[0], example_inputs=x
        )
        out = prepared_model(x)
        print(out.__format__(".4f"))
        converted_model = ipex.quantization.convert(prepared_model)
        out = converted_model(x)
        print(out.__format__(".4f"))


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

    @llga_fp32_bf16_test_env
    def test_special_mapped_op(self):
        class M1(nn.Module):
            def __init__(self):
                super(M1, self).__init__()

            def forward(self, x, y):
                z = x + 1
                z.zero_()
                y.fill_(3)
                return z, y

        m = M1()
        x = torch.tensor([2, 2])
        y = torch.tensor([2, 4])
        graph, traced_model, _ = self.prepareModel(m, [x, y])
        FileCheck().check_not("aten::zero_").check_not("aten::fill_").run(graph)
        self.assertEqual(traced_model(x, y), m(x, y))

        class M2(nn.Module):
            def __init__(self):
                super(M2, self).__init__()

            def forward(self, x):
                return x.normal_()

        m = M2()
        x = torch.rand(2, 1, 3, 4)
        graph, traced_model, _ = self.prepareModel(m, [x])
        FileCheck().check_not("normal_").run(graph)
        with freeze_rng_state():
            out1 = m(x)
        with freeze_rng_state():
            out2 = traced_model(x)
        self.assertEqual(out1, out2)

        class M3(nn.Module):
            def __init__(self):
                super(M3, self).__init__()

            def forward(self, x):
                x.fill_(3)
                x.zero_()
                return x

        m = M3()
        x = torch.tensor([2, 2])
        graph, traced_model, _ = self.prepareModel(m, [x])
        FileCheck().check_not("aten::zero_").check_not("aten::fill_").run(graph)
        self.assertEqual(traced_model(x), m(x))


class TestDynamicQuantization(JitLlgaTestCase):
    def test_linear_dynamic(self):
        class SubModule(nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.linear = nn.Linear(3, 3)

            def forward(self, x):
                x = self.linear(x)
                return x

        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.linear1 = nn.Sequential(nn.Linear(3, 3))
                self.linear2 = SubModule()
                self.linear3 = nn.Linear(3, 3)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                return x

        m = M().eval()
        x = torch.randn(3, 3)
        for qconfig in dynamic_qconfig:
            graph = self.checkQuantizeTrace(m, [x], atol=2e-1, qconfig=qconfig)
            FileCheck().check_not("aten:linear").check("quantized::linear_dynamic").run(
                graph
            )

    def test_linear_dynamic_bf16(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.linear = nn.Linear(3, 3)

            def forward(self, x):
                x = self.linear(x)
                return x

        x = torch.randn(3, 3)
        m = M().eval()
        graph, _, _ = self.prepareModel(
            m, [x], qconfig=dynamic_qconfig[0], int8_bf16=True
        )
        FileCheck().check_not("aten:linear").check("quantized::linear_dynamic").run(
            graph
        )

    def test_lstm_dynamic(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.lstm = torch.nn.LSTM(10, 20, 2)

            def forward(self, x, hx, cx):
                x, h_xs = self.lstm(x, (hx, cx))
                return x, h_xs

        m = M().eval()
        x = torch.randn(5, 3, 10)
        h = torch.randn(2, 3, 20)
        c = torch.randn(2, 3, 20)
        for qconfig in dynamic_qconfig:
            graph = self.checkQuantizeTrace(m, [x, h, c], atol=2e-1, qconfig=qconfig)
            FileCheck().check_not("aten:lstm").check("aten::quantized_lstm").run(graph)


class TestDictInput(JitLlgaTestCase):
    def test_only_dict_input(self):
        class SubModule(nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.linear = nn.Linear(3, 3)

            def forward(self, x):
                x = self.linear(x)
                return x

        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.linear1 = nn.Sequential(nn.Linear(3, 3))
                self.linear2 = SubModule()
                self.linear3 = nn.Linear(3, 3)

            def forward(self, x1, x2, x3):
                x1 = self.linear1(x1)
                x2 = self.linear2(x2)
                x3 = self.linear3(x3)
                return x1 + x2 + x3

        int8_bf16_list = [True, False]
        for qconfig, int8_bf16 in itertools.product(static_qconfig, int8_bf16_list):
            # Step1: Test model with tuple(x1, x2, x3) input.
            m = M().eval()
            m2 = copy.deepcopy(m).eval()
            x1 = torch.randn(3, 3)
            x2 = torch.randn(3, 3)
            x3 = torch.randn(3, 3)
            graph = self.checkQuantizeTrace(
                m, [x1, x2, x3], atol=2e-1, qconfig=qconfig, int8_bf16=int8_bf16
            )
            FileCheck().check("aten::linear").run(graph)
            patterns = [
                [
                    "aten::dequantize",
                    "aten::linear",
                ],
                [
                    "aten::dequantize",
                    "aten::linear",
                ],
                [
                    "aten::dequantize",
                    "aten::linear",
                    "aten::add",
                    "aten::add",
                ],
            ]
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)
            self.checkPatterns(graph, patterns)

            # Step2: Test model with Dict{"x1": x1, "x2": x2, "x3": x3} input.
            graph = self.checkQuantizeTrace(
                m2,
                atol=2e-1,
                qconfig=qconfig,
                int8_bf16=int8_bf16,
                x_kwarg={"x1": x1, "x2": x2, "x3": x3},
            )
            FileCheck().check("aten::linear").run(graph)
            patterns = [
                [
                    "aten::dequantize",
                    "aten::linear",
                ],
                [
                    "aten::dequantize",
                    "aten::linear",
                ],
                [
                    "aten::dequantize",
                    "aten::linear",
                    "aten::add",
                    "aten::add",
                ],
            ]
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)
            self.checkPatterns(graph, patterns)


if __name__ == "__main__":
    run_tests()
