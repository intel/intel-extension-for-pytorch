import unittest, copy
import torch
import torch.nn as nn
import intel_pytorch_extension as ipex
from common_utils import TestCase
import time, sys
from torch.testing._core import _get_default_tolerance
import itertools

def get_rand_seed():
    return int(time.time() * 1000000000)

class TestFunction(TestCase):
    def test_forward_dtype(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        _in_cpu = torch.rand((1, 1, 7, 7))
        _conv = torch.nn.Conv2d(1, 1, (3, 3))
        with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)):
            out_autocast = _conv(_in_cpu)
        self.assertEqual(out_autocast.dtype, torch.bfloat16)

    def test_nested_useage(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        _in_cpu = torch.rand((1, 1, 7, 7))
        _conv = torch.nn.Conv2d(1, 1, (3, 3))
        with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)):
            with ipex.amp.autocast(enabled=False):
                out_autocast = _conv(_in_cpu)
            self.assertEqual(out_autocast.dtype, torch.float)

            with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.float)):
                out_autocast = _conv(_in_cpu)
            self.assertEqual(out_autocast.dtype, torch.float)

class TestConv(TestCase):
    # In autocast, Conv should be forced into the bf16
    def test_conv2d_forward(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        _in_cpu = torch.rand((1, 1, 7, 7))
        _conv = torch.nn.Conv2d(1, 1, (3, 3))
        
        out = _conv(_in_cpu)
        with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)), torch.no_grad():
            out_autocast = _conv(_in_cpu)
        self.assertEqual(out_autocast.dtype, torch.bfloat16)
        self.assertEqual(out, out_autocast.to(torch.float), 1e-2)

    def test_conv2d_backward(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)

        input = torch.rand((1, 1, 7, 7))
        _in_cpu = input.clone().requires_grad_()
        in_autocast = input.clone().requires_grad_()
        
        _conv = torch.nn.Conv2d(1, 1, (3, 3))
        conv_cpu = copy.deepcopy(_conv)
        conv_auto_cast = copy.deepcopy(_conv)
        
        out = conv_cpu(_in_cpu).sum()
        out.backward()
        with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)):
            out_autocast = conv_auto_cast(in_autocast).sum()
        out_autocast.backward()
            #loss = criterion(y_autocast, target)
        self.assertEqual(out_autocast.dtype, torch.bfloat16)
        self.assertEqual(in_autocast.grad.dtype, torch.float)
        self.assertEqual(_in_cpu.grad, in_autocast.grad, 1e-2)

class TestAutocastWithJit(TestCase):
    def setUp(self):
        super(TestAutocastWithJit, self).setUp()
        from test_jit import Conv_Bn_Relu, BatchNorm_Conv_BatchNorm, ConvBatchNorm_Fixed, ConvReshapeBatchNorm,\
                            CascadedConvBnSumRelu, LinearBn, Linear_Reshape_Bn
        self.models = [Conv_Bn_Relu(2, 3, 32, kernel_size=3, stride=1), BatchNorm_Conv_BatchNorm(2, 3, 32, kernel_size=3, stride=1),\
                    ConvBatchNorm_Fixed(2, 3, 32, kernel_size=3, stride=1), ConvBatchNorm_Fixed(3, 3, 32, kernel_size=3, stride=1),\
                    ConvReshapeBatchNorm(2, 3, 32, (64, 16, 62, 62), kernel_size=3, stride=1),\
                    CascadedConvBnSumRelu(2, 3, 64, 32, kernel_size=3, stride=1),\
                    LinearBn(2 ,32, 32, bias=True),\
                    Linear_Reshape_Bn(2 ,32, 32,(1,1,64,16),bias=True)]
        self.inputs = [torch.randn(32, 3, 64, 64), torch.randn(32, 3, 64, 64),\
                    torch.randn(32, 3, 64, 64), torch.randn(32, 3, 32, 32, 32),\
                    torch.randn(32, 3, 64, 64),\
                    torch.rand(32, 3, 64, 64),\
                    torch.rand(1, 1, 32, 32),\
                    torch.rand(1, 1, 32, 32)]

    def test_generate_autocast_jit_trace_model(self):
        def test_generate_autocast_jit_trace_model(model, x):
            model.eval()
            ipex.core.disable_jit_opt()
            with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)), torch.no_grad():
                traced_model = torch.jit.trace(model, x)
            ipex.core.enable_jit_opt()
            with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)), torch.no_grad():
                traced_model2 = torch.jit.trace(model, x.clone())
        for i in range(self.models.__len__()):
            test_generate_autocast_jit_trace_model(self.models[i], self.inputs[i])

    def test_nchw_autocast_jit_trace_model(self):
        def test_nchw_autocast_jit_trace_model(model, x):
            model.eval()
            ipex.core.disable_jit_opt()
            with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)), torch.no_grad():
                traced_model = torch.jit.trace(model, x)
            with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)), torch.no_grad():
                y = traced_model(x.clone())
                y2 = model(x.clone())
            ipex.core.enable_jit_opt()
            torch.testing.assert_allclose(y.double(), y2.double(), rtol=1e-05, atol=_get_default_tolerance(y, y2)[1])
        for i in range(self.models.__len__()):
            test_nchw_autocast_jit_trace_model(self.models[i], self.inputs[i])

    def test_nhwc_autocast_jit_trace_model(self):
        def test_nhwc_autocast_jit_trace_model(model, x):
            model.eval()
            ipex.core.disable_jit_opt()
            with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)), torch.no_grad():
                traced_model = torch.jit.trace(model, x.to(memory_format=torch.channels_last))
            with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)), torch.no_grad():
                y = traced_model(x.clone().to(memory_format=torch.channels_last))
                y2 = model(x.clone().to(memory_format=torch.channels_last))
            ipex.core.enable_jit_opt()
            torch.testing.assert_allclose(y.double(), y2.double(), rtol=1e-05, atol=_get_default_tolerance(y, y2)[1])
        for i in range(self.models.__len__()):
            if self.inputs[i].size().__len__() == 5:
                # NHWC 3D case not support yet
                continue
            test_nhwc_autocast_jit_trace_model(self.models[i], self.inputs[i])

class TestCustomerOps(TestCase):
    def test_interaction_op(self):
        def interact_fusion(x, ly):
            A = [x] + ly
            R = ipex.interaction(*A)
            return R
        
        def interact_fusion_autocast(x, ly):
            A = [x] + ly
            with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)):
                R = ipex.interaction(*A)
            return R

        dtypes=[torch.float32]
        for dtype in dtypes:
            x1 = torch.randn([2048, 128]).to(dtype).clone().detach().requires_grad_()
            x2 = x1.clone().bfloat16().detach().requires_grad_()
            ly1 = []
            ly2 = []
            for i in range(0, 26):
                V = torch.randn([2048, 128]).to(dtype).clone().detach().requires_grad_()
                ly1.append(V)
                ly2.append(V.clone().bfloat16().detach().requires_grad_())

            A = interact_fusion(x1, ly1) # all fp32
            B = interact_fusion_autocast(x2, ly2) # all bf16
            C = interact_fusion_autocast(x1, ly2) #  fp32 dense bf16 emb 
            D = interact_fusion_autocast(x2, ly1) #  bf16 dense fp32 emb

            self.assertEqual(A.dtype, torch.float)
            self.assertEqual(B.dtype, torch.bfloat16)
            #promote to fp32
            self.assertEqual(C.dtype, torch.float)
            self.assertEqual(D.dtype, torch.float)

            self.assertTrue(torch.allclose(A, B.float(), rtol=0.05, atol=0.1))
            self.assertTrue(torch.allclose(A, C.float(), rtol=0.05, atol=0.1))
            self.assertTrue(torch.allclose(A, D.float(), rtol=0.05, atol=0.1))

            A.mean().backward()
            B.mean().backward()
            
            self.assertEqual(x1.grad.dtype, torch.float)
            self.assertEqual(x2.grad.dtype, torch.bfloat16)

            self.assertEqual(x1.grad, x2.grad, 1e-03)
            for i in range(0, 26):
                self.assertEqual(ly1[i].grad.dtype, torch.float)
                self.assertEqual(ly2[i].grad.dtype, torch.bfloat16)
                self.assertEqual(ly1[i].grad, ly2[i].grad, 1e-03)

    def test_embeddingbag_op(self):
        cpu_emb = nn.EmbeddingBag(10, 3, mode='sum', sparse=True)
        autocast_emb = copy.deepcopy(cpu_emb)

        input = torch.LongTensor([1,2,4,5,4,3,2,9])
        # bf16_input = input.clone().detach()

        offsets = torch.LongTensor([0,1,2,3,4,5,6,7])
        # bf16_offsets = offsets.clone().detach()

        cpu_out = cpu_emb(input, offsets)
        with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)), torch.no_grad():
            inference_out = autocast_emb(input, offsets)

        self.assertEqual(cpu_out.dtype, torch.float)
        self.assertEqual(inference_out.dtype, torch.bfloat16)
        self.assertEqual(cpu_out, inference_out.float(), 1e-2)

        # re-init autocast_emb
        autocast_emb = copy.deepcopy(cpu_emb)
        with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)):
            traininig_out = autocast_emb(input, offsets)

        # do not cast weight to bf16 while not inference only
        self.assertEqual(traininig_out.dtype, torch.float)
        self.assertEqual(cpu_out, traininig_out)

class M(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, bias, dropout, batch_first):
        super(M, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, bias=bias, dropout=dropout, batch_first=batch_first)

    def forward(self, x, h = None):
        x, h = self.lstm(x, h)
        return x, h
class TestLSTM(TestCase):
    def _lstm_params_list(self):
        params_dict = {
            "input_size": [1, 2],
            "hidden_size": [5],
            "num_layers": [1, 3],
            "bidirectional": [False, True],
            "bias": [False, True],
            "empty_state": [False, True],
            "batch_first": [False, True],
            "dropout": [0, 1],
            "batch_size": [1, 2],
            "seq_len": [1, 3]
        }

        params_list = []
        for key, value in params_dict.items():
            params_list.append(value)
        return params_list

    def _cast_dtype(self, input, bf16):
        if bf16:
            input = input.to(torch.bfloat16)
        return input

    def _test_lstm(self, training, bf16, prec = 1e-5):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        with torch.set_grad_enabled(training):
            params_list = self._lstm_params_list()
            for input_size, hidden_size, num_layers, bidirectional, bias, empty_state, batch_first, dropout, batch_size, seq_len in itertools.product(*params_list):
                # dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1
                if dropout > 0 and num_layers == 1:
                    continue

                num_directions = 2 if bidirectional else 1
                
                if batch_first:
                    input = torch.randn(batch_size, seq_len, input_size)
                else:
                    input = torch.randn(seq_len, batch_size, input_size)
                h = torch.randn(num_layers * num_directions, batch_size, hidden_size)
                c = torch.randn(num_layers * num_directions, batch_size, hidden_size)

                input_ipex = copy.deepcopy(input)
                h_ipex = copy.deepcopy(h)
                c_ipex = copy.deepcopy(c)

                model = M(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, bias=bias, dropout=dropout, batch_first=batch_first)
                model.train() if training else model.eval()

                model_ipex = copy.deepcopy(model)
                model_ipex.train() if training else model_ipex.eval()
                ipex.utils._replace_lstm_with_ipex_lstm(model_ipex)

                with ipex.amp.autocast(enabled=bf16, configure=ipex.conf.AmpConf(torch.bfloat16)):
                    if empty_state:
                        y, hy = model(self._cast_dtype(input, bf16))
                        y_ipex, hy_ipex = model_ipex(input)
                    else:
                        y, hy = model(input, (self._cast_dtype(h, bf16), self._cast_dtype(c, bf16)))
                        y_ipex, hy_ipex = model_ipex(input, (h, c))

                if not training and bf16:
                    self.assertEqual(input_ipex.dtype, torch.float)
                    self.assertEqual(h_ipex.dtype, torch.float)
                    self.assertEqual(c_ipex.dtype, torch.float)

                    # with mkldnn LSTM, y, hy[0] is bf16 and hy[1] is fp32
                    self.assertEqual(y_ipex.dtype, torch.bfloat16)
                    self.assertEqual(hy_ipex[0].dtype, torch.bfloat16)
                    self.assertEqual(hy_ipex[1].dtype, torch.float)
                self.assertEqual(y, y_ipex, prec=prec)
                self.assertEqual(hy[0], hy_ipex[0], prec=prec)

                self.assertEqual(hy[1], self._cast_dtype(hy_ipex[1], bf16), prec=prec)

    def _test_lstm_pack_padded_sequence(self):
        embedding_dim = 1024
        hidden_dim = 10
        batch_size = 24
        num_layers = 1
        bidirectional = True
        num_direc = 2 if bidirectional else 1
        max_lens = 96

        sent = torch.randn(batch_size, max_lens, embedding_dim)
        hid_0 = torch.rand(num_layers * num_direc, batch_size, hidden_dim)
        hid_1 = torch.randn(num_layers * num_direc, batch_size, hidden_dim)

        sentences = sent.clone().requires_grad_(False)
        sent_lens = torch.Tensor([1, 2, 3, 4, 5, 1, 3, 2, 96, 5, 3, 1, 1, 2, 1, 2, 3, 6, \
        1, 2, 4, 6, 2, 1])

        assert sent_lens.shape[0] == batch_size
        assert sent_lens.max().item() == max_lens

        hidden_0 = hid_0.clone().requires_grad_(False)
        hidden_1 = hid_1.clone().requires_grad_(False)
        embeds = torch.nn.utils.rnn.pack_padded_sequence(sentences, sent_lens, batch_first=True, enforce_sorted=False)
        
        model = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        
        model_ipex = copy.deepcopy(model)
        ipex.utils._replace_lstm_with_ipex_lstm(model_ipex)

        lstm_out, hidden_out = model(embeds, (hidden_0, hidden_1))
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        lstm_out_ipex, hidden_out_ipex = model_ipex(embeds, (hidden_0, hidden_1))
        lstm_out_ipex, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out_ipex, batch_first=True)

        self.assertEqual(lstm_out, lstm_out_ipex)
        self.assertEqual(hidden_out[0], hidden_out_ipex[0])
        self.assertEqual(hidden_out[1], hidden_out_ipex[1])

    def test_lstm_inference(self):
        self._test_lstm(training=False, bf16=False)

        self._test_lstm(training=False, bf16=True, prec=2e-2)
        
        self._test_lstm(training=True, bf16=False)

        # TODO: autocast does not support LSTM bf16 training
        # self._test_lstm(training=True, bf16=True)
    
    def test_lstm_pack_padded_sequence(self):
        self._test_lstm_pack_padded_sequence()

if __name__ == '__main__':
    test = unittest.main()