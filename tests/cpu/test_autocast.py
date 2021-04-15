import unittest, copy
import torch
import torch.nn as nn
import intel_pytorch_extension as ipex
from common_utils import TestCase
import time, sys

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

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = torch.nn.Conv2d(3, 16, (3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TestSimpleNet(TestCase):
    def test_generate_jit_trace_model(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)

        model = SimpleNet() 
        model.eval()
        #ipex.core.disable_jit_opt()
        x = torch.rand((1, 3, 224, 224))
        with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)), torch.no_grad():
            traced_model = torch.jit.trace(model, x)
        with torch.no_grad():
            y = traced_model(x)
            #print(traced_model.graph_for(x))
        #ipex.core.enable_jit_opt()
        self.assertEqual(y.dtype, torch.float) #conv whitelist, bn blacklist, relu fallthrough

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

if __name__ == '__main__':
    test = unittest.main()