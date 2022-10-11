import torch
import torch.fx.experimental.optimization as optimization
import intel_extension_for_pytorch as ipex
import intel_extension_for_pytorch._C as core
from intel_extension_for_pytorch.nn.utils._weight_prepack import _IPEXLinear as _IPEXLinear, _IPEXConv2d as _IPEXConv2d
from torch.testing._internal.common_utils import TestCase
from torch.optim import Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, RMSprop, Rprop, SGD
import unittest
import itertools
import copy
from common_utils import TestModule
from intel_extension_for_pytorch.optim._lamb import Lamb
import os

class ConvBatchNorm(torch.nn.Module):
    def __init__(self,):
        super(ConvBatchNorm, self).__init__()
        self.input1 = torch.randn(1, 3, 224, 224)
        self.conv = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.bn = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        return self.bn(self.conv(x))

class TwoLayerMLP(torch.nn.Module):
    def __init__(self):
        super(TwoLayerMLP, self).__init__()
        self.input1 = torch.randn(2, 2)
        self.input2 = torch.randn(3, 3)
        self.l1 = torch.nn.Linear(2, 2)
        self.l2 = torch.nn.Linear(3, 3)

    def forward(self, x1, x2):
        return self.l1(x1).sum() + self.l2(x2).sum()

class OneLayerMLP(torch.nn.Module):
    def __init__(self):
        super(OneLayerMLP, self).__init__()
        self.input1 = torch.randn(2, 2)
        self.l1 = torch.nn.Linear(2, 2)

    def forward(self, x1):
        return self.l1(x1)

class ConvTranspose2d(torch.nn.Module):
    def __init__(self, ):
        super(ConvTranspose2d, self).__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(5, 5, (3 ,3))
        self.input1 = torch.randn(5, 5, 3, 3)

    def forward(self, x):
        x = self.conv_transpose2d(x)
        return x

class TestOptimizeCases(TestCase):
    def test_optimize_parameters_behavior(self):
        model = ConvBatchNorm().eval()
        pre_te_enable_status = torch._C._jit_texpr_fuser_enabled()
        torch._C._jit_set_texpr_fuser_enabled(False)
        for level in ["O0", "O1"]:
            # disbale conv_bn folding
            opt_M = ipex.optimize(model, level=level, dtype=torch.float, conv_bn_folding=False)
            with torch.no_grad():
                x = model.input1
                traced_model = torch.jit.trace(opt_M, x)
                trace_graph = traced_model.graph_for(x)
            self.assertTrue(any(n.kind() == "ipex::batch_norm" for n in trace_graph.nodes()))
            # TODO check weight_prepack.
        torch._C._jit_set_texpr_fuser_enabled(pre_te_enable_status)

    def test_optimize_bf16_model(self):
        model = ConvBatchNorm()
        optimized_model = ipex.optimize(model.eval(), dtype=torch.bfloat16)
        # model should not has master weight attr for infernence model.
        self.assertTrue(not hasattr(optimized_model.conv, 'master_weight'))
        # model should has master weight attr for infernence model.
        sgd = torch.optim.SGD(model.parameters(), lr=0.1)
        optimized_model, optimized_sgd = ipex.optimize(model.train(), optimizer=sgd, dtype=torch.bfloat16, split_master_weight_for_bf16=False)
        self.assertTrue(hasattr(optimized_model.conv, 'master_weight'))

    def test_optimize_pretrain_model(self):
        optimizer_options = [Lamb, Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, RMSprop, Rprop, SGD]

        options = itertools.product([torch.float, torch.bfloat16], optimizer_options)
        for dtype, optimizer in options:
            model = ConvBatchNorm().to(memory_format=torch.channels_last).train()
            model.conv.weight.requires_grad_(False)
            model.conv.bias.requires_grad_(False)
            origin_model = copy.deepcopy(model)
            lr = 1e-4 if optimizer is SGD else 1e-2
            origin_optimizer = optimizer(origin_model.parameters(), lr=lr)
            ipex_model, ipex_optimizer = ipex.optimize(origin_model, optimizer=origin_optimizer, dtype=dtype)
            for origi_p, opti_p in zip(origin_model.parameters(), ipex_model.parameters()):
                self.assertEqual(origi_p.requires_grad, opti_p.requires_grad)

            x = model.input1.to(memory_format=torch.channels_last)
            origin_x = x.clone()
            ipex_x = x.clone()
            with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                y1 = origin_model(origin_x)
                grad_y = torch.ones_like(y1)
                origin_optimizer.zero_grad()
                y1.backward(grad_y)
                origin_optimizer.step()
                # train one step for ipex.
                y2 = ipex_model(ipex_x)
                ipex_optimizer.zero_grad()
                y2.backward(grad_y)
                ipex_optimizer.step()
                self.assertEqual(y1, y2, rtol=1e-4, atol=5e-02)
                origin_model_state = origin_model.state_dict()
                ipex_model_state = ipex_model.state_dict()
                for var_name in origin_model_state:
                    self.assertEqual(origin_model_state[var_name], ipex_model_state[var_name], rtol=1e-4, atol=5e-02)
                self.assertTrue(origin_model.conv.weight.grad==None)
                self.assertTrue(ipex_model.conv.weight.grad==None)

    def test_optimize_unsupport_dtype_conversion(self):
        class Conv(torch.nn.Module):
            def __init__(self,):
                super(Conv, self).__init__()
                self.conv = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

            def forward(self, x):
                return self.conv(x)

        model = Conv().double()
        with self.assertWarnsRegex(UserWarning,
                                   "WARNING: Can't convert model's parameters dtype"):
            optimized_model = ipex.optimize(model.eval(), dtype=torch.bfloat16)

    def test_optimize_bf16_upsupported(self):
        class Conv(torch.nn.Module):
            def __init__(self,):
                super(Conv, self).__init__()
                self.conv = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        def forward(self, x):
            return self.conv(x)

        model = Conv()
        if not core.onednn_has_bf16_support():
            msg = r"BF16 weight prepack needs the cpu support avx512bw, avx512vl and avx512dq, please set dtype to torch.float or set weights_prepack to False."
            with self.assertRaisesRegex(AssertionError, msg):
                optimized_model = ipex.optimize(model.eval(), dtype=torch.bfloat16)

    def test_optimize_unsupport_freeze_optimization(self):
        model = ConvBatchNorm().eval()
        x = model.input1
        with torch.no_grad():
            traced_model = torch.jit.trace(model, x)
            frozen_model = torch.jit.freeze(traced_model)
        optimized_model = ipex.optimize(frozen_model)
        self.assertTrue(frozen_model == optimized_model)

    def test_optimize_inplace_behavior_eval_mode(self):
        M_ori = TestModule()
        options = itertools.product([torch.float32, torch.bfloat16], ["O0", "O1"])
        for dtype, level in options:
            # non-inplace
            M = copy.deepcopy(M_ori).eval()
            opt_M = ipex.optimize(M, dtype=dtype, level=level, inplace=False)
            self.assertTrue(M.linear.weight.data_ptr() != opt_M.linear.weight.data_ptr())
            self.assertTrue(M.conv.weight.data_ptr() != opt_M.conv.weight.data_ptr())
            self.assertTrue(M.embeddingbag.weight.data_ptr() != opt_M.embeddingbag.weight.data_ptr())

            # inplace
            M = copy.deepcopy(M_ori).eval()
            opt_M = ipex.optimize(M, dtype=dtype, level=level, inplace=True)
            # After ConvBN folding,  opt_M will be Graph Module while the M is original nn.Module which they
            # share parameters. But the changes on Graph Module cannot be reflected on original module. So
            # only the un-opitimized  weight will use same mem buffer with original module.
            # While dtype = float, ipex.optimize will choose mkl backend and does not prepack weight
            if level == "O1":
                self.assertTrue(M.conv.weight.data_ptr() != opt_M.conv.weight.data_ptr())
                self.assertTrue(dtype is torch.float or M.linear.weight.data_ptr() != opt_M.linear.weight.data_ptr())
            # un-optimized part should be inplaced
            self.assertTrue(M.embeddingbag.weight.data_ptr() == opt_M.embeddingbag.weight.data_ptr())

    def test_optimize_inplace_behavior_training_mode_with_optimizer(self):
        M_ori = TestModule()
        options = itertools.product([torch.float32, torch.bfloat16], ["O0", "O1"])
        for dtype, level in options:
            # non-inplace
            M = copy.deepcopy(M_ori).train()
            sgd = torch.optim.SGD(M.parameters(), lr=0.1)
            opt_M, _ = ipex.optimize(M, dtype=dtype, optimizer=sgd, level=level, inplace=False)
            self.assertTrue(M.linear.weight.data_ptr() != opt_M.linear.weight.data_ptr())
            self.assertTrue(M.conv.weight.data_ptr() != opt_M.conv.weight.data_ptr())
            self.assertTrue(M.embeddingbag.weight.data_ptr() != opt_M.embeddingbag.weight.data_ptr())
            if level == "O1":
                self.assertEqual(M.linear.weight.dtype, torch.float)
                self.assertEqual(M.conv.weight.dtype, torch.float)
                self.assertEqual(M.embeddingbag.weight.dtype, torch.float)
                self.assertEqual(M.bn.weight.dtype, torch.float)
                self.assertEqual(opt_M.linear.weight.dtype, dtype)
                self.assertEqual(opt_M.conv.weight.dtype, dtype)
                self.assertEqual(opt_M.embeddingbag.weight.dtype, dtype)
                self.assertEqual(opt_M.bn.weight.dtype, torch.float)

            # inplace
            M = copy.deepcopy(M_ori).train()
            sgd = torch.optim.SGD(M.parameters(), lr=0.1)
            opt_M, _ = ipex.optimize(M, dtype=dtype, optimizer=sgd, level=level, inplace=True)
            self.assertTrue(M.linear.weight.data_ptr() == opt_M.linear.weight.data_ptr())
            self.assertTrue(M.conv.weight.data_ptr() == opt_M.conv.weight.data_ptr())
            self.assertTrue(M.embeddingbag.weight.data_ptr() == opt_M.embeddingbag.weight.data_ptr())
            if level == "O1":
                self.assertEqual(M.linear.weight.dtype, dtype)
                self.assertEqual(M.conv.weight.dtype, dtype)
                self.assertEqual(M.embeddingbag.weight.dtype, dtype)
                self.assertEqual(M.bn.weight.dtype, torch.float)

    def _test_tensor_convert(self, tensor, bf16_tensor):
        top_half, bot_half = torch.ops.torch_ipex.split_float_bfloat16(tensor)
        # truncated top half should equal with convert fp32 to bf16 by ".bfloat()"
        self.assertEqual(bf16_tensor, top_half)
        # recovery float tensor with top half and bottom half
        float_tensor = torch.ops.torch_ipex.cat_bfloat16_float(top_half, bot_half)
        self.assertEqual(tensor, float_tensor)
        self.assertEqual(tensor.stride(), top_half.stride())
        self.assertEqual(tensor.stride(), float_tensor.stride())

    def test_tensor_convert(self):
        # contiguous case
        tensor = torch.rand(100, 100)
        self._test_tensor_convert(tensor, tensor.bfloat16())
        # transposed case
        self._test_tensor_convert(tensor.t(), tensor.bfloat16().t())
        # sliced-out case
        self._test_tensor_convert(tensor[2:5, 2:5], tensor.bfloat16()[2:5, 2:5])
        # nc11 channel-last case
        tensor = torch.rand(128, 256, 1, 1).to(memory_format=torch.channels_last)
        self._test_tensor_convert(tensor, tensor.bfloat16())

    def test_module_conversion(self):
        M_ori = TestModule()
        options = itertools.product([torch.bfloat16, torch.float32], ["O0", "O1"], [True, False])
        for dtype, level, auto_kernel_selection in options:
            sgd = torch.optim.SGD(M_ori.parameters(), lr=0.1)
            opt_M, _ = ipex.optimize(M_ori, dtype=dtype, optimizer=sgd, level=level, auto_kernel_selection=auto_kernel_selection)
            if level == "O0":
                self.assertTrue(isinstance(opt_M.linear, torch.nn.Linear))
                self.assertTrue(isinstance(opt_M.conv, torch.nn.Conv2d))
            else:
                self.assertTrue(isinstance(opt_M.linear, _IPEXLinear))
                self.assertTrue(isinstance(opt_M.conv, _IPEXConv2d))

    def test_record_shape(self):
        options = itertools.product([OneLayerMLP, TwoLayerMLP], [True, False])
        for module, inference_only in options:
            M = module()
            input = M.input1
            if isinstance(M, TwoLayerMLP):
                input = (M.input1, M.input2)
            if inference_only:
                M.eval()
                opt_M = ipex.optimize(M, sample_input=input, auto_kernel_selection=True)
            else:
                optimizer = torch.optim.SGD(M.parameters(), lr=0.01)
                opt_M, _ = ipex.optimize(M, optimizer=optimizer, sample_input=input, auto_kernel_selection=True)
            self.assertEqual(opt_M.l1.batch_size_collapsed, 2)
            if isinstance(M, TwoLayerMLP):
                self.assertEqual(opt_M.l2.batch_size_collapsed, 3)

    def test_traced_model_serialization(self):
        for module in [ConvBatchNorm, OneLayerMLP, ConvTranspose2d]:
            for dtype in [torch.float, torch.bfloat16]:
                M = module().eval()
                input = M.input1.to(dtype)
                opt_M = ipex.optimize(M, dtype=dtype, auto_kernel_selection=True)
                with torch.no_grad():
                    traced_M = torch.jit.trace(opt_M, input).eval()
                    traced_M.save('traced_m.pt')
                    loaded_M = torch.jit.load('traced_m.pt')
                    self.assertEqual(traced_M(input), loaded_M(input))
                    os.remove('traced_m.pt')

    def test_optimized_model_with_fx(self):
        for module in [ConvBatchNorm, OneLayerMLP, ConvTranspose2d]:
            for dtype in [torch.float, torch.bfloat16]:
                M = module().eval()
                input = M.input1.to(dtype)
                opt_M = ipex.optimize(M, dtype=dtype, auto_kernel_selection=True)
                ref_out = opt_M(input)
                fx_M = optimization.fuse(opt_M)
                fx_out = fx_M(input)
                self.assertEqual(ref_out, fx_out)
                with torch.no_grad():
                    traced_M = torch.jit.trace(fx_M, input).eval()
                    traced_M = torch.jit.freeze(traced_M)
                    # do graph opt
                    traced_M(input)
                    # get optimized results
                    out = traced_M(input)
                    self.assertEqual(ref_out, out)

    def test_optimized_model_with_sample_input(self):
        for module in [ConvBatchNorm, OneLayerMLP, ConvTranspose2d]:
            model = module().train()
            input = model.input1
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            origin_model_state = copy.deepcopy(model.state_dict())
            ipex_model, _ = ipex.optimize(model, dtype=torch.float32, inplace=False, optimizer=optimizer, sample_input=input)
            ipex_model_state = ipex_model.state_dict()
            for var_name in origin_model_state:
                self.assertEqual(origin_model_state[var_name], ipex_model_state[var_name])


if __name__ == '__main__':
    test = unittest.main()
