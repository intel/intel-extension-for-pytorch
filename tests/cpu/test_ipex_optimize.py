import torch
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.nn.utils._weight_prepack import _IPEXLinear as _IPEXLinear, _IPEXConv2d as _IPEXConv2d
from torch.testing._internal.common_utils import TestCase
import unittest
import itertools
import copy
from common_utils import TestModule

class ConvBatchNorm(torch.nn.Module):
    def __init__(self,):
        super(ConvBatchNorm, self).__init__()
        self.conv = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        return self.bn(self.conv(x))

class TestOptimizeCases(TestCase):
    def test_optimize_parameters_behavior(self):
        model = ConvBatchNorm().eval()
        for level in ["O0", "O1"]:
            # disbale conv_bn folding
            opt_M = ipex.optimize(model, level=level, dtype=torch.float, conv_bn_folding=False)
            with torch.no_grad():
                x = torch.randn(1, 3, 224, 224)
                traced_model = torch.jit.trace(opt_M, x)
                trace_graph = traced_model.graph_for(x)
            self.assertTrue(any(n.kind() == "aten::batch_norm" for n in trace_graph.nodes()))
            # TODO check weight_prepack.

    def test_optimize_bf16_model(self):
        model = ConvBatchNorm()
        optimized_model = ipex.optimize(model.eval(), dtype=torch.bfloat16)
        # model should not has master weight attr for infernence model.
        self.assertTrue(not hasattr(optimized_model.conv, 'master_weight'))
        # model should has master weight attr for infernence model.
        sgd = torch.optim.SGD(model.parameters(), lr=0.1)
        optimized_model, optimized_sgd = ipex.optimize(model.train(), optimizer=sgd, dtype=torch.bfloat16, split_master_weight_for_bf16=False)
        self.assertTrue(hasattr(optimized_model.conv, 'master_weight'))

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
            elif dtype is torch.float32 and not auto_kernel_selection:
              self.assertTrue(isinstance(opt_M.linear, torch.nn.Linear))
              self.assertTrue(isinstance(opt_M.conv, _IPEXConv2d))
            else:
              self.assertTrue(isinstance(opt_M.linear, _IPEXLinear))
              self.assertTrue(isinstance(opt_M.conv, _IPEXConv2d))

if __name__ == '__main__':
    test = unittest.main()
