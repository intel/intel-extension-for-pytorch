import torch
import intel_pytorch_extension as ipex
from torch.testing._internal.common_utils import TestCase
import unittest
import itertools
import copy

class TestModule(torch.nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.linear = torch.nn.Linear(5, 10)
        self.conv = torch.nn.Conv2d(1, 10, 5, 1)
        self.bn = torch.nn.BatchNorm2d(num_features=10)
        self.embeddingbag = torch.nn.EmbeddingBag(10, 3, mode='sum')

    def forward(self, x, y, indices, offsets):
        x = self.conv(x)
        x = self.bn(x)
        y = self.linear(y)
        z = self.embeddingbag(indices, offsets)
        return x + y

class TestOptimizeCases(TestCase):
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
              # fused part cannot be inplaced
              self.assertTrue(M.conv.weight.data_ptr() != opt_M.conv.weight.data_ptr())
              # O1 level will prepack the linear weight, cannot be inplaced
              if level == "O1":
                  self.assertTrue(M.linear.weight.data_ptr() != opt_M.linear.weight.data_ptr())
              else:
                  self.assertTrue(M.linear.weight.data_ptr() == opt_M.linear.weight.data_ptr())
              # non optimized part should be inplaced
              self.assertTrue(M.embeddingbag.weight.data_ptr() == opt_M.embeddingbag.weight.data_ptr())

    def test_optimize_inplace_behavior_training_mode_without_optimizer(self):
          M_ori = TestModule()
          options = itertools.product([torch.float32, torch.bfloat16], ["O0", "O1"])
          for dtype, level in options:
              M = copy.deepcopy(M_ori).train()
              # non-inplace
              opt_M = ipex.optimize(M, dtype=dtype, level=level, inplace=False)
              self.assertTrue(M.linear.weight.data_ptr() != opt_M.linear.weight.data_ptr())
              self.assertTrue(M.conv.weight.data_ptr() != opt_M.conv.weight.data_ptr())
              self.assertTrue(M.embeddingbag.weight.data_ptr() != opt_M.embeddingbag.weight.data_ptr())

              # inplace
              M = copy.deepcopy(M_ori).train()
              opt_M = ipex.optimize(M, dtype=dtype, level=level, inplace=True)
              # training mode will not generate GraphModule by convbn fusion, inplace indicate directly
              # modify the module given by user
              self.assertTrue(M.conv.weight.data_ptr() == opt_M.conv.weight.data_ptr())
              self.assertTrue(M.linear.weight.data_ptr() == opt_M.linear.weight.data_ptr())
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
              
              # inplace
              M = copy.deepcopy(M_ori).train()
              sgd = torch.optim.SGD(M.parameters(), lr=0.1)
              opt_M, _ = ipex.optimize(M, dtype=dtype, optimizer=sgd, level=level, inplace=True)
              self.assertTrue(M.linear.weight.data_ptr() == opt_M.linear.weight.data_ptr())
              self.assertTrue(M.conv.weight.data_ptr() == opt_M.conv.weight.data_ptr())
              self.assertTrue(M.embeddingbag.weight.data_ptr() == opt_M.embeddingbag.weight.data_ptr())

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
if __name__ == '__main__':
    test = unittest.main()
