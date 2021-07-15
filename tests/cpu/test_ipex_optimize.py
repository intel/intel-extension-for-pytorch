import torch
import intel_pytorch_extension as ipex
from torch.testing._internal.common_utils import TestCase
import unittest
import itertools

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
          M = TestModule().eval()
          options = itertools.product([torch.float32, torch.bfloat16], ["O0", "O1"])
          for dtype, level in options:
              # non-inplace
              opt_M = ipex.optimize(M, dtype=dtype, level=level, inplace=False)
              self.assertTrue(M.linear.weight.data_ptr() != opt_M.linear.weight.data_ptr())
              self.assertTrue(M.conv.weight.data_ptr() != opt_M.conv.weight.data_ptr())
              self.assertTrue(M.embeddingbag.weight.data_ptr() != opt_M.embeddingbag.weight.data_ptr())

              # inplace
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
          M = TestModule().train()
          sgd = torch.optim.SGD(M.parameters(), lr=0.1)
          options = itertools.product([torch.float32, torch.bfloat16], ["O0", "O1"])
          for dtype, level in options:
              # non-inplace
              opt_M = ipex.optimize(M, dtype=dtype, level=level, inplace=False)
              self.assertTrue(M.linear.weight.data_ptr() != opt_M.linear.weight.data_ptr())
              self.assertTrue(M.conv.weight.data_ptr() != opt_M.conv.weight.data_ptr())
              self.assertTrue(M.embeddingbag.weight.data_ptr() != opt_M.embeddingbag.weight.data_ptr())

              # inplace
              opt_M = ipex.optimize(M, dtype=dtype, level=level, inplace=True)
              # training mode will not generate GraphModule by convbn fusion, inplace indicate directly
              # modify the module given by user
              self.assertTrue(M.conv.weight.data_ptr() == opt_M.conv.weight.data_ptr())
              self.assertTrue(M.linear.weight.data_ptr() == opt_M.linear.weight.data_ptr())
              self.assertTrue(M.embeddingbag.weight.data_ptr() == opt_M.embeddingbag.weight.data_ptr())

    def test_optimize_inplace_behavior_training_mode_with_optimizer(self):
          M = TestModule().train()
          sgd = torch.optim.SGD(M.parameters(), lr=0.1)
          options = itertools.product([torch.float32, torch.bfloat16], ["O0", "O1"])
          for dtype, level in options:
              # non-inplace
              opt_M, _ = ipex.optimize(M, dtype=dtype, optimizer=sgd, level=level, inplace=False)
              self.assertTrue(M.linear.weight.data_ptr() != opt_M.linear.weight.data_ptr())
              self.assertTrue(M.conv.weight.data_ptr() != opt_M.conv.weight.data_ptr())
              self.assertTrue(M.embeddingbag.weight.data_ptr() != opt_M.embeddingbag.weight.data_ptr())

              opt_M, _ = ipex.optimize(M, dtype=dtype, optimizer=sgd, level=level, inplace=True)
              self.assertTrue(M.linear.weight.data_ptr() == opt_M.linear.weight.data_ptr())
              self.assertTrue(M.conv.weight.data_ptr() == opt_M.conv.weight.data_ptr())
              self.assertTrue(M.embeddingbag.weight.data_ptr() == opt_M.embeddingbag.weight.data_ptr())


if __name__ == '__main__':
    test = unittest.main()
