import math
import random
import unittest
import time

from functools import reduce
import torch
import torch_ipex as ipex
# import intel_pytorch_extension as ipex

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import gradcheck
from torch.autograd.gradcheck import gradgradcheck
from torch._six import inf, nan

from common_utils import (
    TestCase, TEST_WITH_ROCM, run_tests,
    IS_WINDOWS, IS_FILESYSTEM_UTF8_ENCODING, NO_MULTIPROCESSING_SPAWN,
    do_test_dtypes, IS_SANDCASTLE, IS_FBCODE, IS_REMOTE_GPU, load_tests, slowTest,
    skipCUDAMemoryLeakCheckIf, BytesIOContext,
    skipIfRocm, skipIfNoSciPy, TemporaryFileName, TemporaryDirectoryName,
    wrapDeterministicFlagAPITest, DeterministicGuard, make_tensor)

K=1 #128
C=16 #64
MB = 28


class TestMLPCases(TestCase):
  def get_rand_seed(self):
      return int(time.time() * 1000000000)

  def _ipxex_linear(self, random_seed, data_type = torch.float32):
    torch.manual_seed(random_seed)
    fc = ipex.IpexMLPLinear(C, K).to(data_type)
    return fc

  def _cpu_linear(self, random_seed, data_type = torch.float32):
    torch.manual_seed(random_seed)
    fc = torch.nn.Linear(C, K).to(data_type)
    return fc

  def _run_mlp(self, random_seed, fc_module, data_type = torch.float32):
    torch.manual_seed(random_seed)
    x1 = torch.randn(MB, C, requires_grad=True).to(data_type).requires_grad_(True)
    y1 = fc_module(x1)
    z1 = y1.mean()
    z1.backward()
    return x1.grad, fc_module.weight.grad, fc_module.bias.grad

  def test_mlp(self):
    for data_type in [torch.float32, torch.bfloat16]:
      seed = self.get_rand_seed()
      ipex_fc = self._ipxex_linear(seed, data_type)
      cpu_fc = self._cpu_linear(seed, data_type)

      seed = self.get_rand_seed()
      input_grad_ipex, weight_grad_ipex, bias_grad_ipex = self._run_mlp(seed, ipex_fc, data_type)
      input_grad_cpu, weight_grad_cpu, bias_grad_cpu = self._run_mlp(seed, cpu_fc, data_type)

      if input_grad_ipex is None:
        self.assertTrue(input_grad_cpu is None)
      else:
        self.assertEqual(input_grad_ipex.to(torch.float32), input_grad_cpu.to(torch.float32), atol=1e-1, rtol=1e-5)
      self.assertEqual(weight_grad_ipex.to(torch.float32), weight_grad_cpu.to(torch.float32), atol=1e-1, rtol=1e-5)
      self.assertEqual(bias_grad_ipex.to(torch.float32), bias_grad_cpu.to(torch.float32), atol=1e-1, rtol=1e-5)

if __name__ == '__main__':
    test = unittest.main()
