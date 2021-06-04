import math
import random
import unittest
from functools import reduce

import torch
import torch_ipex._C as ipex

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

class TestOptConf(TestCase):
    def test_auto_dnnl(self):
        self.assertTrue(ipex.get_auto_dnnl())
        ipex.disable_auto_dnnl()
        self.assertFalse(ipex.get_auto_dnnl())
        ipex.enable_auto_dnnl()
        self.assertTrue(ipex.get_auto_dnnl())

    def test_mix_bf16_fp32(self):
        self.assertFalse(ipex.get_mix_bf16_fp32())
        ipex.enable_mix_bf16_fp32()
        self.assertTrue(ipex.get_mix_bf16_fp32())
        ipex.disable_mix_bf16_fp32()
        self.assertFalse(ipex.get_mix_bf16_fp32())

    def test_mix_bf16_fp32_train(self):
        self.assertFalse(ipex.get_train())
        ipex.set_execution_mode(train=True)
        self.assertTrue(ipex.get_train())
        ipex.set_execution_mode(train=False)
        self.assertFalse(ipex.get_train())

    def test_jit_fuse(self):
        self.assertTrue(ipex.get_jit_opt())
        ipex.disable_jit_opt()
        self.assertFalse(ipex.get_jit_opt())
        ipex.enable_jit_opt()
        self.assertTrue(ipex.get_jit_opt())

if __name__ == '__main__':
    test = unittest.main()
