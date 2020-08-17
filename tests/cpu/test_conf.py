import math
import random
import unittest
from functools import reduce

import torch
import _torch_ipex as ipex

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import gradcheck
from torch.autograd.gradcheck import gradgradcheck
from torch._six import inf, nan

from common_utils import TestCase, iter_indices, TEST_NUMPY, TEST_SCIPY, TEST_MKL, \
    TEST_LIBROSA, run_tests, download_file, skipIfNoLapack, suppress_warnings, \
    IS_WINDOWS, PY3, NO_MULTIPROCESSING_SPAWN, do_test_dtypes, do_test_empty_full, \
    IS_SANDCASTLE, load_tests, brute_pdist, brute_cdist, slowTest, \
    skipCUDANonDefaultStreamIf, skipCUDAMemoryLeakCheckIf

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
        ipex.enable_train()
        self.assertTrue(ipex.get_train())
        ipex.disable_train()
        self.assertFalse(ipex.get_train())

    def test_jit_fuse(self):
        self.assertTrue(ipex.get_jit_opt())
        ipex.disable_jit_opt()
        self.assertFalse(ipex.get_jit_opt())
        ipex.enable_jit_opt()
        self.assertTrue(ipex.get_jit_opt())

if __name__ == '__main__':
    test = unittest.main()
