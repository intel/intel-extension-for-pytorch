import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import ipex

import pytest

cpu_device = torch.device("cpu")
sycl_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_index_randperm(self):
        src = torch.empty(150, 45, device=sycl_device).random_(0, 2**22)
        idx = torch.randperm(src.shape[0], device=sycl_device)
        res = src[idx]
        res_cpu = src.cpu()[idx.cpu()]
        self.assertEqual(res.cpu(), res_cpu)
