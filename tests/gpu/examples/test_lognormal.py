import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import ipex

import matplotlib.pyplot as plt
import pytest

cpu_device = torch.device('cpu')
dpcpp_device = torch.device("xpu")


class TestLogNormal(TestCase):
    @pytest.mark.skipif("not torch.xpu.has_onemkl()")
    def test_lognormal(self, dtype=torch.float):
        lognormal = torch.ones(1000000, device=dpcpp_device)
        torch.xpu.manual_seed(100)  # manually set seed
        lognormal.log_normal_(std=1 / 4)

        lognormal_1 = torch.ones(1000000, device=dpcpp_device)
        torch.xpu.manual_seed(100)
        lognormal_1.log_normal_(std=1 / 4)

        print("lognormal_1.device:", lognormal_1.device)
        print("lognormal.device:", lognormal.device)

        self.assertEqual(lognormal_1.cpu(), lognormal.cpu())
