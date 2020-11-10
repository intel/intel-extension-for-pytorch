import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase
import torch_ipex
import os
import copy
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestNNMethod(TestCase):
    def test_conv_relu_fusion(self, dtype=torch.float):
        #env_origin = copy.deepcopy(os.environ)
        #os.environ["IPEX_LAZY_REORDER"] = "1"
        #os.environ["IPEX_WEIGHT_CACHE"] = "1"

        x_cpu = torch.randn([1, 1, 3, 3], device=cpu_device)
        y_cpu = torch.randn([1, 1, 3, 3], device=cpu_device)
        z_cpu = torch.randn([1, 1, 1, 1], device=cpu_device)
        
        ref1 = x_cpu + y_cpu
        ref2 = x_cpu + z_cpu
        ref3 = z_cpu + x_cpu
        ref4 = x_cpu + 2
        
        x_dpcpp = x_cpu.to("xpu")
        y_dpcpp = y_cpu.to("xpu")
        z_dpcpp = z_cpu.to("xpu")
        
        
        #real1 = x_dpcpp + y_dpcpp
        real2 = x_dpcpp + z_dpcpp
        real3 = z_dpcpp + x_dpcpp
        real4 = x_dpcpp + 2
        
        #print(ref1 - real1.cpu())
        print(ref2 - real2.cpu())
        print(ref3 - real3.cpu())
        print(ref4 - real4.cpu())
        
        #self.assertEqual(ref1, real1.to(cpu_device))
        self.assertEqual(ref2, real2.to(cpu_device))
        self.assertEqual(ref3, real3.to(cpu_device))
        self.assertEqual(ref4, real4.to(cpu_device))
