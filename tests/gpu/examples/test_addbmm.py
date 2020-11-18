import torch
import torch.nn as nn

import torch_ipex
from torch.testing._internal.common_utils import TestCase


dpcpp_device = torch.device("xpu")
cpu_device = torch.device("cpu")


class  TestTorchMethod(TestCase):
    def test_addbmm(self, dtype=torch.float):

        m1_cpu = torch.randn([10, 3, 4], dtype=dtype)
        m2_cpu = torch.randn([10, 4, 2], dtype=dtype)
        x_cpu = torch.ones([3, 2], dtype=dtype)
        x_cpu2 = torch.ones([3, 2], dtype=dtype)

        m1_dpcpp = m1_cpu.to(dpcpp_device)
        m2_dpcpp = m2_cpu.to(dpcpp_device)
        x_dpcpp = x_cpu.to(dpcpp_device)
        x_dpcpp2 = x_cpu2.to(dpcpp_device)
    
        alpha = 2.0
        beta = 3.5
        
        print("cpu addbmm_ self", x_cpu)
        x_cpu.addbmm_(m1_cpu, m2_cpu, beta=beta, alpha=alpha)
        print("cpu addbmm_ result", x_cpu)

        print("dpcpp addbmm_ self", x_dpcpp.cpu())
        x_dpcpp.addbmm_(m1_dpcpp, m2_dpcpp, beta=beta, alpha=alpha)
        print("dpcpp addbmm_ result", x_dpcpp.cpu())
        self.assertEqual(x_cpu,x_dpcpp.cpu())
       
        print("cpu addbmm self", x_cpu)
        y = x_cpu2.addbmm(m1_cpu, m2_cpu, beta=beta, alpha=alpha)
        print("cpu addbmm result", y)

        print("dpcpp addbmm self", x_dpcpp.cpu())
        y_sycl = x_dpcpp2.addbmm(m1_dpcpp, m2_dpcpp, beta=beta, alpha=alpha)
        print("dpcpp addbmm result", y_sycl.cpu())
        self.assertEqual(y, y_sycl.cpu())
