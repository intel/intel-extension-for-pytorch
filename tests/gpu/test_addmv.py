import torch
import torch.nn as nn

import torch_ipex
from torch.testing._internal.common_utils import TestCase


dpcpp_device = torch.device("dpcpp")
cpu_device = torch.device("cpu")


class  TestTorchMethod(TestCase):
    def test_addmv(self, dtype=torch.float):

        m1_cpu = torch.randn([2, 3], dtype=dtype)
        m2_cpu = torch.randn([3], dtype=dtype)
        x_cpu = torch.ones([2], dtype=dtype)
        x_cpu2 = torch.ones([1], dtype=dtype)

        m1_dpcpp = m1_cpu.to(dpcpp_device)
        m2_dpcpp = m2_cpu.to(dpcpp_device)
        x_dpcpp = x_cpu.to(dpcpp_device)
        x_dpcpp2 = x_cpu2.to(dpcpp_device)
        
        print("cpu addmm_ self", x_cpu)
        x_cpu.addmv_(m1_cpu, m2_cpu)
        print("cpu addmm_ result", x_cpu)

        print("dpcpp addmm_ self", x_dpcpp.cpu())
        x_dpcpp.addmv_(m1_dpcpp, m2_dpcpp)
        print("dpcpp addmm_ result", x_dpcpp.cpu())
        self.assertEqual(x_cpu,x_dpcpp.cpu())
        
        print("cpu addmv_ self", x_cpu2)
        y = x_cpu2.addmv(m1_cpu, m2_cpu)
        print("cpu addmv_ result", y)

        print("dpcpp addmv_ self", x_dpcpp2.cpu())
        y_sycl = x_dpcpp2.addmv(m1_dpcpp, m2_dpcpp)
        print("dpcpp addmv_ result", y_sycl.cpu())
        self.assertEqual(y, y_sycl.cpu())

