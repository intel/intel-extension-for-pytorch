import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex
import torch.nn as nn
import matplotlib.pyplot as plt

cpu_device = torch.device('cpu')
dpcpp_device = torch.device('dpcpp')


class TestTorchMethod(TestCase):
    def test_cholesky_inverse(self, dtype=torch.float):
        a = torch.randn(3, 3).to(cpu_device)
        
        a = torch.mm(a, a.t()) + 1e-05 * torch.eye(3) # make symmetric positive definite
        u = torch.cholesky(a)
        print("a", a)
        
        t = torch.cholesky_inverse(u)
        print("cpu", t)
        t_dpcpp = torch.cholesky_inverse(u.to(dpcpp_device))
        print("dpcpp", t_dpcpp.to(cpu_device))
        self.assertEqual(t, t_dpcpp.to(cpu_device))
       
        t1 = torch.cholesky_inverse(u, upper=True)
        print("cpu", t1)
        t1_dpcpp = torch.cholesky_inverse(u.to(dpcpp_device), upper=True)
        print("dpcpp", t1_dpcpp.to(cpu_device))
        self.assertEqual(t1, t1_dpcpp.to(cpu_device))
