import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

import numpy

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_silu(self, dtype=torch.float):

        user_cpu = torch.tensor([[1.11, 2.22, 3.33], [4.44, 5.55, 6.66]],
                                device=cpu_device, requires_grad=True)
        silu = nn.SiLU()
        cpu_res = silu(user_cpu)
        print(cpu_res)
        cpu_res.backward(torch.tensor(
            [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]], device=cpu_device))
        user_grad_cpu = user_cpu.grad.clone()
        print(user_grad_cpu)
        silu.zero_grad()

        silu_dpcpp = silu.to('xpu')
        user_dpcpp = torch.tensor([[1.11, 2.22, 3.33], [
            4.44, 5.55, 6.66]], device=dpcpp_device, requires_grad=True)
        dpcpp_res = silu_dpcpp(user_dpcpp)
        print(dpcpp_res.to("cpu"))
        dpcpp_res.backward(torch.tensor(
            [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]], device=dpcpp_device))
        user_grad_dpcpp = user_dpcpp.grad.clone()
        print(user_grad_dpcpp.cpu())
        silu_dpcpp.zero_grad()

        self.assertEqual(cpu_res, dpcpp_res)
        self.assertEqual(user_grad_cpu, user_grad_dpcpp)
