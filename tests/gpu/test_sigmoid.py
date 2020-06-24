import numpy
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestNNMethod(TestCase):
    def test_sigmoid(self, dtype=torch.float):

        user_cpu = torch.tensor([[1.11, 2.22, 3.33], [4.44, 5.55, 6.66]],
                                device=cpu_device, requires_grad=True)
        m = nn.Sigmoid()
        cpu_res = m(user_cpu)
        print(cpu_res)
        cpu_res.backward(torch.tensor(
            [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]], device=cpu_device))
        print(user_cpu.grad)

        user_dpcpp = torch.tensor([[1.11, 2.22, 3.33], [
            4.44, 5.55, 6.66]], device=dpcpp_device, requires_grad=True)
        dpcpp_res = m(user_dpcpp)
        print(dpcpp_res.to("cpu"))
        dpcpp_res.backward(torch.tensor(
            [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]], device=dpcpp_device))
        print(user_dpcpp.grad.to("cpu"))
        self.assertEqual(cpu_res,  dpcpp_res.cpu())
        self.assertEqual(user_cpu.grad, user_dpcpp.grad.cpu())
