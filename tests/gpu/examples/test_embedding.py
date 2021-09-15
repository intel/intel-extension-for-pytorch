import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_embedding(self, dtype=torch.float):

        print("Weights ...")
        # embed = nn.Embedding(138493, 64)
        embed = nn.Embedding(10, 3)
        embed.weight.data.normal_(0., 0.01)
        print(embed.weight)
        print()

        print("Indices ...")
        user_cpu = torch.zeros([8], device=cpu_device, dtype=torch.long)
        user_cpu[0] = 2
        user_cpu[1] = 6
        user_cpu[2] = 1
        user_cpu[3] = 9
        user_cpu[4] = 2
        user_cpu[5] = 2
        user_cpu[6] = 9
        user_cpu[7] = 4
        print(user_cpu)
        print()

        print("CPU Forward ...")
        res_cpu = embed(user_cpu)
        print(res_cpu)
        print()

        grad_cpu = torch.ones(res_cpu.shape, device=cpu_device)
        grad_cpu = grad_cpu + grad_cpu

        print("CPU Backward ...")
        embed.zero_grad()
        res_cpu.backward(grad_cpu)
        for param in embed._parameters.values():
            print(param._grad)
        print()

        print("SYCL Forward ...")
        embed.to(dpcpp_device)
        res_dpcpp = embed(user_cpu.to(dpcpp_device))
        print(res_dpcpp.to("cpu"))
        print()

        print("SYCL Backward ...")
        embed.zero_grad()
        res_dpcpp.backward(grad_cpu.to(dpcpp_device))
        for param in embed._parameters.values():
            print(param._grad.device)
            print(param._grad.to("cpu"))
        print()
        self.assertEqual(res_cpu, res_dpcpp.to(cpu_device))
