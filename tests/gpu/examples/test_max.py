import torch
from torch.testing._internal.common_utils import TestCase

import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_max(self, dtype=torch.float):

        #
        # Test maxall OP.
        #
        # y = torch.randn(3, 3)
        #
        # y_dpcpp = y.to("xpu")
        # z_dpcpp = torch.max(y_dpcpp)
        #
        # print("Testing maxall OP!\n")
        # print("For Tensor:", y)
        # print("torch.max on cpu returns", torch.max(y))
        # print("torch.max on dpcpp device returns", z_dpcpp.to("cpu"))
        # print("\n")

        #
        # Test cmax OP.
        #
        # print("Testing cmax OP!\n")
        # c_dpcpp = torch.randn(4).to("xpu")
        # d_dpcpp = torch.tensor([[ 0.0120, -0.9505, -0.3025, -1.4899]], device = dpcpp_device)
        # e_dpcpp = torch.max(c_dpcpp, d_dpcpp)
        #
        # print("For Tensor:", c_dpcpp.to("cpu"))
        # print("For Tensor:", d_dpcpp.to("cpu"))
        # print("torch.max on cpu returns", torch.max(c_dpcpp.to("cpu"), d_dpcpp.to("cpu")))
        # print("torch.max on dpcpp device returns", e_dpcpp.to("cpu"))
        # print("\n")

        #
        # Test max OP.
        #
        print("Testing max OP!\n")
        a_dpcpp = torch.randn((4, 10)).to("xpu")

        print("1-test (-2) dim")

        a_cpu = a_dpcpp.to("cpu")
        b_cpu = torch.max(a_cpu, -2)
        print("For Tensor:", a_cpu)
        print("torch.max on cpu returns", b_cpu)
        b_dpcpp, b_dpcpp_index = a_dpcpp.max(-2)
        print("torch.max on dpcpp device returns",
              b_dpcpp.to("cpu"), b_dpcpp_index.to("cpu"))
        print("\n")
        self.assertEqual(b_cpu[0], b_dpcpp.cpu())
        self.assertEqual(b_cpu[1], b_dpcpp_index.cpu())

        print("2-test (-1) dim")

        a_cpu = a_dpcpp.to("cpu")
        b_cpu = torch.max(a_cpu, -1)
        print("For Tensor:", a_cpu)
        print("torch.max on cpu returns", b_cpu)
        b_dpcpp, b_dpcpp_index = a_dpcpp.max(-1)
        print("torch.max on dpcpp device returns",
              b_dpcpp.to("cpu"), b_dpcpp_index.to("cpu"))
        print("\n")
        self.assertEqual(b_cpu[0], b_dpcpp.cpu())
        self.assertEqual(b_cpu[1], b_dpcpp_index.cpu())

        print("3-test (0) dim")
        a_cpu = a_dpcpp.to("cpu")
        b_cpu = torch.max(a_cpu, 0)
        print("For Tensor:", a_cpu)
        print("torch.max on cpu returns", b_cpu)
        b_dpcpp, b_dpcpp_index = torch.max(a_dpcpp, 0)
        print("torch.max on dpcpp device returns",
              b_dpcpp.to("cpu"), b_dpcpp_index.to("cpu"))
        print("\n")
        self.assertEqual(b_cpu[0], b_dpcpp.cpu())
        self.assertEqual(b_cpu[1], b_dpcpp_index.cpu())

        print("4-test (1) dim")
        a_cpu = a_dpcpp.to("cpu")
        b_cpu = torch.max(a_cpu, 1)
        print("For Tensor:", a_cpu)
        print("torch.max on cpu returns", b_cpu)
        b_dpcpp, b_dpcpp_index = torch.max(a_dpcpp, 1)
        print("torch.max on dpcpp device returns",
              b_dpcpp.to("cpu"), b_dpcpp_index.to("cpu"))
        print("\n")
        self.assertEqual(b_cpu[0], b_dpcpp.cpu())
        self.assertEqual(b_cpu[1], b_dpcpp_index.cpu())
