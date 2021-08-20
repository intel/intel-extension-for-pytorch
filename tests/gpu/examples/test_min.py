import torch
from torch.testing._internal.common_utils import TestCase
import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_min(self, dtype=torch.float):
        #
        # Test minall OP.

        y = torch.randn(3, 3)

        y_dpcpp = y.to("xpu")
        z_dpcpp = torch.min(y_dpcpp)

        print("Testing minall OP!\n")
        print("For Tensor:", y)
        print("torch.min on cpu returns", torch.min(y))
        print("torch.min on dpcpp device returns", z_dpcpp.to("cpu"))
        print("\n")
        self.assertEqual(y, y_dpcpp.cpu())
        self.assertEqual(torch.min(y), z_dpcpp.cpu())

        #
        # Test cmin OP.
        #
        print("Testing cmin OP!\n")
        c_dpcpp = torch.randn(4).to("xpu")
        d_dpcpp = torch.tensor(
            [0.0120, -0.9505, -0.3025, -1.4899], device=dpcpp_device)
        e_dpcpp = torch.min(c_dpcpp, d_dpcpp)

        print("For Tensor:", c_dpcpp.to("cpu"))
        print("For Tensor:", d_dpcpp.to("cpu"))
        print("torch.min on cpu returns", torch.min(
            c_dpcpp.to("cpu"), d_dpcpp.to("cpu")))
        print("torch.min on dpcpp device returns", e_dpcpp.to("cpu"))
        print("\n")
        self.assertEqual(torch.min(
            c_dpcpp.to("cpu"), d_dpcpp.to("cpu")), e_dpcpp.to("cpu"))

        #
        # Test min OP.
        #
        print("Testing min OP!\n")
        a_dpcpp = torch.randn((9, 5)).to("xpu")

        print("1-test (-2) dim")
        a_cpu = a_dpcpp.to("cpu")
        b_dpcpp, b_dpcpp_index = a_dpcpp.min(-2)

        print("For Tensor:", a_cpu)
        print("torch.min on cpu returns", torch.min(a_cpu, -2))
        print("torch.min on dpcpp device returns",
              b_dpcpp.to("cpu"), b_dpcpp_index.to("cpu"))
        print("\n")
        self.assertEqual(torch.min(a_cpu, -2)[0], b_dpcpp.to("cpu"))
        self.assertEqual(torch.min(a_cpu, -2)[1], b_dpcpp_index.to("cpu"))

        print("2-test (-1) dim")
        a_cpu = a_dpcpp.to("cpu")
        b_dpcpp, b_dpcpp_index = a_dpcpp.min(-1)

        print("For Tensor:", a_cpu)
        print("torch.min on cpu returns", torch.min(a_cpu, -1))
        print("torch.min on dpcpp device returns",
              b_dpcpp.to("cpu"), b_dpcpp_index.to("cpu"))
        print("\n")
        self.assertEqual(torch.min(a_cpu, -1)[0], b_dpcpp.to("cpu"))
        self.assertEqual(torch.min(a_cpu, -1)[1], b_dpcpp_index.to("cpu"))

        print("3-test (0) dim")
        a_cpu = a_dpcpp.to("cpu")
        b_dpcpp, b_dpcpp_index = torch.min(a_dpcpp, 0)

        print("For Tensor:", a_cpu)
        print("torch.min on cpu returns", torch.min(a_cpu, 0))
        print("torch.min on dpcpp device returns",
              b_dpcpp.to("cpu"), b_dpcpp_index.to("cpu"))
        print("\n")
        self.assertEqual(torch.min(a_cpu, 0)[0], b_dpcpp.to("cpu"))
        self.assertEqual(torch.min(a_cpu, 0)[1], b_dpcpp_index.to("cpu"))

        print("4-test (1) dim")
        a_cpu = a_dpcpp.to("cpu")
        b_dpcpp, b_dpcpp_index = torch.min(a_dpcpp, 1)

        print("For Tensor:", a_cpu)
        print("torch.min on cpu returns", torch.min(a_cpu, 1))
        print("torch.min on dpcpp device returns",
              b_dpcpp.to("cpu"), b_dpcpp_index.to("cpu"))
        print("\n")
        self.assertEqual(torch.min(a_cpu, 1)[0], b_dpcpp.to("cpu"))
        self.assertEqual(torch.min(a_cpu, 1)[1], b_dpcpp_index.to("cpu"))
