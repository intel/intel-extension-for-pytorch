import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_max_1(self, dtype=torch.float):
        #
        # Test max OP.
        #
        #print("Testing max OP!\n")
        a_dpcpp = torch.randn(1, 2).to("xpu")
        #print("1-test (-2) dim")

        a_cpu = a_dpcpp.to("cpu")
        b_cpu = torch.max(a_cpu, -2)
        #print("For Tensor:", a_cpu)
        #print("torch.max on cpu returns", b_cpu)
        b_dpcpp, b_dpcpp_index = a_dpcpp.max(-2)
        # print(
        #     "torch.max on dpcpp device returns",
        #     b_dpcpp.to("cpu"),
        #     b_dpcpp_index.to("cpu"),
        # )
        # print("\n")
        self.assertEqual(b_cpu[0], b_dpcpp.cpu())
        self.assertEqual(b_cpu[1], b_dpcpp_index.cpu())

    def test_max_2(self, dtype=torch.float):
        #
        # Test max OP.
        #
        #print("Testing max OP!\n")
        a_dpcpp = torch.randn(20, 30522).to("xpu")
        #print("1-test (-2) dim")

        a_cpu = a_dpcpp.to("cpu")
        b_cpu = torch.max(a_cpu, -2)
        #print("For Tensor:", a_cpu)
        #print("torch.max on cpu returns", b_cpu)
        b_dpcpp, b_dpcpp_index = a_dpcpp.max(-2)
        # print(
        #     "torch.max on dpcpp device returns",
        #     b_dpcpp.to("cpu"),
        #     b_dpcpp_index.to("cpu"),
        # )
        # print("\n")
        self.assertEqual(b_cpu[0], b_dpcpp.cpu())
        self.assertEqual(b_cpu[1], b_dpcpp_index.cpu())
