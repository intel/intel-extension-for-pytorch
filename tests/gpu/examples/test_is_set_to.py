import torch
from torch.testing._internal.common_utils import TestCase
import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_is_set_to(self, dtype=torch.float):

        tensor1 = torch.tensor([1, 2, 3], device=cpu_device)
        tensor2 = torch.tensor([4, 5, 6], device=cpu_device)
        print("CPU:")
        print(tensor1.is_set_to(tensor2))
        print(tensor1.is_set_to(tensor1))

        print("DPCPP:")
        tensor1_dpcpp = torch.tensor([1, 2, 3], device=dpcpp_device)
        tensor2_dpcpp = torch.tensor([4, 5, 6], device=dpcpp_device)
        print(tensor1_dpcpp.is_set_to(tensor2_dpcpp))
        print(tensor1_dpcpp.is_set_to(tensor1_dpcpp))
        self.assertEqual(tensor1.is_set_to(tensor2),
                         tensor1_dpcpp.is_set_to(tensor2_dpcpp))
        self.assertEqual(tensor1.is_set_to(tensor1),
                         tensor1_dpcpp.is_set_to(tensor1_dpcpp))
