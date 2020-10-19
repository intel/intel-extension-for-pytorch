import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")

class TestTensorMethod(TestCase):
    def test_split(self, dtype=torch.float):
        input_cpu = torch.arange(10).reshape(5,2)
        input_dpcpp = input_cpu.to(dpcpp_device)

        print('input_cpu = ', input_cpu)
        print('input_dpcpp = ', input_dpcpp)

        self.assertEqual(input_cpu.split(split_size=2), input_dpcpp.split(split_size=2))
        self.assertEqual(input_cpu.split(split_size=[1, 4]), input_dpcpp.split(split_size=[1, 4]))
