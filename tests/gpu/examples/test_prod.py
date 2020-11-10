import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_prod(self, dtype=torch.float):

        input = torch.randn(4, dtype=torch.float32, device=torch.device("cpu"))
        print("cpu input:", input)
        print("cpu output:", torch.prod(input))

        input_dpcpp = input.to("xpu")
        print("gpu input:", input_dpcpp.cpu())
        print("gpu output:", torch.prod(input_dpcpp).cpu())
        self.assertEqual(input, input_dpcpp.cpu())
        self.assertEqual(torch.prod(input), torch.prod(input_dpcpp).cpu())
