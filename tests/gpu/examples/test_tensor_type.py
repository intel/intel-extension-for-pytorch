import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_tensor_type(self, dtype=torch.float):
        tensor1 = torch.rand([2, 3], dtype=dtype)
        tensor2 = tensor1.clone().type(torch.xpu.IntTensor)
        tensor1 = tensor1.type(torch.IntTensor)
        self.assertEqual(tensor1, tensor2.to(cpu_device))
