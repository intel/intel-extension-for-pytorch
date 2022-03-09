import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTensorMethod(TestCase):
    def test_unbind(self, dtype=torch.float):
        x_cpu = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        x_dpcpp = x_cpu.to(dpcpp_device)
        print('x_cpu = ', x_cpu)
        print('x_dpcpp = ', x_dpcpp)

        print('x_cpu.unbind() = ', x_cpu.unbind())
        print('x_dpcpp.unbind() = ', x_dpcpp.unbind())
        self.assertEqual(x_cpu.unbind(), x_dpcpp.unbind())

        print('torch.unbind(x_cpu) = ', torch.unbind(x_cpu))
        print('torch.unbind(x_dpcpp) = ', torch.unbind(x_dpcpp))
        self.assertEqual(torch.unbind(x_cpu), torch.unbind(x_dpcpp))
