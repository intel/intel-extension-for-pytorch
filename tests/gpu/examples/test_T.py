import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTensorMethod(TestCase):
    def test_T_and_t(self, dtype=torch.float):
        x_cpu = torch.randn(2, 3, dtype=dtype)
        print('x_cpu = ', x_cpu)
        print('x_cpu.T = ', x_cpu.T)

        x_dpcpp = x_cpu.to(dpcpp_device)
        print('x_cpu = ', x_dpcpp)
        print('x_cpu.T = ', x_dpcpp.T)
        self.assertEqual(x_cpu.T, x_dpcpp.T.to(cpu_device))

        x_cpu_t = x_cpu.t()
        x_dpcpp_t = x_dpcpp.t()
        print('x_cpu = ', x_cpu_t)
        print('x_cpu.T = ', x_dpcpp_t.to(cpu_device))
        self.assertEqual(x_cpu_t, x_dpcpp_t.to(cpu_device))

        x_cpu_t = torch.t(x_cpu)
        x_dpcpp_t = torch.t(x_dpcpp)
        print('x_cpu = ', x_cpu_t)
        print('x_cpu.T = ', x_dpcpp_t.to(cpu_device))
        self.assertEqual(x_cpu_t, x_dpcpp_t.to(cpu_device))
