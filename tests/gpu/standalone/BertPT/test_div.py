import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

shapes = [
        (1, 16, 512, 512)
]

class TestTorchMethod(TestCase):
    def test_div(self, dtype=torch.float):
        for shape in shapes:
            #print("\n================== test shape: ", shape, "==================")
            user_cpu = torch.randn(shape, device=cpu_device)
            res_cpu = torch.div(user_cpu, 2)
            #print("begin xpu compute:")
            user_xpu = user_cpu.to("xpu")
            res_xpu = torch.div(user_xpu, 2)
            #print("xpu result:")
            #print(res_xpu.cpu())
            self.assertEqual(res_cpu, res_xpu.cpu())

            res_cpu.requires_grad_(True)
            res_xpu.requires_grad_(True)
            res_cpu.backward(user_cpu)
            res_xpu.backward(user_xpu)
            self.assertEqual(user_cpu.grad, user_xpu.grad)
