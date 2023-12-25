import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

shapes_add = [
            ((1, 2048, 7, 7),(1, 2048, 7, 7)),
            ((1, 64, 56, 56),(1, 64, 56, 56)),
            ((1, 256, 56, 56),(1, 256, 56, 56)),
            ((1, 512, 28, 28),(1, 512, 28, 28)),
            ((1, 1024, 14, 14),(1, 1024, 14, 14)),
            ((2, 384, 1024),(2, 384, 1024)),
            ((2, 16, 384, 384),(2, 1, 1, 384)),
        ] 
shapes_add_ = [  
            ((2, 384, 1024),(1, 384, 1024)),
        ]

class TestTorchMethod(TestCase):
    def test_add(self, dtype=torch.float):
        for shape in shapes_add:
            print("\n================== test shape: ", shape, "==================")
            s_cpu = torch.randn(shape[0], dtype=dtype, device=cpu_device)
            x_cpu = torch.randn(shape[1], dtype=dtype, device=cpu_device)
            s_xpu = s_cpu.to(dpcpp_device)
            x_xpu = x_cpu.to(dpcpp_device)

            print("s_cpu = ", s_cpu)
            print("s_xpu = ", s_xpu.to(cpu_device))
            print("x_cpu = ", x_cpu)
            print("x_xpu = ", x_xpu.to(cpu_device))

            y_cpu = torch.add(x_cpu, s_cpu)
            y_xpu = torch.add(x_xpu, s_xpu)

            print("sum cpu = ", y_cpu)
            print("sum xpu = ", y_xpu.to(cpu_device))

            self.assertEqual(y_cpu, y_xpu.cpu())
    
    def test_add_(self, dtype=torch.float):
        for shape in shapes_add_:
            print("\n================== test shape: ", shape, "==================")
            x = torch.randn(shape[0], dtype=dtype, device=cpu_device)
            y = torch.randn(shape[1], dtype=dtype, device=cpu_device)
            x_xpu = x.xpu()
            y_xpu = y.xpu()
            y_cpu=x.add_(y)
            y_xpu=x_xpu.add_(y_xpu)
            self.assertEqual(y_cpu, y_xpu.cpu())
