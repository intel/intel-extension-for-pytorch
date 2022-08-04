import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_RangeFactories(self, dtype=torch.float):

        # x=torch.tensor([1,1,1,1,1], device=cpu_device)
        x = torch.logspace(start=-10, end=10, steps=5, device=cpu_device)
        y = torch.linspace(start=-10, end=10, steps=5, device=cpu_device)
        z = torch.arange(1, 2.5, 0.5, device=cpu_device)
        n = torch.range(1, 2.5, 0.5, device=cpu_device)

        # x_dpcpp=x.to("xpu")
        x_out = torch.logspace(start=-10, end=10, steps=5, device=dpcpp_device)
        y_out = torch.linspace(start=-10, end=10, steps=5, device=dpcpp_device)
        z_out = torch.arange(1, 2.5, 0.5, device=dpcpp_device)
        n_out = torch.range(1, 2.5, 0.5, device=dpcpp_device)

        print("cpu: ")
        print(x)
        print(y)
        print(z)
        print(n)

        print("dpcpp: ")
        print(x_out.to("cpu"))
        print(y_out.to("cpu"))
        print(z_out.to("cpu"))
        print(n_out.to("cpu"))
        self.assertEqual(x[0], x_out[0].cpu())
        self.assertEqual(x[1], x_out[1].cpu())
        self.assertEqual(x[2], x_out[2].cpu())
        # The std POW in different compiler packages, gcc, computecpp and dpc++, give the results with big gap.
        # Vanilla cpu : 0x47c35000, DPC++ GPU : 0x47c35001, ComputeCpp GPU : 0x47c35002
        # The potential reason is differenct default rounding mode in compilers.
        # Here we enlarge the tolerance to pass this case.
        self.assertEqual(x[3], x_out[3].cpu(), 0.1)
        self.assertEqual(x[4], x_out[4].cpu())
        self.assertEqual(y, y_out.cpu())
        self.assertEqual(z, z_out.cpu())
        self.assertEqual(n, n_out.cpu())
