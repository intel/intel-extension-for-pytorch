import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTensorMethod(TestCase):
    def test_meshgrid(self, dtype=torch.float):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        grid_x, grid_y = torch.meshgrid(x, y)
        print("\ngrid_x: ", grid_x)
        print("\ngrid_y: ", grid_y)

        print("---")
        x_dpcpp = x.to(dpcpp_device)
        y_dpcpp = y.to(dpcpp_device)
        grid_x_dpcpp, grid_y_dpcpp = torch.meshgrid(x_dpcpp, y_dpcpp)
        print("\ngrid_x_dpcpp: ", grid_x_dpcpp.to(cpu_device))
        print("\ngrid_y_dpcpp: ", grid_y_dpcpp.to(cpu_device))

        self.assertEqual(grid_x, grid_x_dpcpp.to(cpu_device))
        self.assertEqual(grid_y, grid_y_dpcpp.to(cpu_device))
