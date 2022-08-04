import torch
from torch.distributions import Geometric
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa


cpu_device = torch.device("cpu")
sycl_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_geometric(self):
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True, device=sycl_device)
        r = torch.tensor(0.3, requires_grad=True, device=sycl_device)
        self.assertEqual(Geometric(p).sample((8,)).size(), (8, 3))
        self.assertFalse(Geometric(p).sample().requires_grad)
        self.assertEqual(Geometric(r).sample((8,)).size(), (8,))
        self.assertEqual(Geometric(r).sample().size(), ())
        self.assertEqual(Geometric(r).sample((3, 2)).size(), (3, 2))
        self.assertRaises(NotImplementedError, Geometric(r).rsample)
