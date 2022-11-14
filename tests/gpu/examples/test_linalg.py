import torch
import intel_extension_for_pytorch # noqa
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_tensorinv_empty(self, device=dpcpp_device, dtype=torch.float64):
        for ind in range(1, 4):
            # Check for empty inputs. NumPy does not work for these cases.
            a = torch.empty(0, 0, 1, 2, 3, 0, dtype=dtype, device=device)
            a_inv = torch.linalg.tensorinv(a, ind=ind)
            self.assertEqual(a_inv.shape, a.shape[ind:] + a.shape[:ind])
