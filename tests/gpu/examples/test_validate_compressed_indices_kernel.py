import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

import numpy as np

np.set_printoptions(threshold=np.inf)

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_validate_compressed_indices_kernel(self, dtype=torch.float):
        x = torch.tensor([0, 2, 4])
        b = torch.tensor([0, 1, 0, 1])
        torch.ops.aten._validate_compressed_sparse_indices(True, x, b, 2, 3, 4)
        torch.ops.aten._validate_compressed_sparse_indices(False, x, b, 2, 3, 4)

        x_dpcpp = x.to(dpcpp_device)
        b_dpcpp = b.to(dpcpp_device)
        torch.ops.aten._validate_compressed_sparse_indices(
            False, x_dpcpp, b_dpcpp, 2, 3, 4
        )
        self.assertEqual(x, x_dpcpp.to(cpu_device))
        self.assertEqual(b, b_dpcpp.to(cpu_device))
