import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_coo_to_csr_convert_nocoalesced(self, dtype=torch.float):
        # ----------------------cpu--------------------
        indices = torch.tensor([
            [0, 1, 6, 14, 27, 35, 39, 40, 66, 71],
            [31, 92, 65, 50, 34, 62, 22, 56, 74, 89],
        ], dtype=torch.int32)
        values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dtype, device=cpu_device)
        sparse_coo = torch.sparse_coo_tensor(indices, values, torch.Size([100, 100]), dtype=dtype, device=cpu_device)

        row_indices_coo = indices[0]
        row_indices_csr = torch._convert_indices_from_coo_to_csr(
            row_indices_coo, sparse_coo.shape[0], out_int32=row_indices_coo.dtype == torch.int32)

        # ----------------------xpu--------------------
        row_indices_coo_xpu = row_indices_coo.to(dpcpp_device)
        row_indices_csr_xpu = torch._convert_indices_from_coo_to_csr(
            row_indices_coo_xpu, sparse_coo.shape[0], out_int32=row_indices_coo_xpu.dtype == torch.int32)

        self.assertEqual(row_indices_csr, row_indices_csr_xpu.to(cpu_device))

    def test_coo_to_csr_convert_coalesced(self, dtype=torch.float):
        # ----------------------cpu--------------------
        indices = torch.tensor([
            [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ], dtype=torch.int32)
        values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dtype, device=cpu_device)
        sparse_coo = torch.sparse_coo_tensor(indices, values, torch.Size([100, 100]), dtype=dtype, device=cpu_device)

        # coalesce process will accumulate the multi-valued elements into a single value using summation.
        coalesced_coo = sparse_coo.coalesce()
        row_indices_coo = coalesced_coo.indices()[0]

        row_indices_csr = torch._convert_indices_from_coo_to_csr(
            row_indices_coo, sparse_coo.shape[0], out_int32=row_indices_coo.dtype == torch.int32)

        # ----------------------xpu--------------------
        row_indices_coo_xpu = row_indices_coo.to(dpcpp_device)
        row_indices_csr_xpu = torch._convert_indices_from_coo_to_csr(
            row_indices_coo_xpu, sparse_coo.shape[0], out_int32=row_indices_coo_xpu.dtype == torch.int32)

        self.assertEqual(row_indices_csr, row_indices_csr_xpu.to(cpu_device))
