import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_sparse(self, dtype=torch.float):
        i_cpu = torch.LongTensor([[0, 1, 1], [2, 0, 0]])
        v_cpu = torch.FloatTensor([3, 4, 5])
        r_cpu = torch.sparse_coo_tensor(i_cpu, v_cpu, torch.Size([2, 3]))
        dense_dim_cpu = r_cpu.dense_dim()
        sparse_dim_cpu = r_cpu.sparse_dim()
        indices_cpu = r_cpu._indices()
        value_cpu = r_cpu._values()
        coalesce_cpu = r_cpu.coalesce()
        print(coalesce_cpu)

        i_xpu = i_cpu.to("xpu")
        v_xpu = v_cpu.to("xpu")
        r_xpu = torch.sparse_coo_tensor(i_xpu, v_xpu, torch.Size([2, 3]))
        dense_dim_xpu = r_xpu.dense_dim()
        sparse_dim_xpu = r_xpu.sparse_dim()
        indices_xpu = r_xpu._indices()
        value_xpu = r_xpu._values()
        coalesce_xpu = r_xpu.coalesce()
        print(coalesce_xpu.cpu())
        self.assertEqual(r_cpu, r_xpu.cpu())
        self.assertEqual(dense_dim_cpu, dense_dim_xpu)
        self.assertEqual(sparse_dim_cpu, sparse_dim_xpu)
        self.assertEqual(indices_cpu, indices_xpu)
        self.assertEqual(value_cpu, value_xpu)
        # assertEqual needs to_dense op support
        # self.assertEqual(coalesce_cpu, coalesce_xpu)

        sizes = [(10, 10), (10, 1), (4, 5, 6)]
        for size in sizes:
            x = torch.rand(size)
            y = torch.zeros(size)
            src_cpu = torch.where(x > 0.8, x, y)
            src_xpu = src_cpu.clone().to("xpu")
            self.assertEqual(src_cpu.to_sparse(), src_xpu.to_sparse().to("cpu"))
            self.assertEqual(
                src_cpu.to_sparse().sparse_dim(), src_xpu.to_sparse().sparse_dim()
            )

    def test_efficientzerotensor(self, dtype=torch.float):
        sizes = [(10, 10), (10, 1), (4, 5, 6)]
        for size in sizes:
            x_cpu = torch._efficientzerotensor(size)
            x_xpu = torch._efficientzerotensor(size, device=dpcpp_device)
            self.assertEqual(x_cpu, x_xpu.cpu())
            self.assertEqual(x_cpu.dim(), x_xpu.dim())

    def test_sparse_dense_convert(self):
        i = torch.LongTensor([[2, 4]])
        v = torch.FloatTensor([[1, 3], [5, 7]])
        x = torch.sparse.FloatTensor(i, v).to_dense()
        i_xpu = i.to("xpu")
        v_xpu = v.to("xpu")
        x_xpu = torch.sparse.FloatTensor(i_xpu, v_xpu).to_dense()
        self.assertEqual(x, x_xpu.cpu())
