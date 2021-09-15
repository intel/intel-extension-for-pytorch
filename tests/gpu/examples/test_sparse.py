import torch
from torch.testing._internal.common_utils import TestCase

import ipex


class TestTorchMethod(TestCase):
    def test_sparse(self, dtype=torch.float):
        i_cpu = torch.LongTensor([[0, 1, 1], [2, 0, 0]])
        v_cpu = torch.FloatTensor([3, 4, 5])
        r_cpu = torch.torch._sparse_coo_tensor_unsafe(i_cpu, v_cpu, torch.Size([2, 3]))
        dense_dim_cpu = r_cpu.dense_dim()
        sparse_dim_cpu = r_cpu.sparse_dim()
        indices_cpu = r_cpu._indices()
        value_cpu = r_cpu._values()
        coalesce_cpu = r_cpu.coalesce()
        print(coalesce_cpu)

        i_xpu = i_cpu.to("xpu")
        v_xpu = v_cpu.to("xpu")
        r_xpu = torch.torch._sparse_coo_tensor_unsafe(i_xpu, v_xpu, torch.Size([2, 3]))
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
