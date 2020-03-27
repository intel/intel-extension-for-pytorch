import unittest
import copy

import torch
import intel_pytorch_extension as ipex
import torch.nn as nn
from common_utils import TestCase
from numbers import Number

from hypothesis import given
from hypothesis import strategies as st

class TestEMB(TestCase):
    @given(sparse=st.sampled_from((True, False)))
    def test_embedding_bag(self, sparse):
        # Forward testing
        cpu_emb = nn.EmbeddingBag(10, 3, mode='sum', sparse=sparse)
        cpu_input = torch.LongTensor([1,2,4,5,4,3,2,9])
        cpu_offsets = torch.LongTensor([0,4])
        cpu_output = cpu_emb(cpu_input, cpu_offsets)

        dpcpp_emb = copy.deepcopy(cpu_emb).to('dpcpp:0')
        dpcpp_input = cpu_input.clone().detach().to('dpcpp:0')
        dpcpp_offsets = cpu_offsets.clone().detach().to('dpcpp:0')
        dpcpp_output = dpcpp_emb(dpcpp_input, dpcpp_offsets)

        self.assertEqual(cpu_output, dpcpp_output.to('cpu'))

        # Backward testing
        cpu_gy = torch.rand(2, 3, device='cpu')
        dpcpp_gy = cpu_gy.clone().detach().to('dpcpp:0')
        cpu_output.backward(cpu_gy)
        dpcpp_output.backward(dpcpp_gy)

        if sparse:
            self.assertEqual(cpu_emb.weight.grad.data._nnz(), dpcpp_emb.weight.grad.data._nnz())
            self.assertEqual(cpu_emb.weight.grad.data.sparse_dim(), dpcpp_emb.weight.grad.data.sparse_dim())
            self.assertEqual(cpu_emb.weight.grad.data.dense_dim(), dpcpp_emb.weight.grad.data.dense_dim())
            self.assertEqual(cpu_emb.weight.grad.data.is_coalesced(), dpcpp_emb.weight.grad.data.is_coalesced())
            self.assertEqual(cpu_emb.weight.grad.data._indices(), dpcpp_emb.weight.grad.data._indices().to('cpu'))
            self.assertEqual(cpu_emb.weight.grad.data._values(), dpcpp_emb.weight.grad.data._values().to('cpu'))
        else:
            self.assertEqual(cpu_emb.weight.grad.data, dpcpp_emb.weight.grad.data.to('cpu'))

class TestSparse(TestCase):
    def genSparseTensor(self, size, sparse_dim, nnz, device='cpu'):
        assert all(size[d] > 0 for d in range(sparse_dim)) or nnz == 0, 'invalid arguments'

        v_size = [nnz] + list(size[sparse_dim:])
        v = torch.randn(*v_size, device=device)
        i = torch.rand(sparse_dim, nnz, device=device)
        i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
        i = i.to(torch.long)

        x = torch.sparse_coo_tensor(i, v, torch.Size(size))

        x = x.detach().clone()
        return x, x._indices().clone(), x._values().clone()

    def _gen_sparse(self, sparse_dim, nnz, with_size):
        if isinstance(with_size, Number):
            with_size = [with_size] * sparse_dim

        return self.genSparseTensor(with_size, sparse_dim, nnz)

    def _test_basic_ops_shape(self, nnz_x1, nnz_x2, shape):
        x1, _, _ = self._gen_sparse(len(shape), nnz_x1, shape)
        x2, _, _ = self._gen_sparse(len(shape), nnz_x2, shape)

        y1 = x1.clone()
        y1.add_(x2)

        y2 = x1.clone().to('dpcpp:0')
        y2.add_(x2.to('dpcpp:0'))

        y3 = x1.clone().to_dense().to('dpcpp:0')
        y3.add_(x2.to('dpcpp:0'))

        expected = x1.to_dense() + x2.to_dense()
        self.assertEqual(y1.to_dense(), expected)
        # self.assertEqual(y2.to_dense().to('cpu'), expected)
        self.assertEqual(y3.to('cpu'), expected)

        z1 = x1 + x2
        z1 = z1.zero_()
        z2 = y2.zero_()
        self.assertEqual(z2.to('cpu'), z1)

        
    def test_basic_ops(self):
        self._test_basic_ops_shape(9, 12, [5, 6])
        self._test_basic_ops_shape(9, 12, [10, 10, 10])
        self._test_basic_ops_shape(9, 12, [50, 30, 20])
        self._test_basic_ops_shape(9, 12, [5, 5, 5, 5, 5, 5])
        self._test_basic_ops_shape(0, 12, [10, 10, 10])
        self._test_basic_ops_shape(9, 0, [10, 10, 10])
        self._test_basic_ops_shape(0, 0, [10, 10, 10])
        self._test_basic_ops_shape(0, 0, [10, 10, 0])


if __name__ == '__main__':
    test = unittest.main()
