import unittest
import copy

import torch
import intel_pytorch_extension as ipex
import torch.nn as nn
from common_utils import TestCase

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

if __name__ == '__main__':
    test = unittest.main()
