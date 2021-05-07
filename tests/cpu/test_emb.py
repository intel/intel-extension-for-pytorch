import torch
import torch.nn as nn
import unittest
import copy
from common_utils import TestCase

class TestEMB(TestCase):
    def _test_emb(self, mode):
        #E = nn.EmbeddingBag(10, 5, mode="sum", sparse=True)
        aten_emb = nn.EmbeddingBag(10, 3, mode=mode, sparse=True)
        ipex_emb = copy.deepcopy(aten_emb)
        bf16_emb = copy.deepcopy(aten_emb).bfloat16()
        # a batch of 2 samples of 4 indices each
        input = torch.LongTensor([1,2,4,5,4,3,2,9])
        offsets = torch.LongTensor([0,1,2,3,4,5,6,7])
        # aten path
        aten_out = aten_emb(input, offsets)
        aten_out.mean().backward()

        # ipex fast path (both fp32/bf16)
        import intel_pytorch_extension
        ipex_out = ipex_emb(input, offsets)
        ipex_out.mean().backward()
        if mode == 'sum':
            bf16_out = bf16_emb(input, offsets)
            bf16_out.mean().backward()
            self.assertEqual(aten_out, bf16_out.float(), 0.01)
            self.assertEqual(bf16_emb.weight.grad.data._values().dtype, torch.bfloat16)
        del(intel_pytorch_extension)

        self.assertEqual(aten_out, ipex_out)

        self.assertEqual(aten_emb.weight.grad.data._nnz(), ipex_emb.weight.grad.data._nnz())
        self.assertEqual(aten_emb.weight.grad.data.sparse_dim(), ipex_emb.weight.grad.data.sparse_dim())
        self.assertEqual(aten_emb.weight.grad.data.dense_dim(), ipex_emb.weight.grad.data.dense_dim())
        self.assertEqual(aten_emb.weight.grad.data.is_coalesced(), ipex_emb.weight.grad.data.is_coalesced())
        self.assertEqual(aten_emb.weight.grad.data._indices(), ipex_emb.weight.grad.data._indices())
        self.assertEqual(aten_emb.weight.grad.data._values(), ipex_emb.weight.grad.data._values())
        self.assertEqual(aten_emb.weight.grad.data._values(), ipex_emb.weight.grad.data._values(), 0.01)

    def test_emb_fast_path(self):
        self._test_emb(mode='mean')

    def test_emb_fallback_path(self):
        self._test_emb(mode='sum')

if __name__ == '__main__':
    test = unittest.main()
