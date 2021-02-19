import torch
import torch.nn as nn
import intel_pytorch_extension as ipex
import unittest
import copy
from common_utils import TestCase

class TestEMB(TestCase):
    def test_emb(self):
        #E = nn.EmbeddingBag(10, 5, mode="sum", sparse=True)
        cpu_emb = nn.EmbeddingBag(10, 3, mode='sum', sparse=True)
        dpcpp_emb = copy.deepcopy(cpu_emb).to(ipex.DEVICE)
        bf16_emb = copy.deepcopy(cpu_emb).to(ipex.DEVICE).bfloat16()
        # a batch of 2 samples of 4 indices each
        cpu_input = torch.LongTensor([1,2,4,5,4,3,2,9])
        dpcpp_input = cpu_input.clone().detach().to(ipex.DEVICE)

        cpu_offsets = torch.LongTensor([0,1,2,3,4,5,6,7])
        dpcpp_offsets = cpu_offsets.clone().detach().to(ipex.DEVICE)

        cpu_out = cpu_emb(cpu_input, cpu_offsets)

        #torch.embedding_bag = ipex.embeddingbag
        dpcpp_out = dpcpp_emb(dpcpp_input, dpcpp_offsets)
        bf16_out = bf16_emb(dpcpp_input, dpcpp_offsets)

        self.assertEqual(cpu_out, dpcpp_out.to('cpu'))
        self.assertEqual(cpu_out, bf16_out.to('cpu').float(), 0.01)

        cpu_out.mean().backward()
        dpcpp_out.mean().backward()
        bf16_out.float().mean().backward()

        self.assertEqual(cpu_emb.weight.grad.data._nnz(), dpcpp_emb.weight.grad.data._nnz())
        self.assertEqual(cpu_emb.weight.grad.data.sparse_dim(), dpcpp_emb.weight.grad.data.sparse_dim())
        self.assertEqual(cpu_emb.weight.grad.data.dense_dim(), dpcpp_emb.weight.grad.data.dense_dim())
        self.assertEqual(cpu_emb.weight.grad.data.is_coalesced(), dpcpp_emb.weight.grad.data.is_coalesced())
        self.assertEqual(cpu_emb.weight.grad.data._indices(), dpcpp_emb.weight.grad.data._indices().to('cpu'))
        self.assertEqual(cpu_emb.weight.grad.data._values(), dpcpp_emb.weight.grad.data._values().to('cpu'))

        self.assertEqual(cpu_emb.weight.grad.data._values(), dpcpp_emb.weight.grad.data._values().to('cpu'), 0.01)
        self.assertEqual(bf16_emb.weight.grad.data._values().dtype, torch.bfloat16)

if __name__ == '__main__':
    test = unittest.main()
