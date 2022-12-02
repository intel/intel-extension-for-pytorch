import unittest

import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex
from common_utils import TestCase

class MHA_Model_BERT(nn.Module):
    def __init__(self, scale, num_heads, head_dims, permute_idx, trans_a, trans_b):
        super(MHA_Model_BERT, self).__init__()
        self.scale = scale
        self.num_heads = num_heads
        self.head_dims = head_dims
        self.embed_dims = self.num_heads * self.head_dims
        self.query = nn.Linear(self.embed_dims, self.embed_dims, bias=True)
        self.key = nn.Linear(self.embed_dims, self.embed_dims, bias=True)
        self.value = nn.Linear(self.embed_dims, self.embed_dims, bias=True)
        self.permute_idx = permute_idx
        self.trans_a = trans_a
        self.trans_b = trans_b

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dims)
        x = x.view(new_x_shape)
        return x.permute(self.permute_idx)

    def forward(self, x, mask):        
        query_layer = self.transpose_for_scores(self.query(x))
        key_layer = self.transpose_for_scores(self.key(x)).transpose(self.trans_a, self.trans_b)
        value_layer = self.transpose_for_scores(self.value(x))
        attention_scores = torch.matmul(query_layer, key_layer) / self.scale + mask
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(self.permute_idx).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dims,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer

class MHA_Model_Distil(nn.Module):
    def __init__(self, scale, num_heads, head_dims, trans_a, trans_b, trans_c, fill_value=-float("inf")):
        super(MHA_Model_Distil, self).__init__()
        self.scale = scale
        self.n_head = num_heads
        self.head_dims = head_dims
        self.dim = self.n_head * self.head_dims
        self.q_lin = nn.Linear(self.dim, self.dim, bias=True)
        self.k_lin = nn.Linear(self.dim, self.dim, bias=True)
        self.v_lin = nn.Linear(self.dim, self.dim, bias=True)
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.trans_c = trans_c
        self.fill_value = fill_value

    def forward(self, x, mask):
        bs, q_length, dim = x.size()
        k_length = x.size(1)
        def shape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            return x.view(bs, -1, self.n_head, self.head_dims).transpose(self.trans_a, self.trans_b)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            """group heads"""
            return x.transpose(self.trans_a, self.trans_b).contiguous().view(bs, -1, self.n_head * self.head_dims)
        q = shape(self.q_lin(x))
        k = shape(self.k_lin(x))
        v = shape(self.v_lin(x))
        mask_reshp = (bs, 1, 1, k_length)
        q = q / self.scale
        scores = torch.matmul(q, k.transpose(self.trans_b, self.trans_c))
        mask = (mask == 0).view(mask_reshp).expand_as(scores)
        scores = scores.masked_fill(mask, self.fill_value)
        weights = nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)
        context_layer = unshape(context)

        return context_layer

class MHA_Model_ViT(nn.Module):
    def __init__(self, scale, num_heads, head_dims, permute_idx, trans_a, trans_b, select_a, select_b):
        super(MHA_Model_ViT, self).__init__() 
        self.scale = 1.0 / scale
        self.num_heads = num_heads
        self.head_dims = head_dims
        self.embed_dims = self.num_heads * self.head_dims
        self.qkv = nn.Linear(self.embed_dims, self.embed_dims * 3, bias=True)
        self.permute_idx = permute_idx
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.select_a = select_a
        self.select_b = select_b

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dims).permute(self.permute_idx)
        q, k, v = qkv[0], qkv[self.select_a], qkv[self.select_b]
        attn = (q @ k.transpose(self.trans_a, self.trans_b)) * self.scale
        attn = attn.softmax(dim=-1)
        context_layer = (attn @ v).transpose(self.select_a, self.select_b).reshape(B, N, self.embed_dims)

        return context_layer

bs = [5, 3, 11]
seq = [128, 384, 31]
scales = [8, 13, 21]
num_heads = [12, 16, 29]
head_dims = [64, 96, 17]

class TransFreeMHATester(TestCase):

    def test_transfree_mha_bf16(self):
        for i in range(len(bs)):
            mat = torch.randn(bs[i], seq[i], num_heads[i] * head_dims[i]).to(torch.bfloat16)
            mask_base = torch.randn(bs[i], 1, 1, seq[i]).to(torch.bfloat16)
            mask_distil = torch.randn(bs[i], seq[i]).to(torch.bfloat16)

            mha_model = MHA_Model_BERT(scales[i], num_heads[i], head_dims[i], [0, 2, 1, 3], -1, -2).eval()
            mha_ipex = ipex.optimize(mha_model, dtype=torch.bfloat16, level="O1")

            vit_mha_model = MHA_Model_ViT(scales[i], num_heads[i], head_dims[i], [2, 0, 3, 1, 4], -2, -1, 1, 2).eval()
            vit_mha_ipex = ipex.optimize(vit_mha_model, dtype=torch.bfloat16, level="O1")

            with torch.cpu.amp.autocast(), torch.no_grad():
                mha_ipex = torch.jit.trace(mha_ipex, (mat, mask_base, ))
                mha_ipex = torch.jit.freeze(mha_ipex)

                vit_mha_ipex = torch.jit.trace(vit_mha_ipex, (mat, ))
                vit_mha_ipex = torch.jit.freeze(vit_mha_ipex)

                for _ in range(2):
                    mha_jit = mha_ipex(mat, mask_base)
                    vit_mha_jit = vit_mha_ipex(mat)

                mha_ref = mha_model(mat, mask_base)
                vit_mha_ref = vit_mha_model(mat)

                self.assertEqual(mha_ref, mha_jit, prec=1e-2)
                self.assertEqual(vit_mha_ref, vit_mha_jit, prec=1e-2)

                mha_graph = mha_ipex.graph_for(mat, mask_base)
                vit_mha_graph = vit_mha_ipex.graph_for(mat)


                self.assertTrue(any(n.kind() == "ipex::transfree_mha" for n in mha_graph.nodes()))
                self.assertTrue(any(n.kind() == "ipex::transfree_vit_mha" for n in vit_mha_graph.nodes()))

            for fill_value in [-float("inf"), torch.tensor(torch.finfo(float).min)]:
                distil_mha_model = MHA_Model_Distil(scales[i], num_heads[i], head_dims[i], 1, 2, 3, fill_value).eval()
                distil_mha_ipex = ipex.optimize(distil_mha_model, dtype=torch.bfloat16, level="O1")

                with torch.cpu.amp.autocast(), torch.no_grad():
                    distil_mha_ipex = torch.jit.trace(distil_mha_ipex, (mat, mask_distil, ))
                    distil_mha_ipex = torch.jit.freeze(distil_mha_ipex)

                    for _ in range(2):
                        distil_mha_jit = distil_mha_ipex(mat, mask_distil)
                    distil_mha_ref = distil_mha_model(mat, mask_distil)
                    self.assertEqual(distil_mha_ref, distil_mha_jit, prec=1e-2)
                    distil_mha_graph = distil_mha_ipex.graph_for(mat, mask_distil)
                    self.assertTrue(any(n.kind() == "ipex::transfree_distil_mha" for n in distil_mha_graph.nodes()))

    def test_fake_mha_bf16(self):
        mat = torch.randn(16, 16, 256).to(torch.bfloat16)
        mask_base = torch.randn(16, 1, 1, 16).to(torch.bfloat16)
        mask_distil = torch.randn(16, 16).to(torch.bfloat16)

        fake_mha_model = []
        fake_mha_ipex = []

        fake_mha_model.append(MHA_Model_BERT(16, 16, 16, [0, 2, 3, 1], -1, -2).eval())
        fake_mha_model.append(MHA_Model_BERT(16, 16, 16, [0, 2, 1, 3], -2, -3).eval())
        fake_mha_ipex.append(ipex.optimize(fake_mha_model[0], dtype=torch.bfloat16, level="O1"))
        fake_mha_ipex.append(ipex.optimize(fake_mha_model[1], dtype=torch.bfloat16, level="O1"))

        fake_mha_model.append(MHA_Model_Distil(16, 16, 16, 1, 2, 1).eval())
        fake_mha_model.append(MHA_Model_Distil(16, 16, 16, 2, 1, 3).eval())
        fake_mha_ipex.append(ipex.optimize(fake_mha_model[2], dtype=torch.bfloat16, level="O1"))
        fake_mha_ipex.append(ipex.optimize(fake_mha_model[3], dtype=torch.bfloat16, level="O1"))

        fake_mha_model.append(MHA_Model_ViT(16, 16, 16, [2, 0, 1, 3, 4], -2, -1, 1, 2).eval())
        fake_mha_model.append(MHA_Model_ViT(16, 16, 16, [2, 0, 3, 1, 4], -2, -3, 1, 2).eval())
        fake_mha_model.append(MHA_Model_ViT(16, 16, 16, [2, 0, 3, 1, 4], -2, -1, 0, 2).eval())
        fake_mha_ipex.append(ipex.optimize(fake_mha_model[4], dtype=torch.bfloat16, level="O1"))
        fake_mha_ipex.append(ipex.optimize(fake_mha_model[5], dtype=torch.bfloat16, level="O1"))
        fake_mha_ipex.append(ipex.optimize(fake_mha_model[6], dtype=torch.bfloat16, level="O1"))

        with torch.cpu.amp.autocast(), torch.no_grad():
            fake_mha_jit = []
            fake_mha_ref = []

            for i in range(0, 2):
                fake_mha_ipex[i] = torch.jit.trace(fake_mha_ipex[i], (mat, mask_base, ))
                fake_mha_ipex[i] = torch.jit.freeze(fake_mha_ipex[i])
                for _ in range(2):
                    fake_mha_ipex[i](mat, mask_base)
                fake_mha_jit.append(fake_mha_ipex[i](mat, mask_base))
                fake_mha_ref.append(fake_mha_model[i](mat, mask_base))
                fake_mha_graph = fake_mha_ipex[i].graph_for(mat, mask_base)
                self.assertFalse(any(n.kind() == "ipex::transfree_mha" for n in fake_mha_graph.nodes()))
            
            for i in range(2, 4):
                fake_mha_ipex[i] = torch.jit.trace(fake_mha_ipex[i], (mat, mask_distil, ))
                fake_mha_ipex[i] = torch.jit.freeze(fake_mha_ipex[i])
                for _ in range(2):
                    fake_mha_ipex[i](mat, mask_distil)
                fake_mha_jit.append(fake_mha_ipex[i](mat, mask_distil))
                fake_mha_ref.append(fake_mha_model[i](mat, mask_distil))
                fake_mha_graph = fake_mha_ipex[i].graph_for(mat, mask_distil)
                self.assertFalse(any(n.kind() == "ipex::transfree_distil_mha" for n in fake_mha_graph.nodes()))

            for i in range(4, 7):
                fake_mha_ipex[i] = torch.jit.trace(fake_mha_ipex[i], mat)
                fake_mha_ipex[i] = torch.jit.freeze(fake_mha_ipex[i])
                for _ in range(2):
                    fake_mha_ipex[i](mat)
                fake_mha_jit.append(fake_mha_ipex[i](mat))
                fake_mha_ref.append(fake_mha_model[i](mat))
                fake_mha_graph = fake_mha_ipex[i].graph_for(mat)
                self.assertFalse(any(n.kind() == "ipex::transfree_vit_mha" for n in fake_mha_graph.nodes()))

            for i in range(7):
                self.assertEqual(fake_mha_ref[i], fake_mha_jit[i], prec=1e-2)

    def test_transfree_mha_fp32(self):
        for i in range(len(bs)):
            mat = torch.randn(bs[i], seq[i], num_heads[i] * head_dims[i]).to(torch.float)
            mask_base = torch.randn(bs[i], 1, 1, seq[i]).to(torch.float)
            mask_distil = torch.randn(bs[i], seq[i]).to(torch.float)

            mha_model = MHA_Model_BERT(scales[i], num_heads[i], head_dims[i], [0, 2, 1, 3], -1, -2).eval()
            mha_ipex = ipex.optimize(mha_model, dtype=torch.float, level="O1")

            distil_mha_model = MHA_Model_Distil(scales[i], num_heads[i], head_dims[i], 1, 2, 3).eval()
            distil_mha_ipex = ipex.optimize(distil_mha_model, dtype=torch.float, level="O1")

            vit_mha_model = MHA_Model_ViT(scales[i], num_heads[i], head_dims[i], [2, 0, 3, 1, 4], -2, -1, 1, 2).eval()
            vit_mha_ipex = ipex.optimize(vit_mha_model, dtype=torch.float, level="O1")

            with torch.no_grad():
                mha_ipex = torch.jit.trace(mha_ipex, (mat, mask_base, ))
                mha_ipex = torch.jit.freeze(mha_ipex)

                distil_mha_ipex = torch.jit.trace(distil_mha_ipex, (mat, mask_distil, ))
                distil_mha_ipex = torch.jit.freeze(distil_mha_ipex)

                vit_mha_ipex = torch.jit.trace(vit_mha_ipex, (mat, ))
                vit_mha_ipex = torch.jit.freeze(vit_mha_ipex)

                for _ in range(2):
                    mha_jit = mha_ipex(mat, mask_base)
                    distil_mha_jit = distil_mha_ipex(mat, mask_distil)
                    vit_mha_jit = vit_mha_ipex(mat)
                
                mha_ref = mha_model(mat, mask_base)
                distil_mha_ref = distil_mha_model(mat, mask_distil)
                vit_mha_ref = vit_mha_model(mat)

                self.assertEqual(mha_ref, mha_jit, prec=1e-5)
                self.assertEqual(distil_mha_ref, distil_mha_jit, prec=1e-5)
                self.assertEqual(vit_mha_ref, vit_mha_jit, prec=1e-5)

                mha_graph = mha_ipex.graph_for(mat, mask_base)
                distil_mha_graph = distil_mha_ipex.graph_for(mat, mask_distil)
                vit_mha_graph = vit_mha_ipex.graph_for(mat)

                self.assertTrue(any(n.kind() == "ipex::matmul_outtrans" for n in mha_graph.nodes()))
                self.assertTrue(any(n.kind() == "ipex::matmul_outtrans" for n in distil_mha_graph.nodes()))
                self.assertTrue(any(n.kind() == "ipex::matmul_outtrans" for n in vit_mha_graph.nodes()))
                
    def test_fake_mha_fp32(self):
        mat = torch.randn(16, 16, 256)
        mask_base = torch.randn(16, 1, 1, 16)
        mask_distil = torch.randn(16, 16)

        fake_mha_model = []
        fake_mha_ipex = []

        fake_mha_model.append(MHA_Model_BERT(16, 16, 16, [0, 2, 3, 1], -1, -2).eval())
        fake_mha_model.append(MHA_Model_BERT(16, 16, 16, [0, 2, 1, 3], -2, -3).eval())
        fake_mha_ipex.append(ipex.optimize(fake_mha_model[0], dtype=torch.float, level="O1"))
        fake_mha_ipex.append(ipex.optimize(fake_mha_model[1], dtype=torch.float, level="O1"))

        fake_mha_model.append(MHA_Model_Distil(16, 16, 16, 1, 2, 1).eval())
        fake_mha_model.append(MHA_Model_Distil(16, 16, 16, 2, 1, 3).eval())
        fake_mha_ipex.append(ipex.optimize(fake_mha_model[2], dtype=torch.float, level="O1"))
        fake_mha_ipex.append(ipex.optimize(fake_mha_model[3], dtype=torch.float, level="O1"))

        fake_mha_model.append(MHA_Model_ViT(16, 16, 16, [2, 0, 1, 3, 4], -2, -1, 1, 2).eval())
        fake_mha_model.append(MHA_Model_ViT(16, 16, 16, [2, 0, 3, 1, 4], -2, -3, 1, 2).eval())
        fake_mha_model.append(MHA_Model_ViT(16, 16, 16, [2, 0, 3, 1, 4], -2, -1, 0, 2).eval())
        fake_mha_ipex.append(ipex.optimize(fake_mha_model[4], dtype=torch.float, level="O1"))
        fake_mha_ipex.append(ipex.optimize(fake_mha_model[5], dtype=torch.float, level="O1"))
        fake_mha_ipex.append(ipex.optimize(fake_mha_model[6], dtype=torch.float, level="O1"))

        with torch.no_grad():
            fake_mha_jit = []
            fake_mha_ref = []

            for i in range(0, 2):
                fake_mha_ipex[i] = torch.jit.trace(fake_mha_ipex[i], (mat, mask_base, ))
                fake_mha_ipex[i] = torch.jit.freeze(fake_mha_ipex[i])
                for _ in range(2):
                    fake_mha_ipex[i](mat, mask_base)
                fake_mha_jit.append(fake_mha_ipex[i](mat, mask_base))
                fake_mha_ref.append(fake_mha_model[i](mat, mask_base))
                fake_mha_graph = fake_mha_ipex[i].graph_for(mat, mask_base)
                self.assertTrue(any(n.kind() == "ipex::mha_scores_calc" for n in fake_mha_graph.nodes()))
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
                    fake_mha_ipex[i](mat, mask_base)
                if i == 0:
                    self.assertTrue("dil_matmul" in str(p.key_averages()))
                else:
                    self.assertTrue("dil_mha_bmm" in str(p.key_averages()))
            
            for i in range(2, 4):
                fake_mha_ipex[i] = torch.jit.trace(fake_mha_ipex[i], (mat, mask_distil, ))
                fake_mha_ipex[i] = torch.jit.freeze(fake_mha_ipex[i])
                for _ in range(2):
                    fake_mha_ipex[i](mat, mask_distil)
                fake_mha_jit.append(fake_mha_ipex[i](mat, mask_distil))
                fake_mha_ref.append(fake_mha_model[i](mat, mask_distil))
                fake_mha_graph = fake_mha_ipex[i].graph_for(mat, mask_distil)
                self.assertTrue(any(n.kind() == "ipex::distil_mha_scores_calc" for n in fake_mha_graph.nodes()))
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
                    fake_mha_ipex[i](mat, mask_distil)
                if i == 2:
                    self.assertTrue("dil_mha_bmm" in str(p.key_averages()))
                else:
                    self.assertTrue("dil_matmul" in str(p.key_averages()))

            for i in range(4, 7):
                fake_mha_ipex[i] = torch.jit.trace(fake_mha_ipex[i], mat)
                fake_mha_ipex[i] = torch.jit.freeze(fake_mha_ipex[i])
                for _ in range(2):
                    fake_mha_ipex[i](mat)
                fake_mha_jit.append(fake_mha_ipex[i](mat))
                fake_mha_ref.append(fake_mha_model[i](mat))
                fake_mha_graph = fake_mha_ipex[i].graph_for(mat)
                self.assertTrue(any(n.kind() == "ipex::matmul_mul" for n in fake_mha_graph.nodes()))
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
                    fake_mha_ipex[i](mat)
                if i == 6:
                    self.assertTrue("dil_matmul" in str(p.key_averages()))
                else:
                    self.assertTrue("dil_mha_bmm" in str(p.key_averages()))

            for i in range(7):
                self.assertEqual(fake_mha_ref[i], fake_mha_jit[i], prec=1e-5)

if __name__ == '__main__':
    test = unittest.main()
