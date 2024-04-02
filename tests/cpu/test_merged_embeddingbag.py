import torch
import unittest
from torch.testing._internal.common_utils import TestCase
from bench.custom_op_bench.merged_embeddingbag import (
    EmbeddingBagList,
    MergedEmb,
    EmbeddingBagListCatDense,
    MergedEmbCatDense,
    MergedEmbSGD,
    MergedEmbAdaGrad,
)
import intel_extension_for_pytorch as ipex
import copy


class TestMergedEmbedding(TestCase):
    multi_hot = [
        3,
        2,
        1,
        2,
        6,
        1,
        1,
        1,
        1,
        7,
        3,
        8,
        1,
        6,
        9,
        5,
        1,
        1,
        1,
        12,
        100,
        27,
        10,
        3,
        1,
        1,
    ]

    def _test_autocast(self, m: torch.nn.Module, refm: torch.nn.Module, inputs: tuple):
        m.eval()
        refm.eval()
        with torch.no_grad(), torch.cpu.amp.autocast():
            out = m.forward(*inputs)
            ref_out = refm.forward(*inputs)
        self.assertTrue(all(o.dtype is torch.bfloat16 for o in out))
        # refm(embedding_bag) may not support autocast (while fallback)
        self.assertEqual(
            [o.float() for o in out], [o.float() for o in ref_out], atol=0.1, rtol=0.1
        )

    def _test_inference(self, m: torch.nn.Module, refm: torch.nn.Module, inputs: tuple):
        m.eval()
        refm.eval()
        with torch.no_grad():
            out = m.forward(*inputs)
            ref_out = refm.forward(*inputs)
        self.assertEqual(out, ref_out)

        # check module tracable and can get correct result with traced module
        with torch.no_grad():
            jit_m = torch.jit.trace(m, inputs)
            jit_m = torch.jit.freeze(jit_m)
            jit_m(*inputs)
            jit_m(*inputs)
            jit_out = jit_m(*inputs)
        self.assertEqual(jit_out, ref_out)

    def _test_training(
        self, m: torch.nn.Module, refm: torch.nn.Module, inputs: tuple, opt=None
    ):
        m.train()
        refm.train()
        fused_update_test = False if opt is None else True
        if fused_update_test:
            opt.zero_grad()
        out = m(*inputs)
        ref_out = refm(*inputs)
        self.assertEqual(out, ref_out)
        sum(out).sum().backward()
        sum(ref_out).sum().backward()
        if fused_update_test:
            opt.step()
        if m.merged_emb.weights[0].dtype in (torch.float16, torch.bfloat16):
            rtol, atol = 0.1, 0.1
            """
            Mismatched elements: 3328 / 128000 (2.6%)
            Greatest absolute difference: 0.03125 at index (378, 0) (up to 0.016 allowed)
            Greatest relative difference: 0.01275634765625 at index (297, 0) (up to 1e-05 allowed)

            The aten's embedding bag do not use float as accumulate type for grad in backward
            """
        else:
            rtol, atol = 1e-5, 0.016
        for i in range(len(out)):
            if fused_update_test:
                self.assertEqual(
                    m.merged_emb.weights[i], refm.list[i].weight, rtol=rtol, atol=atol
                )
            else:
                self.assertEqual(
                    m.merged_emb.weights[i].grad,
                    refm.list[i].weight.grad,
                    rtol=rtol,
                    atol=atol,
                )

    def test_inference(self):
        B = 1029
        NUM_TABLE = 26
        for mode in ["mean", "sum"]:
            for index_type in [torch.int32, torch.int64]:
                indices = [
                    torch.randint(1000, (B * self.multi_hot[i],)).to(index_type)
                    for i in range(NUM_TABLE)
                ]
                for include_last_offset in [True, False]:
                    n_offset = B + 1 if include_last_offset else B
                    offsets = [
                        torch.arange(
                            0, n_offset * self.multi_hot[i], self.multi_hot[i]
                        ).to(index_type)
                        for i in range(NUM_TABLE)
                    ]
                    for dtype in [
                        torch.float64,
                        torch.float32,
                        torch.bfloat16,
                        torch.float16,
                    ]:
                        for NUM_DIM in [128, 129]:
                            # 128 for fast path, 129 for general path
                            emb_list = EmbeddingBagList(
                                NUM_TABLE,
                                NUM_DIM,
                                dtype,
                                include_last_offset=include_last_offset,
                                mode=mode,
                            )
                            # test merged emb
                            m = MergedEmb(emb_list)
                            ref_m = copy.deepcopy(emb_list)
                            self._test_inference(m, ref_m, (indices, offsets))

                            if dtype == torch.float:
                                self._test_autocast(m, ref_m, (indices, offsets))

                            # test merged emb + cat
                            if mode == "mean":
                                # TODO: Support mean for "cat"
                                continue
                            ref_m = EmbeddingBagListCatDense(emb_list)
                            m = MergedEmbCatDense(emb_list)
                            dense = torch.randn(B, NUM_DIM, dtype=dtype)
                            self._test_inference(m, ref_m, (indices, offsets, dense))

    def test_training(self):
        B = 1029
        NUM_TABLE = 26
        for mode in ["mean", "sum"]:
            for index_type in [torch.int64, torch.int32]:
                indices = [
                    torch.randint(1000, (B * self.multi_hot[i],)).to(index_type)
                    for i in range(NUM_TABLE)
                ]
                for include_last_offset in [True, False]:
                    n_offset = B + 1 if include_last_offset else B
                    offsets = [
                        torch.arange(
                            0, n_offset * self.multi_hot[i], self.multi_hot[i]
                        ).to(index_type)
                        for i in range(NUM_TABLE)
                    ]
                    for dtype in [
                        torch.bfloat16,
                        torch.float32,
                        torch.float16,
                        torch.float64,
                    ]:
                        for NUM_DIM in [128, 129]:
                            # 128 for fast path, 129 for general path
                            # test merged emb return dense grad
                            emb_list = EmbeddingBagList(
                                NUM_TABLE,
                                NUM_DIM,
                                dtype,
                                include_last_offset=include_last_offset,
                                mode=mode,
                            )
                            m = MergedEmb(copy.deepcopy(emb_list))
                            ref_m = copy.deepcopy(emb_list)
                            self._test_training(m, ref_m, (indices, offsets))

                            # test merged emb fused update with sgd
                            if dtype in (torch.float16,):
                                # do not support fp16
                                continue
                            if dtype == torch.bfloat16:
                                # for bf16, only support split sgd
                                emb_list = EmbeddingBagList(
                                    NUM_TABLE,
                                    NUM_DIM,
                                    torch.float32,
                                    include_last_offset=include_last_offset,
                                )
                            m = MergedEmbSGD(copy.deepcopy(emb_list), lr=0.1)
                            ref_m = copy.deepcopy(emb_list)
                            opt = torch.optim.SGD(ref_m.parameters(), lr=0.1)
                            if dtype == torch.bfloat16:
                                m.merged_emb.to_bfloat16_train()
                                ref_m, opt = ipex.optimize(
                                    ref_m, dtype=torch.bfloat16, optimizer=opt
                                )
                            self._test_training(m, ref_m, (indices, offsets), opt=opt)

                            # test merged emb fused update with adagrad
                            m = MergedEmbAdaGrad(copy.deepcopy(emb_list), lr=0.01)
                            ref_m = copy.deepcopy(emb_list)
                            opt = torch.optim.Adagrad(ref_m.parameters(), lr=0.01)
                            if dtype == torch.bfloat16:
                                m.merged_emb.to_bfloat16_train()
                                ref_m, opt = ipex.optimize(
                                    ref_m, dtype=torch.bfloat16, optimizer=opt
                                )
                            self._test_training(m, ref_m, (indices, offsets), opt=opt)


if __name__ == "__main__":
    test = unittest.main()
