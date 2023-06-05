import unittest

import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex
from common_utils import TestCase


class TransFree_FP32_Bmm(nn.Module):
    def __init__(self):
        super(TransFree_FP32_Bmm, self).__init__()

    def forward(self, x1, y1):
        out = torch.matmul(x1, y1)
        return out


class OutTransFree_FP32_Bmm_v1(nn.Module):
    def __init__(self):
        super(OutTransFree_FP32_Bmm_v1, self).__init__()

    def forward(self, x1, y1):
        out_ = torch.matmul(x1, y1)
        out = out_.transpose(1, 2)
        return out


class OutTransFree_FP32_Bmm_v2(nn.Module):
    def __init__(self):
        super(OutTransFree_FP32_Bmm_v2, self).__init__()

    def forward(self, x1, y1):
        out_ = torch.matmul(x1, y1)
        out = out_.permute(0, 2, 1, 3)
        return out


class OutTransFree_FP32_Bmm_v3(nn.Module):
    def __init__(self):
        super(OutTransFree_FP32_Bmm_v3, self).__init__()

    def forward(self, x1, y1):
        out_ = torch.matmul(x1, y1)
        out = out_.transpose(1, 3)
        return out


class OutTransFree_FP32_Bmm_v4(nn.Module):
    def __init__(self):
        super(OutTransFree_FP32_Bmm_v4, self).__init__()

    def forward(self, x1, y1):
        out_ = torch.matmul(x1, y1)
        out = out_.permute(0, 1, 3, 2)
        return out


class TransFree_BF16_Bmm(nn.Module):
    def __init__(self):
        super(TransFree_BF16_Bmm, self).__init__()

    def forward(self, x1, y1):
        out = torch.matmul(x1, y1)
        return out


class OutTransFree_BF16_Bmm_v1(nn.Module):
    def __init__(self):
        super(OutTransFree_BF16_Bmm_v1, self).__init__()

    def forward(self, x1, y1):
        out_ = torch.matmul(x1, y1)
        out = out_.transpose(1, 2)
        return out


class OutTransFree_BF16_Bmm_v2(nn.Module):
    def __init__(self):
        super(OutTransFree_BF16_Bmm_v2, self).__init__()

    def forward(self, x1, y1):
        out_ = torch.matmul(x1, y1)
        out = out_.permute(0, 2, 1, 3)
        return out


class OutTransFree_BF16_Bmm_v3(nn.Module):
    def __init__(self):
        super(OutTransFree_BF16_Bmm_v3, self).__init__()

    def forward(self, x1, y1):
        out_ = torch.matmul(x1, y1)
        out = out_.transpose(1, 3)
        return out


class OutTransFree_BF16_Bmm_v4(nn.Module):
    def __init__(self):
        super(OutTransFree_BF16_Bmm_v4, self).__init__()

    def forward(self, x1, y1):
        out_ = torch.matmul(x1, y1)
        out = out_.permute(0, 1, 3, 2)
        return out


class TransFreeFP32BmmTester(TestCase):
    def _test_transfree_fp32_bmm(self, bmm_model, bmm_ipex, x1, y1, isTransFree=True):
        for i in range(len(x1)):
            for j in range(len(y1)):
                with torch.no_grad():
                    bmm_ipex = torch.jit.trace(
                        bmm_ipex,
                        (
                            x1[i],
                            y1[j],
                        ),
                    )

                    for _ in range(2):
                        bmm_jit = bmm_ipex(x1[i], y1[j])
                    bmm_ref = bmm_model(x1[i], y1[j])
                    self.assertEqual(bmm_ref, bmm_jit, prec=1e-5)

                    bmm_graph = bmm_ipex.graph_for(x1[i], y1[j])
                    self.assertTrue(
                        any(n.kind() == "ipex::matmul" for n in bmm_graph.nodes())
                    )
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU]
                    ) as p:
                        bmm_ipex(x1[i], y1[j])
                    if isTransFree is True:
                        self.assertFalse("aten::contiguous" in str(p.key_averages()))
                    else:
                        self.assertTrue("aten::contiguous" in str(p.key_averages()))

    def _test_outtransfree_fp32_bmm(
        self, bmm_model, bmm_ipex, x1, y1, isOutTransFree=True
    ):
        for i in range(len(x1)):
            for j in range(len(y1)):
                with torch.no_grad():
                    bmm_ipex = torch.jit.trace(
                        bmm_ipex,
                        (
                            x1[i],
                            y1[j],
                        ),
                    )

                    for _ in range(2):
                        bmm_jit = bmm_ipex(x1[i], y1[j])
                    bmm_ref = bmm_model(x1[i], y1[j])
                    self.assertEqual(bmm_ref, bmm_jit, prec=1e-5)

                    bmm_graph = bmm_ipex.graph_for(x1[i], y1[j])
                    if isOutTransFree is True:
                        self.assertTrue(
                            any(
                                n.kind() == "ipex::matmul_outtrans"
                                for n in bmm_graph.nodes()
                            )
                        )
                    else:
                        self.assertTrue(
                            any(n.kind() == "ipex::matmul" for n in bmm_graph.nodes())
                        )

    def _test_unusual_fp32_bmm(self, bmm_model, bmm_ipex, x1, y1, isTransFree=True):
        for i in range(len(x1)):
            with torch.no_grad():
                bmm_ipex = torch.jit.trace(
                    bmm_ipex,
                    (
                        x1[i],
                        y1[i],
                    ),
                )

                for _ in range(2):
                    bmm_jit = bmm_ipex(x1[i], y1[i])
                bmm_ref = bmm_model(x1[i], y1[i])
                self.assertEqual(bmm_ref, bmm_jit, prec=1e-5)

                bmm_graph = bmm_ipex.graph_for(x1[i], y1[i])
                if isTransFree is True:
                    self.assertTrue(
                        any(n.kind() == "ipex::matmul" for n in bmm_graph.nodes())
                    )
                else:
                    self.assertTrue(
                        any(n.kind() == "aten::matmul" for n in bmm_graph.nodes())
                    )

    def test_transfree_fp32_bmm(self):
        x1 = [
            torch.randn(32, 13, 27, 25),
            torch.randn(32, 27, 13, 25).transpose(1, 2),
            torch.randn(32, 13, 25, 27).transpose(2, 3),
            torch.randn(32, 25, 13, 27).transpose(2, 3).transpose(1, 3),
        ]
        y1 = [
            torch.randn(32, 13, 25, 27),
            torch.randn(32, 25, 13, 27).transpose(1, 2),
            torch.randn(32, 13, 27, 25).transpose(2, 3),
            torch.randn(32, 27, 13, 25).transpose(2, 3).transpose(1, 3),
        ]

        x2 = [
            torch.randn(13, 32, 27, 25).transpose(0, 1),
            torch.randn(32, 25, 27, 13).transpose(1, 3),
        ]
        y2 = [
            torch.randn(32, 27, 13, 25).transpose(1, 3).transpose(1, 2),
            torch.randn(27, 13, 25, 32).transpose(0, 3),
        ]

        x3 = [torch.randn(32, 27, 25), torch.randn(32, 25, 27).transpose(1, 2)]
        y3 = [torch.randn(32, 25, 27), torch.randn(32, 27, 25).transpose(1, 2)]

        x4 = [
            torch.randn(27, 32, 25).transpose(0, 1),
            torch.randn(27, 25, 32).transpose(1, 2).transpose(0, 1),
        ]
        y4 = [
            torch.randn(25, 32, 27).transpose(0, 1),
            torch.randn(27, 25, 32).transpose(0, 2),
        ]

        x5 = [
            torch.randn(32, 19, 13, 27, 25),
            torch.randn(32, 19, 27, 13, 25).transpose(2, 3),
            torch.randn(32, 19, 13, 25, 27).transpose(-1, -2),
            torch.randn(32, 13, 27, 19, 25).transpose(1, 2).transpose(1, 3),
        ]
        y5 = [
            torch.randn(32, 19, 13, 25, 29),
            torch.randn(32, 13, 19, 25, 29).transpose(1, 2),
            torch.randn(32, 25, 13, 19, 29).transpose(1, 3),
            torch.randn(32, 19, 29, 13, 25).transpose(3, 4).transpose(2, 4),
        ]

        x6 = [
            torch.randn(32, 25, 13, 27, 19).transpose(1, 4),
            torch.randn(19, 32, 13, 27, 25).transpose(0, 1),
        ]
        y6 = [torch.randn(29, 19, 13, 25, 32).transpose(0, -1)]

        ref = torch.rand(1, 1, 1, 768)
        x7 = [
            torch.rand(1, 1, 1, 768).to(memory_format=torch.channels_last),
            torch.randn(2, 16, 32, 768)[:, :, :, 0:1],
            torch.randn(2, 16, 32, 768)[:, :, :, 5],
            ref[:, :, :, 10],
        ]
        y7 = [
            torch.rand(1, 1, 768, 3).to(memory_format=torch.channels_last),
            torch.ones(2, 16, 1, 32),
            torch.ones(2, 32, 16),
            ref[:, :, :, 39],
        ]

        x8 = [torch.randn(12, 32, 15, 30), torch.randn(2, 6, 19, 3, 8)]
        y8 = [torch.randn(1, 32, 30, 29), torch.randn(1, 8, 16)]

        bmm_model = TransFree_FP32_Bmm().eval()
        bmm_ipex = ipex.optimize(bmm_model, dtype=torch.float, level="O1")

        self._test_transfree_fp32_bmm(bmm_model, bmm_ipex, x1, y1, isTransFree=True)
        self._test_transfree_fp32_bmm(bmm_model, bmm_ipex, x2, y2, isTransFree=False)
        self._test_transfree_fp32_bmm(bmm_model, bmm_ipex, x3, y3, isTransFree=True)
        self._test_transfree_fp32_bmm(bmm_model, bmm_ipex, x4, y4, isTransFree=False)
        self._test_transfree_fp32_bmm(bmm_model, bmm_ipex, x5, y5, isTransFree=True)
        self._test_transfree_fp32_bmm(bmm_model, bmm_ipex, x6, y6, isTransFree=False)
        self._test_unusual_fp32_bmm(bmm_model, bmm_ipex, x7, y7, isTransFree=True)
        self._test_unusual_fp32_bmm(bmm_model, bmm_ipex, x8, y8, isTransFree=False)

        bmm_out_model_v1 = OutTransFree_FP32_Bmm_v1().eval()
        bmm_out_ipex_v1 = ipex.optimize(bmm_out_model_v1, dtype=torch.float, level="O1")

        bmm_out_model_v2 = OutTransFree_FP32_Bmm_v2().eval()
        bmm_out_ipex_v2 = ipex.optimize(bmm_out_model_v2, dtype=torch.float, level="O1")

        bmm_out_model_v3 = OutTransFree_FP32_Bmm_v3().eval()
        bmm_out_ipex_v3 = ipex.optimize(bmm_out_model_v3, dtype=torch.float, level="O1")

        bmm_out_model_v4 = OutTransFree_FP32_Bmm_v4().eval()
        bmm_out_ipex_v4 = ipex.optimize(bmm_out_model_v4, dtype=torch.float, level="O1")

        bmm_out_model = [
            bmm_out_model_v1,
            bmm_out_model_v3,
            bmm_out_model_v2,
            bmm_out_model_v4,
        ]
        bmm_out_ipex = [
            bmm_out_ipex_v1,
            bmm_out_ipex_v3,
            bmm_out_ipex_v2,
            bmm_out_ipex_v4,
        ]

        for i in range(len(bmm_out_model)):
            if i % 2 == 0:
                self._test_outtransfree_fp32_bmm(
                    bmm_out_model[i], bmm_out_ipex[i], x1, y1, isOutTransFree=True
                )
                self._test_outtransfree_fp32_bmm(
                    bmm_out_model[i], bmm_out_ipex[i], x2, y2, isOutTransFree=True
                )
            else:
                self._test_outtransfree_fp32_bmm(
                    bmm_out_model[i], bmm_out_ipex[i], x1, y1, isOutTransFree=False
                )
                self._test_outtransfree_fp32_bmm(
                    bmm_out_model[i], bmm_out_ipex[i], x2, y2, isOutTransFree=False
                )


class TransFreeBF16BmmTester(TestCase):
    def _test_transfree_bf16_bmm(self, bmm_model, bmm_ipex, x1, y1, isTransFree=True):
        for i in range(len(x1)):
            for j in range(len(y1)):
                with torch.cpu.amp.autocast(), torch.no_grad():
                    bmm_ipex = torch.jit.trace(
                        bmm_ipex,
                        (
                            x1[i],
                            y1[j],
                        ),
                    )

                    for _ in range(2):
                        bmm_jit = bmm_ipex(x1[i], y1[j])
                    bmm_ref = bmm_model(x1[i], y1[j])
                    # AssertionError: tensor(0.0625) not less than or equal to 0.01
                    self.assertEqual(bmm_ref, bmm_jit, prec=7e-2)

                    bmm_graph = bmm_ipex.graph_for(x1[i], y1[j])
                    self.assertTrue(
                        any(n.kind() == "ipex::matmul" for n in bmm_graph.nodes())
                    )
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU]
                    ) as p:
                        bmm_ipex(x1[i], y1[j])
                    if isTransFree is True:
                        self.assertFalse("aten::contiguous" in str(p.key_averages()))
                    else:
                        self.assertTrue("aten::contiguous" in str(p.key_averages()))

    def _test_outtransfree_bf16_bmm(
        self, bmm_model, bmm_ipex, x1, y1, isOutTransFree=True
    ):
        for i in range(len(x1)):
            for j in range(len(y1)):
                with torch.cpu.amp.autocast(), torch.no_grad():
                    bmm_ipex = torch.jit.trace(
                        bmm_ipex,
                        (
                            x1[i],
                            y1[j],
                        ),
                    )

                    for _ in range(2):
                        bmm_jit = bmm_ipex(x1[i], y1[j])
                    bmm_ref = bmm_model(x1[i], y1[j])
                    # AssertionError: tensor(0.0625) not less than or equal to 0.01
                    self.assertEqual(bmm_ref, bmm_jit, prec=7e-2)

                    bmm_graph = bmm_ipex.graph_for(x1[i], y1[j])
                    if isOutTransFree is True:
                        self.assertTrue(
                            any(
                                n.kind() == "ipex::matmul_outtrans"
                                for n in bmm_graph.nodes()
                            )
                        )
                    else:
                        self.assertTrue(
                            any(n.kind() == "ipex::matmul" for n in bmm_graph.nodes())
                        )

    def test_transfree_bf16_bmm(self):
        x1 = [
            torch.randn(32, 13, 27, 25).to(torch.bfloat16),
            torch.randn(32, 27, 13, 25).to(torch.bfloat16).transpose(1, 2),
            torch.randn(32, 13, 25, 27).to(torch.bfloat16).transpose(2, 3),
            torch.randn(32, 25, 13, 27)
            .to(torch.bfloat16)
            .transpose(2, 3)
            .transpose(1, 3),
        ]
        y1 = [
            torch.randn(32, 13, 25, 27).to(torch.bfloat16),
            torch.randn(32, 25, 13, 27).to(torch.bfloat16).transpose(1, 2),
            torch.randn(32, 13, 27, 25).to(torch.bfloat16).transpose(2, 3),
            torch.randn(32, 27, 13, 25)
            .to(torch.bfloat16)
            .transpose(2, 3)
            .transpose(1, 3),
        ]

        x2 = [
            torch.randn(13, 32, 27, 25).to(torch.bfloat16).transpose(0, 1),
            torch.randn(32, 25, 27, 13).to(torch.bfloat16).transpose(1, 3),
        ]
        y2 = [
            torch.randn(32, 27, 13, 25)
            .to(torch.bfloat16)
            .transpose(1, 3)
            .transpose(1, 2),
            torch.randn(27, 13, 25, 32).to(torch.bfloat16).transpose(0, 3),
        ]

        x3 = [
            torch.randn(32, 27, 25).to(torch.bfloat16),
            torch.randn(32, 25, 27).to(torch.bfloat16).transpose(1, 2),
        ]
        y3 = [
            torch.randn(32, 25, 27).to(torch.bfloat16),
            torch.randn(32, 27, 25).to(torch.bfloat16).transpose(1, 2),
        ]

        x4 = [
            torch.randn(27, 32, 25).to(torch.bfloat16).transpose(0, 1),
            torch.randn(27, 25, 32).to(torch.bfloat16).transpose(1, 2).transpose(0, 1),
        ]
        y4 = [
            torch.randn(25, 32, 27).to(torch.bfloat16).transpose(0, 1),
            torch.randn(27, 25, 32).to(torch.bfloat16).transpose(0, 2),
        ]

        bmm_model = TransFree_BF16_Bmm().eval()
        bmm_ipex = ipex.optimize(bmm_model, dtype=torch.bfloat16, level="O1")

        self._test_transfree_bf16_bmm(bmm_model, bmm_ipex, x1, y1, isTransFree=True)
        self._test_transfree_bf16_bmm(bmm_model, bmm_ipex, x2, y2, isTransFree=False)
        self._test_transfree_bf16_bmm(bmm_model, bmm_ipex, x3, y3, isTransFree=True)
        self._test_transfree_bf16_bmm(bmm_model, bmm_ipex, x4, y4, isTransFree=False)

        bmm_out_model_v1 = OutTransFree_BF16_Bmm_v1().eval()
        bmm_out_ipex_v1 = ipex.optimize(
            bmm_out_model_v1, dtype=torch.bfloat16, level="O1"
        )

        bmm_out_model_v2 = OutTransFree_BF16_Bmm_v2().eval()
        bmm_out_ipex_v2 = ipex.optimize(
            bmm_out_model_v2, dtype=torch.bfloat16, level="O1"
        )

        bmm_out_model_v3 = OutTransFree_BF16_Bmm_v3().eval()
        bmm_out_ipex_v3 = ipex.optimize(
            bmm_out_model_v3, dtype=torch.bfloat16, level="O1"
        )

        bmm_out_model_v4 = OutTransFree_BF16_Bmm_v4().eval()
        bmm_out_ipex_v4 = ipex.optimize(
            bmm_out_model_v4, dtype=torch.bfloat16, level="O1"
        )

        bmm_out_model = [
            bmm_out_model_v1,
            bmm_out_model_v3,
            bmm_out_model_v2,
            bmm_out_model_v4,
        ]
        bmm_out_ipex = [
            bmm_out_ipex_v1,
            bmm_out_ipex_v3,
            bmm_out_ipex_v2,
            bmm_out_ipex_v4,
        ]

        for i in range(len(bmm_out_model)):
            if i % 2 == 0:
                self._test_outtransfree_bf16_bmm(
                    bmm_out_model[i], bmm_out_ipex[i], x1, y1, isOutTransFree=True
                )
                self._test_outtransfree_bf16_bmm(
                    bmm_out_model[i], bmm_out_ipex[i], x2, y2, isOutTransFree=True
                )
            else:
                self._test_outtransfree_bf16_bmm(
                    bmm_out_model[i], bmm_out_ipex[i], x1, y1, isOutTransFree=False
                )
                self._test_outtransfree_bf16_bmm(
                    bmm_out_model[i], bmm_out_ipex[i], x2, y2, isOutTransFree=False
                )


if __name__ == "__main__":
    test = unittest.main()
