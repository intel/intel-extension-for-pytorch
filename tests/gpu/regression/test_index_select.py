import torch
from torch.testing._internal.common_utils import TestCase, IS_WINDOWS
import intel_extension_for_pytorch  # noqa F401
import pytest


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(
        IS_WINDOWS, reason="Memory allocated by this case exceed Windows provide."
    )
    @pytest.mark.skipif(
        not torch.xpu.has_2d_block_array(),
        reason="Memory allocated by this case exceed ATSM provide.",
    )
    def test_index_select_large_1(self, dtype=torch.float):
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
        src = torch.rand((256000, 128))

        index = torch.randint(0, src.size(0), (32000000,))
        dst_cpu = src.index_select(0, index)
        # print("dst_cpu = ", dst_cpu)

        src_xpu = src.to("xpu")
        index_xpu = index.to("xpu")
        dst_xpu = src_xpu.index_select(0, index_xpu)

        # print("dst_xpu = ", dst_xpu)
        # print("diff = ", torch.max(abs(dst_xpu.cpu()-dst_cpu)))

        self.assertEqual(dst_cpu, dst_xpu.cpu())
        del src_xpu
        del index_xpu
        del dst_xpu
        torch.xpu.synchronize()
        torch.xpu.empty_cache()

    @pytest.mark.skipif(
        IS_WINDOWS, reason="Memory allocated by this case exceed Windows provide."
    )
    @pytest.mark.skipif(
        not torch.xpu.has_2d_block_array(),
        reason="Memory allocated by this case exceed ATSM provide.",
    )
    def test_index_select_large_2(self, dtype=torch.float):
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
        src = torch.rand((512000, 128))

        index = torch.randint(0, src.size(0), (20002185,))
        dst_cpu = src.index_select(0, index)
        # print("dst_cpu = ", dst_cpu)

        src_xpu = src.to("xpu")
        index_xpu = index.to("xpu")
        dst_xpu = src_xpu.index_select(0, index_xpu)

        # print("dst_xpu = ", dst_xpu)
        # print("diff = ", torch.max(abs(dst_xpu.cpu()-dst_cpu)))

        self.assertEqual(dst_cpu, dst_xpu.cpu())
        del src_xpu
        del index_xpu
        del dst_xpu
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
