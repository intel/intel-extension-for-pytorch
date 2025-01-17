import numpy as np
import torch
import intel_extension_for_pytorch as ipex
from torch.testing._internal.common_utils import TestCase
import pytest


class TestUtils(TestCase):
    def test_onednn_verbose(self):
        verb_list = [
            torch.xpu.OnednnVerbLevel.OFF,
            torch.xpu.OnednnVerbLevel.ON,
            torch.xpu.OnednnVerbLevel.ON_DETAIL,
        ]
        for verb in verb_list:
            torch.xpu.set_onednn_verbose(verb)

        for verb in verb_list:
            with torch.xpu.onednn_verbose(verb):
                pass

    @pytest.mark.skipif("not torch.xpu.has_onemkl()")
    def test_onemkl_verbose(self):
        verb_list = [
            torch.xpu.OnemklVerbLevel.OFF,
            torch.xpu.OnemklVerbLevel.ON,
            torch.xpu.OnemklVerbLevel.ON_SYNC,
        ]
        for verb in verb_list:
            torch.xpu.set_onemkl_verbose(verb)

        for verb in verb_list:
            with torch.xpu.onemkl_verbose(verb):
                pass

    def test_fp32_math_mode(self):
        mode_list = [
            torch.xpu.FP32MathMode.FP32,
            torch.xpu.FP32MathMode.TF32,
            torch.xpu.FP32MathMode.BF32,
        ]
        for mode in mode_list:
            torch.xpu.set_fp32_math_mode(mode)
            assert torch.xpu.get_fp32_math_mode() == mode, (
                "Fail to enable FP32 math mode: " + mode
            )

        for mode in mode_list:
            with torch.xpu.fp32_math_mode(mode):
                assert torch.xpu.get_fp32_math_mode() == mode, (
                    "Fail to enable FP32 math mode: " + mode
                )

    def test_compute_eng(self):
        eng_list = [
            torch.xpu.XPUComputeEng.RECOMMEND,
            torch.xpu.XPUComputeEng.BASIC,
            torch.xpu.XPUComputeEng.ONEDNN,
            torch.xpu.XPUComputeEng.ONEMKL,
            torch.xpu.XPUComputeEng.XETLA,
        ]
        for eng in eng_list:
            torch.xpu.set_compute_eng(eng)
            assert torch.xpu.get_compute_eng() == eng, (
                "Fail to enable XPU Compute Engine: " + eng
            )

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_has_fp64_dtype(self):
        y = (
            torch.tensor(1, device="xpu", dtype=torch.double) ** 2
        ).cpu().numpy() == np.array(1)
        assert y == torch.xpu.has_fp64_dtype(), "This Device Not Support FP64"

    def test_device_capability(self):
        capability = torch.xpu.get_device_capability()
        assert "max_work_group_size" in capability, "key max_work_group_size not found"
        assert "max_num_sub_groups" in capability, "key max_num_sub_groups not found"
        assert "sub_group_sizes" in capability, "key sub_group_sizes not found"

    def test_mem_get_info(self):
        self.assertGreater(ipex.xpu.mem_get_info()[0], 0)
        self.assertGreater(ipex.xpu.mem_get_info()[1], 0)
