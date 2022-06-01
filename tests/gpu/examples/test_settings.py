import torch
import intel_extension_for_pytorch
from torch.testing._internal.common_utils import TestCase

import pytest


class TestVerbose(TestCase):
    def test_ipex_verbose(self):
        verb_list = [torch.xpu.VERBOSE_LEVEL.ON, torch.xpu.VERBOSE_LEVEL.OFF]
        for verb in verb_list:
            torch.xpu.set_verbose_level(verb)
            assert torch.xpu.get_verbose_level() == verb, 'Fail to set IPEX_VERBOSE level: ' + verb

    def test_onednn_verbose(self):
        verb_list = [torch.xpu.ONEDNN_VERB_LEVEL.OFF, torch.xpu.ONEDNN_VERB_LEVEL.ON, torch.xpu.ONEDNN_VERB_LEVEL.ON_DETAIL]
        for verb in verb_list:
            torch.xpu.set_onednn_verbose(verb)

        for verb in verb_list:
            with torch.xpu.onednn_verbose(verb):
                pass

    def test_onemkl_verbose(self):
        verb_list = [torch.xpu.ONEMKL_VERB_LEVEL.OFF, torch.xpu.ONEMKL_VERB_LEVEL.ON, torch.xpu.ONEMKL_VERB_LEVEL.ON_SYNC]
        for verb in verb_list:
            torch.xpu.set_onemkl_verbose(verb)

        for verb in verb_list:
            with torch.xpu.onemkl_verbose(verb):
                pass

    def test_fp32_math_mode(self):
        mode_list = [torch.xpu.FP32_MATH_MODE.FP32, torch.xpu.FP32_MATH_MODE.TF32, torch.xpu.FP32_MATH_MODE.BF32]
        for mode in mode_list:
            torch.xpu.set_fp32_math_mode(mode)
            assert torch.xpu.get_fp32_math_mode() == mode, 'Fail to enable FP32 math mode: ' + mode

        for mode in mode_list:
            with torch.xpu.fp32_math_mode(mode):
                assert torch.xpu.get_fp32_math_mode() == mode, 'Fail to enable FP32 math mode: ' + mode

    def test_sync_mode(self):
        with torch.xpu.sync_mode():
            assert torch.xpu.using_sync_mode(), 'Fail to set sync mode'

    def test_onednn_layout(self):
        with torch.xpu.onednn_layout():
            assert torch.xpu.using_onednn_layout(), 'Fail to set onednn layout'

    def test_xpu_backend(self):
        backend_list = [torch.xpu.XPU_BACKEND.GPU, torch.xpu.XPU_BACKEND.CPU, torch.xpu.XPU_BACKEND.AUTO]
        for backend in backend_list:
            torch.xpu.set_backend(backend)
            assert torch.xpu.get_backend() == backend, 'Fail to set XPU backend: ' + backend
