import torch
import intel_extension_for_pytorch # noqa
from torch.testing._internal.common_utils import TestCase

import pytest


class TestVerbose(TestCase):
    def test_ipex_verbose(self):
        verb_list = [torch.xpu.VerbLevel.ON, torch.xpu.VerbLevel.OFF]
        for verb in verb_list:
            torch.xpu.set_verbose_level(verb)
            assert torch.xpu.get_verbose_level() == verb, 'Fail to set IPEX_VERBOSE level: ' + verb

    def test_onednn_verbose(self):
        verb_list = [torch.xpu.OnednnVerbLevel.OFF, torch.xpu.OnednnVerbLevel.ON, torch.xpu.OnednnVerbLevel.ON_DETAIL]
        for verb in verb_list:
            torch.xpu.set_onednn_verbose(verb)

        for verb in verb_list:
            with torch.xpu.onednn_verbose(verb):
                pass

    @pytest.mark.skipif("not torch.xpu.has_onemkl()")
    def test_onemkl_verbose(self):
        verb_list = [torch.xpu.OnemklVerbLevel.OFF, torch.xpu.OnemklVerbLevel.ON, torch.xpu.OnemklVerbLevel.ON_SYNC]
        for verb in verb_list:
            torch.xpu.set_onemkl_verbose(verb)

        for verb in verb_list:
            with torch.xpu.onemkl_verbose(verb):
                pass

    def test_fp32_math_mode(self):
        mode_list = [torch.xpu.FP32MathMode.FP32, torch.xpu.FP32MathMode.TF32, torch.xpu.FP32MathMode.BF32]
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
        backend_list = [torch.xpu.Backend.GPU, torch.xpu.Backend.CPU, torch.xpu.Backend.AUTO]
        for backend in backend_list:
            torch.xpu.set_backend(backend)
            assert torch.xpu.get_backend() == backend, 'Fail to set XPU backend: ' + backend
