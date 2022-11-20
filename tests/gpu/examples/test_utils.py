import numpy as np
import torch
import intel_extension_for_pytorch  # noqa
from torch.testing._internal.common_utils import TestCase
from intel_extension_for_pytorch.xpu.utils import using_tile_as_device, has_fp64_dtype
from intel_extension_for_pytorch.xpu import getDeviceIdListForCard
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

    def test_force_onednn_primitive(self):
        with torch.xpu.force_onednn_primitive():
            assert torch.xpu.using_force_onednn_primitive(), 'Fail to force onednn primitive'

class TestDevicdeListForCard(TestCase):
    def test_devicelist_empty(self):
        if torch.xpu.device_count() > 0:
            assert getDeviceIdListForCard(), 'Device list should not be empty'

    def test_devicelist_size(self):
        assert len(getDeviceIdListForCard()) <= torch.xpu.device_count(), \
            'The size of device list should not be larger than device count'

    def test_implicit_mode(self):
        if not using_tile_as_device():
            assert len(getDeviceIdListForCard()) == 1, \
                'The size of device list should be always 1 with implicit mode'


class TestHasDtypes(TestCase):
    def test_has_fp64_dtype(self):
        y = (torch.tensor(1, device="xpu", dtype=torch.double)**2).cpu().numpy() == np.array(1)
        assert (y == has_fp64_dtype()), "This Device Not Support FP64"
