import torch
from torch.testing._internal.common_utils import TestCase, IS_WINDOWS
import os
import psutil
import intel_extension_for_pytorch  # noqa

import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    @pytest.mark.skipif(
        IS_WINDOWS, reason="not applicable on Windows"
    )
    def test_non_blocking_H2D_copy(self):
        def dummy_compute(tensor):
            return (tensor + 1.0) * 0.1

        # verify async H2D caching allocation recycle
        H2D_CNT = 20
        CURRENT_PID = os.getpid()
        verify_start = 10
        pre_anon_value = 0

        def get_heap_anon_memory_consume(PID):
            memory_maps = psutil.Process(PID).memory_maps()
            for map in memory_maps:
                map = str(map)
                if '[heap]' in map:
                    anon_str = 'anonymous'
                    anon_index = map.find(anon_str) + len(anon_str) + 1
                    sub_str = map[anon_index:]
                    value_str = sub_str[0:sub_str.find(',')]
            return int(value_str)

        for i in range(H2D_CNT):
            input_cpu = torch.randn([256, 3, 224, 224])
            input_xpu = input_cpu.to(dpcpp_device, non_blocking=True)
            input_xpu = dummy_compute(input_xpu)
            torch.xpu.synchronize()

            # verify async H2D correctness
            self.assertEqual(dummy_compute(input_cpu), input_xpu.cpu())

            current_anon_value = get_heap_anon_memory_consume(CURRENT_PID)
            if i > verify_start:
                # verify async H2D caching allocation recycle
                self.assertEqual(pre_anon_value, current_anon_value)
                pre_anon_value = current_anon_value
            elif i == verify_start:
                pre_anon_value = current_anon_value
            else:
                pass
