import torch
from torch.testing._internal.common_utils import (TestCase,
                                                  repeat_test_for_types)

from torch.testing._internal.common_dtype import (get_all_int_dtypes,
                                                  get_all_fp_dtypes)

import numpy as np

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    @repeat_test_for_types([*(get_all_int_dtypes() + get_all_fp_dtypes())])
    def test_argmin(self, dtype):
        t = torch.ones(3, 3, device=cpu_device, dtype=dtype)
        t_cpu = torch.argmin(t)
        dst_t = t.clone().to(dpcpp_device)
        t_xpu = torch.argmin(dst_t)
        t_np = np.argmin(t_cpu)
        self.assertEqual(t_cpu, t_xpu.to(cpu_device))
        self.assertEqual(t_cpu, t_np)
        self.assertEqual(t_np, t_xpu.to(cpu_device))

    @repeat_test_for_types([*(get_all_int_dtypes() + get_all_fp_dtypes())])
    def test_argmax(self, dtype):
        t = torch.ones(3, 3, device=cpu_device, dtype=dtype)
        t_cpu = torch.argmax(t)
        dst_t = t.clone().to(dpcpp_device)
        t_xpu = torch.argmax(dst_t)
        t_np = np.argmax(t_cpu)
        self.assertEqual(t_cpu, t_xpu.to(cpu_device))
        self.assertEqual(t_cpu, t_np)
        self.assertEqual(t_np, t_xpu.to(cpu_device))
