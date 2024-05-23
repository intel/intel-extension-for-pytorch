import torch
import intel_extension_for_pytorch  # noqa
from torch.xpu.cpp_extension import load, IS_LINUX
import os


module = load(
    name="mod_test_sycl_queue",
    sources=["test_sycl_queue.cpp"],
    extra_cflags=["-O2"],
    verbose=True,
)

import mod_test_sycl_queue
from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):
    def test_sycl_queue(self):
        s = torch.xpu.current_stream()
        q_ptr = s.sycl_queue
        self.assertTrue(mod_test_sycl_queue.is_sycl_queue(q_ptr))

        if IS_LINUX:
            import ctypes

            path = os.path.join(os.path.expanduser("~/.cache"), "torch_extensions")
            for root, dirs, files in os.walk(path):
                if "mod_test_sycl_queue.so" in files:
                    so_file = os.path.join(root, "mod_test_sycl_queue.so")
                    break
            lib = ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
            self.assertTrue(lib.isSYCLQueue(s))
