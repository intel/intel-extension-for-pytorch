import torch
import intel_extension_for_pytorch
from torch.xpu.cpp_extension import load
import os

dnnl_path = os.getenv('DNNLROOT')

if dnnl_path is not None:
    module = load(
        name='check_syclqueue',
        sources=['test_sycl_queue.cpp'],
        extra_cflags=['-O2'],
        verbose=True)

    import check_syclqueue
    from torch.testing._internal.common_utils import TestCase
    import pytest

    class TestTorchMethod(TestCase):
        def test_sycl_queue(self):
            s = torch.xpu.current_stream()
            q_ptr = s.sycl_queue
            self.assertTrue(check_syclqueue.is_sycl_queue_pointer(q_ptr))
else:
    print("Please source <oneapi_dir>/dnnl/<version>/env/vars.sh, and re-run this test case.")
