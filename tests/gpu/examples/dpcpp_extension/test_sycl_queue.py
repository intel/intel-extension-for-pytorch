import torch
import intel_extension_for_pytorch  # noqa
from torch.xpu.cpp_extension import load
import os

dnnl_path = os.getenv("DNNLROOT")

if dnnl_path is not None:
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
            self.assertTrue(mod_test_sycl_queue.is_sycl_queue_pointer(q_ptr))

else:
    print(
        "Please source <oneapi_dir>/dnnl/<version>/env/vars.sh, and re-run this test case."
    )
