import torch
import intel_extension_for_pytorch  # noqa
from torch.xpu.cpp_extension import load, IS_LINUX
import os

dnnl_path = os.getenv("DNNLROOT")

if dnnl_path is not None:
    module = load(
        name="mod_test_sycl_device",
        sources=["test_sycl_device.cpp"],
        extra_cflags=["-O2"],
        verbose=True,
    )

    import mod_test_sycl_device
    from torch.testing._internal.common_utils import TestCase

    class TestTorchMethod(TestCase):
        def test_sycl_device_capsule(self):
            d = torch.xpu.device(0)
            d_ptr = d.sycl_device
            self.assertTrue(mod_test_sycl_device.is_sycl_device(d_ptr))

            if IS_LINUX:
                import ctypes

                path = os.path.join(os.path.expanduser("~/.cache"), "torch_extensions")
                for root, dirs, files in os.walk(path):
                    if "mod_test_sycl_device.so" in files:
                        so_file = os.path.join(root, "mod_test_sycl_device.so")
                        break
                lib = ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
                self.assertTrue(lib.isSYCLDevice(d))

else:
    print(
        "Please source <oneapi_dir>/dnnl/<version>/env/vars.sh, and re-run this test case."
    )
