from intel_extension_for_pytorch.xpu.cpp_extension import load
import os

dnnl_path = os.getenv('DNNLROOT')

if dnnl_path is not None:
    module = load(
        name='operation_syclkernel',
        sources=['operation_syclkernel.cpp', 'device_memory.cpp'],
        extra_cflags=['-O2'],
        verbose=True)
else:
    print("Please source <oneapi_dir>/dnnl/<version>/env/vars.sh, and re-run this test case.")
