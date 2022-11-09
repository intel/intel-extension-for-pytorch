from intel_extension_for_pytorch.xpu.cpp_extension import load
import os


module = load(
    name='operation_syclkernel', 
    sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), each) 
             for each in ['operation_syclkernel.cpp', 'device_memory.cpp']], 
    extra_cflags=['-O2'], 
    verbose=True)
