from setuptools import setup
from intel_extension_for_pytorch.xpu.cpp_extension import DpcppBuildExtension, DPCPPExtension

setup(
    name='operation_syclkernel',
    ext_modules=[
        DPCPPExtension('operation_syclkernel', sources=[
            'operation_syclkernel.cpp',
            'device_memory.cpp'
        ],
            extra_compile_args={
            'cxx': ['-std=c++20', '-fPIC'],
        }),
    ],
    cmdclass={
        'build_ext': DpcppBuildExtension.with_options(use_ninja=True)
    })
