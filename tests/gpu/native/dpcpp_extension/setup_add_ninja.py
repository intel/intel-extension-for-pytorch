from setuptools import setup

from intel_extension_for_pytorch.xpu.cpp_extension import DPCPPExtension, DpcppBuildExtension

setup(
    name='mod_test_add_ninja',
    ext_modules=[
        DPCPPExtension('mod_test_add_ninja', [
            'test_dpcpp_add.cpp',
        ], extra_compile_args=[])
    ],
    cmdclass={
        'build_ext': DpcppBuildExtension.with_options(use_ninja=True)
    }
)
