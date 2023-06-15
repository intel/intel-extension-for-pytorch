from setuptools import setup

from intel_extension_for_pytorch.xpu.cpp_extension import DPCPPExtension, DpcppBuildExtension

setup(
    name='mod_test_add_non_ninja',
    ext_modules=[
        DPCPPExtension('mod_test_add_non_ninja', [
            'test_dpcpp_add.cpp',
        # Here ze_loader is not necessary, just used to check libraries linker
        ], extra_compile_args=[], libraries=['ze_loader'])
    ],
    cmdclass={
        'build_ext': DpcppBuildExtension.with_options(use_ninja=False)
    }
)
