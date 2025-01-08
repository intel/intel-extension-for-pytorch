import os
from setuptools import setup

from intel_extension_for_pytorch.xpu.cpp_extension import (
    DPCPPExtension,
    DpcppBuildExtension,
    IS_LINUX,
)

# Here ze_loader is not necessary, just used to check libraries linker
libraries = ["ze_loader"] if IS_LINUX else []

dpcpp_path = os.getenv("CMPLR_ROOT")
sycl_ext_path = os.path.join(dpcpp_path, "include", "sycl")

setup(
    name="mod_test_add_non_ninja",
    ext_modules=[
        DPCPPExtension(
            "mod_test_add_non_ninja",
            [
                "test_dpcpp_add.cpp",
            ],
            extra_compile_args=[],
            libraries=libraries,
            include_dirs=[sycl_ext_path],
        )
    ],
    cmdclass={"build_ext": DpcppBuildExtension.with_options(use_ninja=False)},
)
