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
dpcpp_sycl_path = os.path.join(dpcpp_path, "include", "sycl")

conda_path = os.getenv("CONDA_PREFIX")
conda_sycl_path = os.path.join(conda_path, "include", "sycl")

setup(
    name="mod_test_add_ninja",
    ext_modules=[
        DPCPPExtension(
            "mod_test_add_ninja",
            [
                "test_dpcpp_add.cpp",
            ],
            extra_compile_args=[],
            libraries=libraries,
            include_dirs=(
                [dpcpp_sycl_path]
                if not os.path.exists(conda_sycl_path)
                else [conda_sycl_path]
            ),
        )
    ],
    cmdclass={"build_ext": DpcppBuildExtension.with_options(use_ninja=True)},
)
