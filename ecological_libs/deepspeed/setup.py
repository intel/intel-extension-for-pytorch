import os
import sys
import warnings
from functools import lru_cache
from setuptools import setup

PACKAGE_NAME = "intel_extension_for_pytorch_deepspeed"

@lru_cache(maxsize=128)
def _get_build_target():
    build_target = ""
    if len(sys.argv) > 1:
        if sys.argv[1] in ["clean"]:
            build_target = "clean"
        elif sys.argv[1] in ["develop"]:
            build_target = "develop"
        elif sys.argv[1] in ["bdist_wheel"]:
            build_target = "bdist_wheel"
        else:
            build_target = "python"
    return build_target

if _get_build_target() in ["develop", "python", "bdist_wheel"]:
    try:
        import intel_extension_for_pytorch
        from torch.xpu.cpp_extension import DPCPPExtension, DpcppBuildExtension
    except ImportError as e:
        raise RuntimeError("Fail to import intel_extension_for_pytorch!")

def get_version_num():
    versions = {}
    version_file = "../../version.txt"
    version_lines = open(version_file, "r").readlines()
    for line in version_lines:
        key, value = line.strip().split(" ")
        versions[key] = value
    for v in ("VERSION_MAJOR", "VERSION_MINOR", "VERSION_PATCH"):
        if v not in versions:
            print("ERROR:", v, "is not found in", version_file)
            sys.exit(1)
    version = (
        versions["VERSION_MAJOR"]
        + "."
        + versions["VERSION_MINOR"]
        + "."
        + versions["VERSION_PATCH"]
    )
    return version

def get_build_version():
    ipex_ds_version = get_version_num()
    return ipex_ds_version


def get_project_dir():
    project_root_dir = os.path.dirname(__file__)
    return os.path.abspath(project_root_dir)


def get_build_dir():
    return os.path.join(get_project_dir(), "build")


def get_csrc_dir():
    project_root_dir = os.path.join(get_project_dir(), "csrc")
    return os.path.abspath(project_root_dir)


def get_module_csrc_dir(module_name):
    module_csrc_dir = get_csrc_dir()
    if module_name == "quantization":
        module_csrc_dir = os.path.join(module_csrc_dir, "quantization")
    if module_name == "transformer_inference":
        module_csrc_dir = os.path.join(module_csrc_dir,
                                       "transformer/inference/csrc")
    return os.path.abspath(module_csrc_dir)


def get_module_include_dir(module_name):
    module_inc_dirs = []
    module_inc_dirs.append(
        os.path.abspath(os.path.join(get_csrc_dir(), "includes")))
    module_inc_dirs.append(
        os.path.abspath(os.path.join(get_csrc_dir(), "includes/dpct")))
    if module_name == "transformer_inference":
        module_inc_dirs.append(
            os.path.abspath(
                os.path.join(get_csrc_dir(),
                             "transformer/inference/includes")))
    return module_inc_dirs


def create_ext_modules():
    modules_names = ['quantization', 'transformer_inference']
    ext_modules = []
    aot_device_list = os.environ.get("USE_AOT_DEVLIST")

    if 'pvc' not in aot_device_list:
        raise OSError("intel_extension_for_pytorch_deepspeed only supports pvc for now.")
        return ext_modules

    for module_name in modules_names:
        cxx_flags = [
            '-fsycl', '-fsycl-targets=spir64_gen', '-g', '-gdwarf-4', '-O3',
            '-std=c++17', '-fPIC', '-DMKL_ILP64', '-fno-strict-aliasing',
            '-DBF16_AVAILABLE'
        ]
        extra_ldflags = [
            '-fPIC', '-fsycl', '-fsycl-targets=spir64_gen',
            '-fsycl-max-parallel-link-jobs=8',
            '-Xs "-options -cl-poison-unsupported-fp64-kernels,cl-intel-enable-auto-large-GRF-mode"',
            '-Xs "-device pvc"', '-Wl,-export-dynamic'
        ]
        cpp_files = []
        include_dirs = get_module_include_dir(module_name)

        for path, _, file_list in os.walk(get_module_csrc_dir(module_name)):
            for file_name in file_list:
                if file_name.endswith('.cpp'):
                    cpp_files += [os.path.join(path, file_name)]

        ext_modules.append(
            DPCPPExtension(name=module_name,
                           sources=cpp_files,
                           include_dirs=include_dirs,
                           extra_compile_args={'cxx': cxx_flags},
                           extra_link_args=extra_ldflags))

    return ext_modules


base_dir = os.path.dirname(os.path.abspath(__file__))
ipex_ds_build_version = get_build_version()


def _build_installation_dependency():
    install_requires = []
    install_requires.append("setuptools")
    return install_requires


ext_modules = []
cmdclass = {}
if _get_build_target() in ["develop", "python", "bdist_wheel"]:
    ext_modules += create_ext_modules()
    cmdclass["build_ext"] = DpcppBuildExtension

long_description = ""
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name=PACKAGE_NAME,
    version=ipex_ds_build_version,
    description="Intel® Extension for PyTorch* DeepSpeed Kernel",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/intel/intel-extension-for-pytorch/",
    author="Intel Corp.",
    install_requires=_build_installation_dependency(),
    packages=[PACKAGE_NAME],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    license="https://www.apache.org/licenses/LICENSE-2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
    ],
)
