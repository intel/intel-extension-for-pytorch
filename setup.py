# This Python file uses the following encoding: utf-8
# !/usr/bin/env python

# Welcome to the Intel Extension for PyTorch setup.py.
#
# Environment variables you are probably interested in:
#
#   DEBUG
#     build with -O0 and -g (debug symbols)
#
#   RELEASE
#     build with optimization level -O2
#
#   REL_WITH_DEB_INFO
#     build with optimization level -O2 and -g (debug symbols)
#
#   CFLAGS
#     flags to apply to both C and C++ files to be compiled (a quirk of setup.py
#     which we have faithfully adhered to in our build system is that CFLAGS
#     also applies to C++ files (unless CXXFLAGS is set), in contrast to the
#     default behavior of autogoo and cmake build systems.)
#
#   CC
#     the C/C++ compiler to use
#
#   MKLROOT
#     specify MKL library path.
#     ONLY NEEDED if you have a specific MKL version you want to link against.
#     Make sure this directory contains include and lib directories.
#     By default, the MKL library installed with pip/conda is used.
#
# Environment variables we respect (these environment variables are
# conventional and are often understood/set by other software.)
#
#   TORCH_VERSION
#     specify the PyTorch version to depend on
#
#   IPEX_VERSION
#     specify the extension version literal
#
#   MAX_JOBS
#     process for parallel compile, must be a Integer
#
#   VERBOSE
#     more output when compile
#
#   IPEX_VERSIONED_BUILD
#     build wheel files versioned with a git commit number
#

##############################################################
# Common build options:
# BUILD_WITH_CPU        - to build IPEX with CPU libraries
# BUILD_WITH_XPU        - to build IPEX with XPU libraries
#
# CPU build options:
# IPEX_DISP_OP          - to output the extension operators name for debug purpose
#
# XPU build options:
# USE_ONEMKL            - to use oneMKL in operators
# USE_PERSIST_STREAM    - to use persistent oneDNN stream
# USE_PRIMITIVE_CACHE   - to Cache oneDNN primitives by framework
# USE_QUEUE_BARRIER     - to use queue submit_barrier API
# USE_SCRATCHPAD_MODE   - to trun on oneDNN scratchpad user mode
# USE_SYCL_ASSERT       - to enable assert in sycl kernel
# USE_ITT_ANNOTATION    - to enable ITT annotation in sycl kernel
# USE_SPLIT_FP64_LOOPS  - to split FP64 loops into separate kernel for element-wise kernels
# USE_OVERRIDE_OP       - to use the operator in IPEX to override the duplicated one in stock PyTorch
# BUILD_STATS           - to count statistics for each component during build process
# BUILD_BY_PER_KERNEL   - to build by DPC++ per_kernel option (exclusive with AOT enabled)
# BUILD_STRIPPED_BIN    - to strip all symbols after build
# BUILD_SEPARATE_OPS    - to build each operator in separate library
# BUILD_OPT_LEVEL       - to add build option -Ox, accept values: 0/1
# BUILD_NO_CLANGFORMAT  - to build without force clang-format
# BUILD_INTERNAL_DEBUG  - to build internal debug code path
# BUILD_WITH_SANITIZER  - use sanitizer check, support one of address, thread, and leak options at a time.
#                         The default option is address
#
##############################################################

from __future__ import print_function
from setuptools.command.build_py import build_py
from setuptools.command.install import install
from functools import lru_cache
from subprocess import check_call
from setuptools.command.build_clib import build_clib
from setuptools.command.egg_info import egg_info
from setuptools import setup, distutils
from pathlib import Path

import sysconfig
import distutils.ccompiler
import distutils.command.clean
import os
import glob
import platform
import shutil
import subprocess
import sys
import re
import json
import errno


# FIXME: always set BUILD_WITH_XPU = ON in XPU repo
os.environ["BUILD_WITH_XPU"] = "ON"
os.environ["BUILD_STATIC_ONEMKL"] = "OFF"

PACKAGE_NAME = "intel_extension_for_pytorch"

# Define env values
ON_ENV_VAL = ["ON", "YES", "1", "Y"]
OFF_ENV_VAL = ["OFF", "NO", "0", "N"]
FULL_ENV_VAL = ON_ENV_VAL + OFF_ENV_VAL


# initialize variables for compilation
IS_LINUX = platform.system() == "Linux"
IS_DARWIN = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"

sanitize_value = None
sanitize_flag = False
sanitize_pattern = re.compile(r"--sanitize(=.+)?")
if "-s" in sys.argv:
    index = sys.argv.index("-s")
    if index + 1 < len(sys.argv) and not sys.argv[index + 1].startswith("-"):
        sanitize_value = sys.argv.pop(index + 1)
    sys.argv.pop(index)
    sanitize_flag = True
else:
    for i, arg in enumerate(sys.argv):
        if sanitize_pattern.match(arg):
            match = sanitize_pattern.match(arg)
            if match.group(1):
                sanitize_value = match.group(1)[1:]
            sys.argv.pop(i)
            sanitize_flag = True


@lru_cache(maxsize=128)
def _get_build_target():
    build_target = ""
    if len(sys.argv) > 1:
        if sys.argv[1] in ["build_clib", "bdist_cppsdk"]:
            build_target = "cppsdk"
        elif sys.argv[1] in ["clean"]:
            build_target = "clean"
        elif sys.argv[1] in ["develop"]:
            build_target = "develop"
        elif sys.argv[1] in ["bdist_wheel"]:
            build_target = "bdist_wheel"
        else:
            build_target = "python"
    return build_target


torch_install_prefix = None
if _get_build_target() == "cppsdk":
    torch_install_prefix = os.environ.get("LIBTORCH_PATH", None)
    if torch_install_prefix is None or not os.path.exists(torch_install_prefix):
        raise RuntimeError("Can not find libtorch from env LIBTORCH_PATH!")
    torch_install_prefix = os.path.abspath(torch_install_prefix)
elif _get_build_target() in ["develop", "python", "bdist_wheel"]:
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CppExtension
    except ImportError as e:
        raise RuntimeError("Fail to import torch!") from e


def _check_env_flag(name, default=""):
    return os.getenv(name, default).upper().strip() in ON_ENV_VAL


def get_build_type():
    return (
        "RelWithDebInfo"
        if _check_env_flag("REL_WITH_DEB_INFO")
        else "Debug" if _check_env_flag("DEBUG") else "Release"
    )


def get_build_aot():
    build_target = _get_build_target()
    aot_env = os.environ.get("TORCH_XPU_ARCH_LIST", "")

    # 1. If TORCH_XPU_ARCH_LIST is not set, we will use the arch list from torch-xpu-ops.
    # 2. If TORCH_XPU_ARCH_LIST is set to "", following torch-xpu-ops behavior, we will use the list from torch-xpu-ops.
    if not aot_env:
        return (
            ",".join(torch.xpu.get_arch_list())
            if build_target in ["develop", "python", "bdist_wheel"]
            else ""
        )
    # 3. If TORCH_XPU_ARCH_LIST is set to "none", following torch-xpu-ops behavior, we will not use AOT.
    if aot_env == "none":
        return ""
    # 4. If TORCH_XPU_ARCH_LIST is set to a specific arch list, we will use it.
    return aot_env


def create_if_not_exist(path_dir):
    if not os.path.exists(path_dir):
        try:
            Path(path_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise RuntimeError("Fail to create path {}".format(path_dir)) from exc


def get_version_num():
    versions = {}
    version_file = "version.txt"
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


def get_pytorch_install_dir():
    if _get_build_target() == "clean":
        return None
    if _get_build_target() == "cppsdk":
        return torch_install_prefix
    else:
        return os.path.dirname(os.path.abspath(torch.__file__))


pytorch_install_dir = get_pytorch_install_dir()


def _get_basekit_rt():
    with open("dependency_version.json", "r") as f:
        result = json.load(f)
        return result["basekit"]


def _build_installation_dependency():
    install_requires = []
    install_requires.append("psutil")
    install_requires.append("numpy")
    install_requires.append("packaging")
    str_os = platform.system().lower()
    if _check_env_flag("ENABLE_ONEAPI_INTEGRATION", default="OFF"):
        for key, value in _get_basekit_rt().items():
            if key in ["oneccl-devel", "impi-devel", "intel-pti"] and not IS_LINUX:
                continue
            if len(value[str_os].split(".")) != 3:
                continue
            install_requires.append(f"{key}=={value[str_os]}")
    return install_requires


def get_cmake_command():
    if IS_WINDOWS:
        return "cmake"
    if shutil.which("cmake3") is not None:
        return "cmake3"
    if shutil.which("cmake") is not None:
        return "cmake"
    else:
        raise RuntimeError("no cmake or cmake3 found")


def get_cpack_command():
    if IS_WINDOWS:
        return "cpack"
    if shutil.which("cpack3") is not None:
        return "cpack3"
    if shutil.which("cpack") is not None:
        return "cpack"
    else:
        raise RuntimeError("no cpack or cpack3 found")


def get_ipex_git_head_sha(base_dir):
    ipex_git_sha = "N/A"
    try:
        ipex_git_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=base_dir
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        pass
    return ipex_git_sha


def get_torch_git_head_sha():
    if _get_build_target() == "clean":
        return None
    if _get_build_target() == "cppsdk":
        libtorch_hash_file = os.path.join(torch_install_prefix, "build-hash")
        if not os.path.exists(libtorch_hash_file):
            raise RuntimeError(
                "can not find build-hash at {}".format(libtorch_hash_file)
            )
        with open(libtorch_hash_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.isalnum():
                    return line
        raise RuntimeError("can not get libtorch hash in {}".format(libtorch_hash_file))
    else:
        torch_git_sha = torch.version.git_version
        return torch_git_sha


def get_submodule_commit(base_dir, submodule_dir):
    if not os.path.isdir(submodule_dir):
        return ""
    try:
        return (
            subprocess.check_output(
                ["git", "submodule", "status", submodule_dir], cwd=base_dir
            )
            .decode("ascii")
            .strip()
            .split()[0]
        )
    except Exception:
        return "N/A"


def get_build_version(ipex_git_sha):
    ipex_version = os.getenv("IPEX_VERSION", get_version_num())
    if "IPEX_VERSION" not in os.environ:
        if _check_env_flag("IPEX_VERSIONED_BUILD", default="1"):
            try:
                ipex_version += "+git" + ipex_git_sha[:7]
            except Exception:
                pass
        else:
            pkg_type = "xpu" if _check_env_flag("BUILD_WITH_XPU") else "cpu"
            ipex_version += "+" + pkg_type
    return ipex_version


def write_buffer_to_file(file_path, buffer):
    create_if_not_exist(os.path.dirname(file_path))
    with open(file_path, "w") as f:
        f.write(buffer)
        f.close()


def get_code_fingerprint(ipex_build_version, ipex_git_sha, torch_git_sha, build_type):
    fingerprint = "{}_{}_{}_{}".format(
        ipex_build_version, ipex_git_sha, torch_git_sha, build_type
    )
    return fingerprint


def check_code_fingerprint_in_file(file_path, fingerprint):
    b_exist = os.path.isfile(file_path)
    if b_exist is False:
        return False

    with open(file_path) as file:
        # read all content of a file
        content = file.read()
        # check if string present in a file
        if fingerprint in content:
            return True
        else:
            return False


def create_version_files(
    base_dir,
    ipex_build_version,
    ipex_git_sha,
    torch_git_sha,
    gpu_onednn_sha,
    cpu_ideep_sha,
):
    print(
        "Building Intel Extension for PyTorch. Version: {}".format(ipex_build_version)
    )
    py_version_path = os.path.join(base_dir, PACKAGE_NAME, "_version.py")
    cpp_version_path = os.path.join(
        base_dir, PACKAGE_NAME, "..", "csrc", "utils", "version.h"
    )
    build_type_str = get_build_type()
    build_aot = get_build_aot()
    # Check code fingerprint to avoid non-modify rebuild.
    current_code_fingerprint = get_code_fingerprint(
        ipex_build_version, ipex_git_sha, torch_git_sha, build_type_str
    )

    b_same_fingerprint = check_code_fingerprint_in_file(
        py_version_path, current_code_fingerprint
    )
    if b_same_fingerprint is False:
        py_buffer = "# Autogenerated file, do not edit!\n"
        py_buffer += "# code fingerprint:\n"
        py_buffer += "# {}\n".format(current_code_fingerprint)
        py_buffer += '__version__ = "{}"\n'.format(ipex_build_version)
        py_buffer += '__ipex_gitrev__ = "{}"\n'.format(ipex_git_sha)
        py_buffer += '__torch_gitrev__ = "{}"\n'.format(
            "" if build_type_str == "Release" else torch_git_sha
        )
        py_buffer += '__gpu_onednn_gitrev__ = "{}"\n'.format(gpu_onednn_sha)
        py_buffer += '__cpu_ideep_gitrev__ = "{}"\n'.format(cpu_ideep_sha)
        py_buffer += '__build_type__ = "{}"\n'.format(build_type_str)
        py_buffer += '__build_aot__ = "{}"\n'.format(build_aot)

        write_buffer_to_file(py_version_path, py_buffer)

    b_same_fingerprint = check_code_fingerprint_in_file(
        cpp_version_path, current_code_fingerprint
    )
    if b_same_fingerprint is False:
        c_buffer = "// Autogenerated file, do not edit!\n"
        c_buffer += "// clang-format off\n"
        c_buffer += "// code fingerprint: {}\n".format(current_code_fingerprint)
        c_buffer += "// clang-format on\n\n"
        c_buffer += "#pragma once\n"
        c_buffer += "#include <string>\n\n"
        c_buffer += "namespace torch_ipex {\n\n"
        c_buffer += "const std::string __version__()\n"
        c_buffer += '{{ return "{}"; }}\n\n'.format(ipex_build_version)
        c_buffer += "const std::string __gitrev__()\n"
        c_buffer += '{{ return "{}"; }}\n\n'.format(ipex_git_sha)
        c_buffer += "const std::string __torch_gitrev__()\n"
        c_buffer += '{{ return "{}"; }}\n\n'.format(torch_git_sha)
        c_buffer += "const std::string __build_type__()\n"
        c_buffer += '{{ return "{}"; }}\n\n'.format(build_type_str)
        c_buffer += "const std::string __build_aot__()\n"
        c_buffer += '{{ return "{}"; }}\n\n'.format(build_aot)
        c_buffer += "}  // namespace torch_ipex\n"

        write_buffer_to_file(cpp_version_path, c_buffer)


def get_project_dir():
    project_root_dir = os.path.dirname(__file__)
    return os.path.abspath(project_root_dir)


def get_build_dir():
    return os.path.join(get_project_dir(), "build")


def get_build_type_dir():
    build_type_dir = os.path.join(get_build_dir(), get_build_type())
    create_if_not_exist(build_type_dir)
    return build_type_dir


def get_package_base_dir():
    return os.path.join(get_build_type_dir(), "packages")


def get_package_dir():
    return os.path.join(get_package_base_dir(), PACKAGE_NAME)


def get_package_lib_dir():
    package_lib_dir = os.path.join(get_package_dir(), "lib")
    create_if_not_exist(package_lib_dir)
    return package_lib_dir


def get_ipex_cpu_dir():
    cpu_root_dir = os.path.join(get_project_dir(), "csrc", "cpu")
    return os.path.abspath(cpu_root_dir)


def get_ipex_cpu_build_dir():
    cpu_build_dir = os.path.join(get_build_type_dir(), "csrc", "cpu")
    create_if_not_exist(cpu_build_dir)
    return cpu_build_dir


def get_xpu_project_dir():
    project_root_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(project_root_dir)


def get_xpu_project_build_dir():
    xpu_build_dir = os.path.join(get_build_type_dir(), "csrc", "gpu")
    create_if_not_exist(xpu_build_dir)
    return xpu_build_dir


def get_xpu_compliers():
    if shutil.which("icx") is None or shutil.which("icpx") is None:
        raise RuntimeError("Failed to find compiler path from OS PATH")
    if IS_WINDOWS:
        # Windows needs absolute path of compilers, otherwise it will find the
        # difference of the compiler name and the absolution path in the first
        # time of incremental build. Reconfiguration will lose all command line
        # cmake definitions.
        # https://discourse.cmake.org/t/configure-will-be-re-run-and-you-may-have-to-reset-some-variables/7325
        win_icx_realpath = shutil.which("icx.exe").replace("\\", "/")
        return win_icx_realpath, win_icx_realpath
    else:
        return "icx", "icpx"


def get_ipex_python_dir():
    project_root_dir = os.path.dirname(__file__)
    python_root_dir = os.path.join(project_root_dir, PACKAGE_NAME, "csrc")
    return os.path.abspath(python_root_dir)


def get_ipex_python_build_dir():
    python_build_dir = os.path.join(get_build_type_dir(), PACKAGE_NAME, "csrc")
    create_if_not_exist(python_build_dir)
    return python_build_dir


def get_ipex_cppsdk_build_dir():
    cppsdk_build_dir = os.path.join(get_build_type_dir(), "csrc", "cppsdk")
    create_if_not_exist(cppsdk_build_dir)
    return cppsdk_build_dir


base_dir = os.path.dirname(os.path.abspath(__file__))
# Generate version info (ipex.__version__)
torch_git_sha = get_torch_git_head_sha()
ipex_git_sha = get_ipex_git_head_sha(base_dir)
ipex_build_version = get_build_version(ipex_git_sha)
ipex_gpu_onednn_git_sha = get_submodule_commit(base_dir, "third_party/oneDNN")
ipex_cpu_ideep_git_sha = get_submodule_commit(base_dir, "third_party/ideep")
create_version_files(
    base_dir,
    ipex_build_version,
    ipex_git_sha,
    torch_git_sha,
    ipex_gpu_onednn_git_sha,
    ipex_cpu_ideep_git_sha,
)


def auto_format_python_code():
    script_path = os.path.join(base_dir, "scripts", "tools", "setup", "flake8.py")
    script_cmd = ["python"]
    script_cmd.append(script_path)
    subprocess.call(
        script_cmd,
        cwd=base_dir,
    )


# global setup modules
class IPEXClean(distutils.command.clean.clean, object):
    def run(self):
        import glob

        with open(".gitignore", "r") as f:
            ignores = f.read()
            pat = re.compile(r"^#( BEGIN NOT-CLEAN-FILES )?")
            for wildcard in filter(None, ignores.split("\n")):
                match = pat.match(wildcard)
                if match:
                    if match.group(1):
                        # Marker is found and stop reading .gitignore.
                        break
                # Ignore lines which begin with '#'.
                else:
                    for filename in glob.glob(wildcard):
                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


def get_cpp_test_dir():
    project_root_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(project_root_dir, "tests", "cpu", "cpp")


def get_cpp_test_build_dir():
    cpp_test_build_dir = os.path.join(get_build_type_dir(), "tests", "cpu", "cpp")
    create_if_not_exist(cpp_test_build_dir)
    return cpp_test_build_dir


# PyBind11 ABI handling is internal to PyBind11; this will be removed after PyTorch 2.9.0"
# Ref: https://github.com/pytorch/pytorch/blob/d5e0f4202ba14632e4d14862ace096609e763462/torch/utils/cpp_extension.py#L1709-L1711
def get_pybind11_abi_compiler_flags():
    return []


def _gen_build_cfg_from_cmake(
    cmake_exec, project_root_dir, cmake_args, build_dir, build_env, use_ninja=False
):
    if IS_WINDOWS:
        if use_ninja:
            check_call(
                [cmake_exec, project_root_dir, "-GNinja"] + cmake_args,
                cwd=build_dir,
                env=build_env,
            )
        else:
            # using MSVC generator
            check_call(
                [
                    cmake_exec,
                    project_root_dir,
                    "-G Visual Studio 17 2022",
                    "-T Intel C++ Compiler 2023",
                ]
                + cmake_args,
                cwd=build_dir,
                env=build_env,
            )
    else:
        # Linux
        check_call(
            [cmake_exec, project_root_dir] + cmake_args, cwd=build_dir, env=build_env
        )


def _build_project(build_args, build_dir, build_env, use_ninja=False):
    if IS_WINDOWS:
        if use_ninja:
            check_call(["ninja"] + build_args, cwd=build_dir, env=build_env)
        else:
            # Current Windows MSVC needs manual build
            pass
    else:
        # Linux
        if use_ninja:
            check_call(["ninja"] + build_args, cwd=build_dir, env=build_env)
        else:
            check_call(["make"] + build_args, cwd=build_dir, env=build_env)


def define_build_options(args, **kwargs):
    for key, value in sorted(kwargs.items()):
        if value is not None:
            args.append("-D{}={}".format(key, value))


class IPEXCPPLibBuild(build_clib, object):
    def run(self):
        self.build_lib = os.path.relpath(get_package_dir())
        self.build_temp = os.path.relpath(get_build_type_dir())

        auto_format_python_code()

        cmake_exec = get_cmake_command()
        if cmake_exec is None:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )
        self.cmake = cmake_exec

        project_root_dir = get_project_dir()
        build_type_dir = get_build_type_dir()

        ipex_python_dir = get_ipex_python_dir()
        ipex_python_build_dir = get_ipex_python_build_dir()

        ipex_cpu_dir = get_ipex_cpu_dir()
        ipex_cpu_build_dir = get_ipex_cpu_build_dir()

        ipex_xpu_dir = get_xpu_project_dir()
        ipex_xpu_build_dir = get_xpu_project_build_dir()

        ipex_cppsdk_build_dir = get_ipex_cppsdk_build_dir()
        cpack_out_file = os.path.abspath(
            os.path.join(build_type_dir, "IPEXCPackConfig.cmake")
        )
        self_extract_script = "gen_self_extract.sh"

        if _get_build_target() == "cppsdk":
            cmake_prefix_path = torch_install_prefix
        else:
            cmake_prefix_path = torch.utils.cmake_prefix_path

        build_option_common = {
            "CMAKE_BUILD_TYPE": get_build_type(),
            "CMAKE_INSTALL_INCLUDEDIR": "include",
            "CMAKE_INSTALL_LIBDIR": "lib",
            "CMAKE_PREFIX_PATH": cmake_prefix_path,
            "CMAKE_INSTALL_PREFIX": os.path.abspath(get_package_dir()),
            "CMAKE_PROJECT_VERSION": get_version_num(),
            "PYTHON_PLATFORM_INFO": platform.platform(),
            "PYTHON_INCLUDE_DIR": sysconfig.get_path("include"),
            "PYTHON_EXECUTABLE": sys.executable,
            "PYTHON_BUILD_VERSION": sys.version.replace("\n", ""),
            "PYTHON_VERSION": platform.python_version(),
            "IPEX_PROJ_NAME": PACKAGE_NAME,
            "LIBIPEX_GITREV": ipex_git_sha,
            "LIBIPEX_VERSION": ipex_build_version,
        }

        build_with_cpu = False if IS_WINDOWS else True  # Default ON
        build_with_xpu = False  # Default OFF

        use_ninja = False
        # Windows uses Ninja as default generator
        if IS_WINDOWS:

            def get_thread_free_suffix():
                return "t" if sysconfig.get_config_var("Py_GIL_DISABLED") == 1 else ""

            use_ninja = True
            build_option_common["PYTHON_LIBRARIES"] = os.path.join(
                sys.base_prefix,
                "libs",
                f"python{sys.version_info.major}{sys.version_info.minor}{get_thread_free_suffix()}.lib",
            )
        sequential_build = False

        cmake_common_args = []
        my_env = os.environ.copy()

        for var, val in my_env.items():
            val = val.strip()
            if var.startswith(("BUILD_", "USE_", "CMAKE_")):
                if var == "CMAKE_PREFIX_PATH":
                    # XXX: Do NOT overwrite CMAKE_PREFIX_PATH. Append into the list, instead!
                    build_option_common[var] = ";".join(
                        [build_option_common[var], val.replace(":", ";")]
                    )
                    continue
                if var == "USE_NINJA" and val.upper() in ON_ENV_VAL:
                    use_ninja = True
                    cmake_common_args.append("-GNinja")
                    continue
                if IS_WINDOWS and var == "USE_MSVC" and val.upper() in ON_ENV_VAL:
                    use_ninja = False
                    continue
                if var == "BUILD_STATS" and val.upper() in ON_ENV_VAL:
                    sequential_build = True
                    # fall through
                if var == "BUILD_WITH_XPU" and val.upper() in ON_ENV_VAL:
                    build_with_xpu = True
                    # fall through
                if var == "BUILD_WITH_CPU" and val.upper() in OFF_ENV_VAL:
                    build_with_cpu = False
                    # fall through
                build_option_common[var] = val

        if IS_WINDOWS and build_with_cpu:
            raise NotImplementedError("IPEX CPU don't support build on Windows.")

        nproc = min(int(os.environ.get("MAX_JOBS", os.cpu_count())), os.cpu_count())
        if sequential_build:
            nproc = 1
            print("WARNING: Practice as sequential build with single process !")

        build_args = ["-j", str(nproc), "install"]
        if _check_env_flag("VERBOSE") and use_ninja:
            build_args.append("-v")

        if build_with_xpu:
            # Generate cmake for XPU module:
            if os.path.isdir(ipex_xpu_dir) is False:
                raise RuntimeError(
                    "It maybe CPU only branch, and it is not contains XPU code."
                )

            gpu_cc, gpu_cxx = get_xpu_compliers()
            build_option_gpu = {
                **build_option_common,
                "BUILD_MODULE_TYPE": "GPU",
                "CMAKE_C_COMPILER": gpu_cc,
                "CMAKE_CXX_COMPILER": gpu_cxx,
                # To skip oneMKL's dependence on TBB if oneAPI version >= 2024.0
                # IPEX GPU will not use TBB-related MKL primitives.
                "MKL_SYCL_THREADING": "intel_thread",
                "USE_AOT_DEVLIST": get_build_aot(),
            }

            if get_build_type() == "Debug":
                build_option_gpu = {
                    **build_option_gpu,
                    "BUILD_SEPARATE_OPS": "ON",
                    "USE_SYCL_ASSERT": "ON",
                    "USE_ITT_ANNOTATION": "ON",
                }
            if sanitize_flag:
                if sanitize_value is not None:
                    sanitize_options = sanitize_value
                else:
                    sanitize_options = "address"  # default value
                build_option_gpu = {
                    **build_option_gpu,
                    "BUILD_WITH_SANITIZER": sanitize_options,
                }
            cmake_args_gpu = cmake_common_args[:]
            define_build_options(cmake_args_gpu, **build_option_gpu)
            my_env_local = my_env.copy()
            ldflags = ""
            if "LDFLAGS" in my_env_local:
                ldflags = f"{my_env_local['LDFLAGS']} "
            my_env_local["LDFLAGS"] = f"{ldflags}-Wl,--no-as-needed"
            if "IPEX_GPU_EXTRA_BUILD_OPTION" in my_env_local:
                my_env_local["LDFLAGS"] = (
                    f"{my_env_local['IPEX_GPU_EXTRA_BUILD_OPTION']} {my_env_local['LDFLAGS']}"
                )
            _gen_build_cfg_from_cmake(
                cmake_exec,
                project_root_dir,
                cmake_args_gpu,
                ipex_xpu_build_dir,
                my_env_local,
                use_ninja,
            )

        if build_with_cpu:
            # Generate cmake for CPU module:
            build_option_cpu = {
                **build_option_common,
                "BUILD_MODULE_TYPE": "CPU",
            }
            if _get_build_target() in ["develop", "python"]:
                build_option_cpu["BUILD_CPU_WITH_ONECCL"] = "OFF"
                build_option_cpu["USE_SHM"] = "ON"
                build_option_cpu["ENABLE_MPI_TESTS"] = "OFF"
                build_option_cpu["BUILD_REG_TESTS"] = "OFF"
                build_option_cpu["ENABLE_MPI_TESTS"] = "OFF"

            cmake_args_cpu = cmake_common_args[:]
            define_build_options(cmake_args_cpu, **build_option_cpu)
            _gen_build_cfg_from_cmake(
                cmake_exec, project_root_dir, cmake_args_cpu, ipex_cpu_build_dir, my_env
            )

            # Generate cmake for the CPP UT
            build_option_cpp_test = {
                **build_option_common,
                "PROJECT_DIR": project_root_dir,
                "PYTORCH_INSTALL_DIR": pytorch_install_dir,
                "CPP_TEST_BUILD_DIR": get_cpp_test_build_dir(),
            }

            cmake_args_cpp_test = cmake_common_args[:]
            define_build_options(cmake_args_cpp_test, **build_option_cpp_test)
            _gen_build_cfg_from_cmake(
                cmake_exec,
                get_cpp_test_dir(),
                cmake_args_cpp_test,
                get_cpp_test_build_dir(),
                my_env,
            )

        if _get_build_target() in ["develop", "python", "bdist_wheel"]:
            # Generate cmake for common python module:
            build_option_python = {
                **build_option_common,
                "BUILD_MODULE_TYPE": "PYTHON",
                "PYBIND11_CL_FLAGS": get_pybind11_abi_compiler_flags(),
            }

            cmake_args_python = cmake_common_args[:]
            define_build_options(cmake_args_python, **build_option_python)
            _gen_build_cfg_from_cmake(
                cmake_exec,
                project_root_dir,
                cmake_args_python,
                ipex_python_build_dir,
                my_env,
                use_ninja,
            )

        elif _get_build_target() == "cppsdk":
            # Generate cmake for CPPSDK package:
            build_option_cppsdk = {
                **build_option_common,
                "BUILD_MODULE_TYPE": "CPPSDK",
                "CPACK_CONFIG_FILE": cpack_out_file,
                "CPACK_OUTPUT_DIR": build_type_dir,
                "LIBIPEX_GEN_SCRIPT": self_extract_script,
            }

            cmake_args_cppsdk = cmake_common_args[:]
            define_build_options(cmake_args_cppsdk, **build_option_cppsdk)
            _gen_build_cfg_from_cmake(
                cmake_exec,
                project_root_dir,
                cmake_args_cppsdk,
                ipex_cppsdk_build_dir,
                my_env,
                use_ninja,
            )

        if build_with_xpu:
            # Build XPU module:
            _build_project(build_args, ipex_xpu_build_dir, my_env, use_ninja)

        if build_with_cpu:
            # Build CPU module:
            _build_project(build_args, ipex_cpu_build_dir, my_env, use_ninja)

            # Build the CPP UT
            _build_project(build_args, get_cpp_test_build_dir(), my_env, use_ninja)

        if _get_build_target() in ["develop", "python", "bdist_wheel"]:
            # Build common python module:
            _build_project(build_args, ipex_python_build_dir, my_env, use_ninja)

        elif _get_build_target() == "cppsdk":
            # Build CPPSDK package:
            _build_project(build_args, ipex_cppsdk_build_dir, my_env, use_ninja)
            cpack_exec = get_cpack_command()
            check_call([cpack_exec, "--config", cpack_out_file])
            gen_script_path = os.path.abspath(
                os.path.join(build_type_dir, self_extract_script)
            )
            if not os.path.isfile(gen_script_path):
                raise "Cannot find script to generate self-extract package in {}".format(
                    gen_script_path
                )
            check_call(gen_script_path, shell=True)

        # Copy the export library, header and cmake file to root/intel_extension_for_pytorch dir.
        # It is only copied in "develop" mode, which can save disk space in "install" mode.
        if _get_build_target() == "develop":
            ret = get_src_lib_and_dst()
            for src, dst in ret:
                self.copy_file(src, dst)


def get_src_lib_and_dst():
    ret = []
    generated_cpp_files = glob.glob(
        os.path.join(get_package_base_dir(), PACKAGE_NAME, "lib", "**/*.so"),
        recursive=True,
    )
    generated_cpp_files.extend(
        glob.glob(
            os.path.join(get_package_base_dir(), PACKAGE_NAME, "bin", "**/*.dll"),
            recursive=True,
        )
    )
    generated_cpp_files.extend(
        glob.glob(
            os.path.join(get_package_base_dir(), PACKAGE_NAME, "lib", "**/*.lib"),
            recursive=True,
        )
    )
    generated_cpp_files.extend(
        glob.glob(
            os.path.join(get_package_base_dir(), PACKAGE_NAME, "include", "**/*.h"),
            recursive=True,
        )
    )
    generated_cpp_files.extend(
        glob.glob(
            os.path.join(get_package_base_dir(), PACKAGE_NAME, "share", "**/*.cmake"),
            recursive=True,
        )
    )
    for src in generated_cpp_files:
        dst = os.path.join(
            get_project_dir(),
            PACKAGE_NAME,
            os.path.relpath(src, os.path.join(get_package_base_dir(), PACKAGE_NAME)),
        )
        dst_path = Path(dst)
        if not dst_path.parent.exists():
            Path(dst_path.parent).mkdir(parents=True, exist_ok=True)
        ret.append((src, dst))
    return ret


def get_src_py_and_yaml_and_dst():
    ret = []
    generated_python_and_yaml_files = glob.glob(
        os.path.join(get_project_dir(), PACKAGE_NAME, "**/*.py"), recursive=True
    )
    for src in generated_python_and_yaml_files:
        dst = os.path.join(
            get_package_base_dir(),
            PACKAGE_NAME,
            os.path.relpath(src, os.path.join(get_project_dir(), PACKAGE_NAME)),
        )
        dst_path = Path(dst)
        if not dst_path.parent.exists():
            Path(dst_path.parent).mkdir(parents=True, exist_ok=True)
        ret.append((src, dst))
    return ret


# python specific setup modules
class IPEXEggInfoBuild(egg_info, object):
    def finalize_options(self):
        super(IPEXEggInfoBuild, self).finalize_options()


class IPEXInstallCmd(install, object):
    def run(self):
        self.build_lib = os.path.relpath(get_package_base_dir())
        return super(IPEXInstallCmd, self).run()


class IPEXPythonPackageBuild(build_py, object):
    def run(self) -> None:
        ret = get_src_py_and_yaml_and_dst()
        for src, dst in ret:
            self.copy_file(src, dst)
        super(IPEXPythonPackageBuild, self).finalize_options()


def make_relative_rpath(path):
    if IS_DARWIN:
        return "-Wl,-rpath,@loader_path/" + path
    elif IS_WINDOWS:
        return path
    else:
        return "-Wl,-rpath,$ORIGIN/" + path


def pyi_module():
    main_libraries = ["intel-ext-pt-python"]
    main_sources = [os.path.join(PACKAGE_NAME, "csrc", "_C.cpp")]

    include_dirs = [
        os.path.realpath("."),
        os.path.realpath(os.path.join(PACKAGE_NAME, "csrc")),
        os.path.join(pytorch_install_dir, "include"),
        os.path.join(pytorch_install_dir, "include", "torch", "csrc", "api", "include"),
    ]

    if not IS_WINDOWS:
        library_dirs = ["lib", os.path.join(pytorch_install_dir, "lib")]

        extra_compile_args = [
            "-Wall",
            "-Wextra",
            "-Wno-strict-overflow",
            "-Wno-unused-parameter",
            "-Wno-missing-field-initializers",
            "-Wno-write-strings",
            "-Wno-unknown-pragmas",
            # This is required for Python 2 declarations that are deprecated in 3.
            "-Wno-deprecated-declarations",
            # Python 2.6 requires -fno-strict-aliasing, see
            # http://legacy.python.org/dev/peps/pep-3123/
            # We also depend on it in our code (even Python 3).
            "-fno-strict-aliasing",
            # Clang has an unfixed bug leading to spurious missing
            # braces warnings, see
            # https://bugs.llvm.org/show_bug.cgi?id=21629
            "-Wno-missing-braces",
        ]

        C_ext = CppExtension(
            "{}._C".format(PACKAGE_NAME),
            libraries=main_libraries,
            sources=main_sources,
            language="c++",
            extra_compile_args=extra_compile_args,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_link_args=[make_relative_rpath("lib")],
        )
    else:
        library_dirs = ["bin", os.path.join(pytorch_install_dir, "lib")]
        extra_link_args = ["/NODEFAULTLIB:LIBCMT.LIB"]
        # /MD links against DLL runtime
        # and matches the flags set for protobuf and ONNX
        # /EHsc is about standard C++ exception handling
        # /DNOMINMAX removes builtin min/max functions
        # /wdXXXX disables warning no. XXXX
        extra_compile_args = [
            "/MD",
            "/EHsc",
            "/DNOMINMAX",
            "/wd4267",
            "/wd4251",
            "/wd4522",
            "/wd4522",
            "/wd4838",
            "/wd4305",
            "/wd4244",
            "/wd4190",
            "/wd4101",
            "/wd4996",
            "/wd4275",
        ]
        C_ext = CppExtension(
            "{}._C".format(PACKAGE_NAME),
            libraries=main_libraries,
            sources=main_sources,
            language="c++",
            extra_compile_args=extra_compile_args,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_link_args=extra_link_args,
        )
    return C_ext


def pyi_isa_help_module():
    main_libraries = []
    main_sources = [
        os.path.join(PACKAGE_NAME, "csrc", "_isa_help_main.cpp"),
        os.path.join(PACKAGE_NAME, "csrc", "cpu", "isa_help", "isa_help.cpp"),
        os.path.join("csrc", "cpu", "isa", "cpu_feature.cpp"),
    ]

    include_dirs = [
        os.path.realpath("."),
        os.path.realpath(os.path.join("csrc", "cpu", "isa")),
        os.path.realpath(os.path.join("csrc", "include")),
        os.path.realpath(os.path.join(PACKAGE_NAME, "csrc")),
        os.path.join(pytorch_install_dir, "include"),
        os.path.join(pytorch_install_dir, "include", "torch", "csrc", "api", "include"),
    ]

    if not IS_WINDOWS:
        library_dirs = ["lib", os.path.join(pytorch_install_dir, "lib")]

        extra_compile_args = [
            "-Wall",
            "-Wextra",
            "-Wno-strict-overflow",
            "-Wno-unused-parameter",
            "-Wno-missing-field-initializers",
            "-Wno-write-strings",
            "-Wno-unknown-pragmas",
            # This is required for Python 2 declarations that are deprecated in 3.
            "-Wno-deprecated-declarations",
            # Python 2.6 requires -fno-strict-aliasing, see
            # http://legacy.python.org/dev/peps/pep-3123/
            # We also depend on it in our code (even Python 3).
            "-fno-strict-aliasing",
            # Clang has an unfixed bug leading to spurious missing
            # braces warnings, see
            # https://bugs.llvm.org/show_bug.cgi?id=21629
            "-Wno-missing-braces",
        ]

        C_ext = CppExtension(
            "{}._isa_help".format(PACKAGE_NAME),
            libraries=main_libraries,
            sources=main_sources,
            language="c++",
            extra_compile_args=extra_compile_args,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_link_args=[make_relative_rpath("lib")],
        )
    else:
        library_dirs = ["bin", os.path.join(pytorch_install_dir, "lib")]
        extra_link_args = ["/NODEFAULTLIB:LIBCMT.LIB"]
        # /MD links against DLL runtime
        # and matches the flags set for protobuf and ONNX
        # /EHsc is about standard C++ exception handling
        # /DNOMINMAX removes builtin min/max functions
        # /wdXXXX disables warning no. XXXX
        extra_compile_args = [
            "/MD",
            "/EHsc",
            "/DNOMINMAX",
            "/wd4267",
            "/wd4251",
            "/wd4522",
            "/wd4522",
            "/wd4838",
            "/wd4305",
            "/wd4244",
            "/wd4190",
            "/wd4101",
            "/wd4996",
            "/wd4275",
            "/DBUILD_IPEX_MAIN_LIB",
        ]
        C_ext = CppExtension(
            "{}._isa_help".format(PACKAGE_NAME),
            libraries=main_libraries,
            sources=main_sources,
            language="c++",
            extra_compile_args=extra_compile_args,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_link_args=extra_link_args,
        )
    return C_ext


ext_modules = []
cmdclass = {
    "build_clib": IPEXCPPLibBuild,
    "bdist_cppsdk": IPEXCPPLibBuild,
    "clean": IPEXClean,
}


def fill_python_target_cmd(cmdclass, ext_modules):
    class IPEXExtBuild(BuildExtension):
        def run(self):
            self.run_command("build_clib")

            self.build_lib = os.path.relpath(get_package_base_dir())
            self.build_temp = os.path.relpath(get_build_type_dir())
            self.library_dirs.append(os.path.relpath(get_package_lib_dir()))
            super(IPEXExtBuild, self).run()

    cmdclass["build_ext"] = IPEXExtBuild
    cmdclass["build_py"] = IPEXPythonPackageBuild
    cmdclass["egg_info"] = IPEXEggInfoBuild
    cmdclass["install"] = IPEXInstallCmd
    ext_modules.append(pyi_module())
    ext_modules.append(pyi_isa_help_module())


if _get_build_target() in ["develop", "python", "bdist_wheel"]:
    fill_python_target_cmd(cmdclass, ext_modules)


long_description = ""
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

entry_points = {
    "console_scripts": [
        "ipexrun = {}.launcher:main".format(PACKAGE_NAME),
    ],
}


setup(
    name=PACKAGE_NAME,
    version=ipex_build_version,
    description="Intel® Extension for PyTorch*",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/intel/intel-extension-for-pytorch",
    author="Intel Corp.",
    install_requires=_build_installation_dependency(),
    packages=[PACKAGE_NAME],
    package_data={PACKAGE_NAME: ["*.so", "lib/*.so", "bin/*.dll", "lib/*.lib"]},
    zip_safe=False,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    entry_points=entry_points,
    license="https://www.apache.org/licenses/LICENSE-2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
    ],
)
