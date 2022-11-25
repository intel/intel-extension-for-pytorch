#!/usr/bin/env python

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
# Environment variables for feature toggles:
#
#   IPEX_DISP_OP=1
#     output the extension operators name for debug purpose
#
# Environment variables we respect (these environment variables are
# conventional and are often understood/set by other software.)
#
#   TORCH_VERSION
#     specify the PyTorch version to depend on
#
#   TORCH_IPEX_VERSION
#     specify the extension version literal
#

##############################################################
# XPU Build options:
# USE_ONEMKL            - to use oneMKL in operators
# USE_CHANNELS_LAST_1D  - to use channels last 1d feature
# USE_PERSIST_STREAM    - to use persistent oneDNN stream
# USE_PRIMITIVE_CACHE   - to Cache oneDNN primitives by framework
# USE_QUEUE_BARRIER     - to use queue submit_barrier API
# USE_SCRATCHPAD_MODE   - to trun on oneDNN scratchpad user mode
# USE_MULTI_CONTEXT     - to create DPC++ runtime context per device
# USE_AOT_DEVLIST       - to set device list for AOT build option, for example, bdw,tgl,ats,..."
# USE_SYCL_ASSERT       - to enable assert in sycl kernel
# BUILD_STATS           - to count statistics for each component during build process
# BUILD_BY_PER_KERNEL   - to build by DPC++ per_kernel option (exclusive with USE_AOT_DEVLIST)
# BUILD_STRIPPED_BIN    - to strip all symbols after build
# BUILD_SEPARATE_OPS    - to build each operator in separate library
# BUILD_SIMPLE_TRACE    - to build simple trace for each registered operator
# BUILD_OPT_LEVEL       - to add build option -Ox, accept values: 0/1
# BUILD_NO_CLANGFORMAT  - to build without force clang-format
# BUILD_INTERNAL_DEBUG  - to build internal debug code path
#
##############################################################

from __future__ import print_function
from distutils.command.build_py import build_py
from distutils.command.install import install
from distutils.cmd import Command
import pkg_resources
from setuptools.command.build_clib import build_clib
from setuptools.command.egg_info import egg_info

from subprocess import check_call, check_output
from setuptools import setup, distutils

import torch
import sysconfig
import distutils.ccompiler
import distutils.command.clean
import multiprocessing
import multiprocessing.pool
import os
import glob
import platform
import shutil
import subprocess
import sys
import copy
from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, CppExtension
import re
import errno

#FIXME: always set BUILD_WITH_XPU = ON in XPU repo
os.environ['BUILD_WITH_XPU'] = 'ON'

# Define env values
ON_ENV_VAL = ['ON', 'YES', '1', 'Y']
OFF_ENV_VAL = ['OFF', 'NO', '0', 'N']
FULL_ENV_VAL = ON_ENV_VAL + OFF_ENV_VAL

# initialize variables for compilation
IS_LINUX    = (platform.system() == 'Linux')
IS_DARWIN   = (platform.system() == 'Darwin')
IS_WINDOWS  = (platform.system() == 'Windows')


try:
    from packaging import version as pkg_ver
except Exception:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'packaging'])
    from packaging import version as pkg_ver


def _check_env_flag(name, default=''):
    return os.getenv(name, default).upper() in ON_ENV_VAL


def get_build_type():
    return 'RelWithDebInfo' if _check_env_flag('REL_WITH_DEB_INFO') else 'Debug' if _check_env_flag('DEBUG') else 'Release'


def create_if_not_exist(path_dir):
    if not os.path.exists(path_dir):
        try:
            Path(path_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise RuntimeError("Fail to create path {}".format(path_dir))


def get_version_num():
    versions = {}
    version_file = 'version.txt'
    version_lines = open(version_file, 'r').readlines()
    for line in version_lines:
        key, value = line.strip().split(' ')
        versions[key] = value
    for v in ('VERSION_MAJOR', 'VERSION_MINOR', 'VERSION_PATCH'):
        if v not in versions:
            print("ERROR:", v, "is not found in", version_file)
            sys.exit(1)
    version = versions['VERSION_MAJOR'] + '.' + versions['VERSION_MINOR'] + '.' + versions['VERSION_PATCH']
    return version


def gen_ipex_version_string():
    pkg_type = 'xpu' if _check_env_flag('BUILD_WITH_XPU') else 'cpu'
    return "{}+{}".format(get_version_num(), pkg_type)


PACKAGE_NAME = "intel_extension_for_pytorch"
PYTHON_VERSION = sys.version_info
TORCH_IPEX_VERSION = gen_ipex_version_string()


# build build_target
def parser_build_args():
    pytorch_install_dir = ''
    USE_CXX11_ABI = 0
    build_target = ''
    if len(sys.argv) > 1:
        if sys.argv[1] in ['build_clib', 'bdist_cppsdk']:
            build_target = 'cppsdk'
            # FIXME: Why we cannot find it by ourselves?
            if len(sys.argv) != 3:
                raise RuntimeError("""Please set path of libtorch directory if "build_clib" or "bdist_cppsdk" is applied.\n
                        Usage: python setup.py [build_clib|bdist_cppsdk] <libtorch_path>""")
            pytorch_install_dir = sys.argv[2]
            if pytorch_install_dir.startswith('.'):
                pytorch_install_dir = os.path.join(os.getcwd(), pytorch_install_dir)
            sys.argv.pop()

            if not os.path.isfile(os.path.join(pytorch_install_dir, 'build-version')):
                raise RuntimeError('{} doestn\'t seem to be a valid libtorch directory.'.format(pytorch_install_dir))

            # FIXME: It's ugly to use grep to fetch the flag, right?
            # Or, there should be API in libtorch to get the flag? We can have a tiny executable to do that?
            out = subprocess.check_output(['grep', 'GLIBCXX_USE_CXX11_ABI', \
                    os.path.join(pytorch_install_dir, 'share', 'cmake', 'Torch', 'TorchConfig.cmake')]).decode('ascii').strip()
            if out == '':
                raise RuntimeError('Unable to get GLIBCXX_USE_CXX11_ABI setting from libtorch: 1')
            matches = re.match('.*\"-D_GLIBCXX_USE_CXX11_ABI=(\d)\".*', out)
            if matches:
                USE_CXX11_ABI = int(matches.groups()[0])
            else:
                raise RuntimeError('Unable to get GLIBCXX_USE_CXX11_ABI setting from libtorch: 2')
        elif sys.argv[1] in ['clean']:
            build_target = 'clean'
        else:
            build_target = 'python'

    pytorch_install_dir = os.path.dirname(os.path.abspath(torch.__file__))
    USE_CXX11_ABI = torch._C._GLIBCXX_USE_CXX11_ABI

    return pytorch_install_dir, USE_CXX11_ABI, build_target


pytorch_install_dir, USE_CXX11_ABI, build_target = parser_build_args()


# configure for MKL
def get_mkl_config():
    mkl_install_dir = os.getenv('MKLROOT', '')
    if build_target != 'clean':
        if mkl_install_dir:
            mkl_header = glob.glob(f'{mkl_install_dir}/include/**/mkl_version.h', recursive = True)
            if len(mkl_header) == 0:
                raise RuntimeError(f'{mkl_install_dir} doesn\'t seem to be a valid MKL library directory.\n{" ":14}mkl_version.h not found.')
            else:
                mkl_header = mkl_header[0]
                mkl_major = 0
                mkl_minor = 0
                mkl_patch = 0
                with open(mkl_header) as fp:
                    for line in fp:
                        matches = re.match('#define __INTEL_MKL__ +(\d+)', line.strip())
                        if matches:
                            mkl_major = int(matches.groups()[0])
                        matches = re.match('#define __INTEL_MKL_MINOR__ +(\d+)', line.strip())
                        if matches:
                            mkl_minor = int(matches.groups()[0])
                        matches = re.match('#define __INTEL_MKL_UPDATE__ +(\d+)', line.strip())
                        if matches:
                            mkl_patch = int(matches.groups()[0])
                mkl_version = f'{mkl_major}.{mkl_minor}.{mkl_patch}'
                if pkg_ver.parse(mkl_version) < pkg_ver.parse('2021.0.0'):
                    raise RuntimeError(f'MKL version({mkl_version}) is not supported. Please use MKL later than 2021.0.0.')
                mkl_library = glob.glob(f'{mkl_install_dir}/lib/**/libmkl_core.a', recursive = True)
                if len(mkl_library) == 0:
                    raise RuntimeError(f'libmkl_core.a not found in {mkl_install_dir}/lib/intel64.')
        if not mkl_install_dir:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'mkl-include>=2021.0.0'])
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', 'mkl-static>=2021.0.0'])
            mkl_install_dir = os.path.abspath(os.path.join(os.path.dirname(sys.executable), ".."))

    # Find the oneMKL library path
    mkl_lib_path = mkl_install_dir + "/lib/"
    mkl_include_path = mkl_install_dir + "/include/"

    return mkl_install_dir, mkl_include_path, mkl_lib_path


# FIXME: MKL path should be handled in CMake, instead of here
# mkl_install_dir, mkl_include_path, mkl_lib_path = get_mkl_config()


def _build_installation_dependency():
    install_requires = []
    install_requires.append('psutil')
    install_requires.append('numpy')
    return install_requires

    # Disable PyTorch wheel binding temporarily
    #TORCH_URL = 'torch @ https://download.pytorch.org/whl/cpu/torch-{0}%2Bcpu-cp{1}{2}-cp{1}{2}-linux_x86_64.whl'.format(TORCH_VERSION, PYTHON_VERSION.major, PYTHON_VERSION.minor)
    #if IS_DARWIN:
    #    TORCH_URL = 'torch=={}'.format(TORCH_VERSION)
    #else:
    #    OS_VER = 'linux_x86_64'
    #    if IS_WINDOWS:
    #        TORCH_URL = 'torch @ https://download.pytorch.org/whl/cpu/torch-{0}%2Bcpu-cp{1}{2}-cp{1}{2}-win_amd64.whl'.format(TORCH_VERSION, PYTHON_VERSION.major, PYTHON_VERSION.minor)
    #        OS_VER = 'win_amd64'

    #    try:
    #        fp = urllib.request.urlopen('https://download.pytorch.org/whl/torch_stable.html', timeout=30)
    #        cont_bytes = fp.read()
    #        cont = cont_bytes.decode('utf8').replace('\n', '')
    #        fp.close()

    #        lines = re.split(r'<br>', cont)

    #        for line in lines:
    #            matches = re.match('<a href="(cpu\/torch-{0}.*cp{1}{2}.*{3}.*)">(.*)<\/a>'.format(TORCH_VERSION, PYTHON_VERSION.major, PYTHON_VERSION.minor, OS_VER), line)
    #            if matches and len(matches.groups()) == 2:
    #                TORCH_URL = 'torch @ https://download.pytorch.org/whl/{}'.format(matches.group(2))
    #                break
    #    except Exception:
    #        pass

    #install_requires.append(TORCH_URL)
    #return install_requires


def get_cmake_command():
    def _get_version(cmd):
        for line in check_output([cmd, '--version']).decode('utf-8').split('\n'):
            if 'version' in line:
                return pkg_ver.parse(line.strip().split(' ')[2])
        raise RuntimeError('no version found')
    "Returns cmake command."
    cmake_command = 'cmake'
    if platform.system() == 'Windows':
        return cmake_command
    cmake3 = shutil.which('cmake3')
    cmake = shutil.which('cmake')
    if cmake3 is not None and _get_version(cmake3) >= pkg_ver.parse("3.13.0"):
        cmake_command = 'cmake3'
        return cmake_command
    elif cmake is not None and _get_version(cmake) >= pkg_ver.parse("3.13.0"):
        return cmake_command
    else:
        raise RuntimeError('no cmake or cmake3 with version >= 3.13.0 found')


def get_git_head_sha(base_dir):
    ipex_git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=base_dir).decode('ascii').strip()
    torch_git_sha = torch.version.git_version
    return ipex_git_sha, torch_git_sha


def get_submodule_commit(base_dir, submodule_dir):
    if not os.path.isdir(submodule_dir):
        return ''
    return subprocess.check_output(
        ['git', 'submodule', 'status', submodule_dir], cwd=base_dir).decode('ascii').strip().split()[0]


def get_build_version(ipex_git_sha):
    ipex_version = os.getenv('TORCH_IPEX_VERSION', TORCH_IPEX_VERSION)
    if _check_env_flag('VERSIONED_IPEX_BUILD', default='0'):
        try:
            ipex_version += '+' + ipex_git_sha[:7]
        except Exception:
            pass
    return ipex_version


def write_buffer_to_file(file_path, buffer):
    create_if_not_exist(os.path.dirname(file_path))
    with open(file_path, 'w') as f:
        f.write(buffer)
        f.close()


def get_code_fingerprint(ipex_build_version, ipex_git_sha, torch_git_sha, build_mode):
    fingerprint = "{}_{}_{}_{}".format(ipex_build_version, ipex_git_sha, torch_git_sha, build_mode)
    return fingerprint


def create_version_files(base_dir, ipex_build_version, ipex_git_sha, torch_git_sha, gpu_onednn_sha):
    print('Building Intel Extension for PyTorch. Version: {}'.format(ipex_build_version))
    py_version_path = os.path.join(base_dir, PACKAGE_NAME, '_version.py')
    cpp_version_path = os.path.join(base_dir, PACKAGE_NAME, '..', 'csrc', 'utils', 'version.h')
    build_mode_str = get_build_type()

    py_buffer = "# Autogenerated file, do not edit!\n"
    py_buffer += "__version__ = '{}'\n".format(ipex_build_version)
    py_buffer += "__ipex_gitrev__ = '{}'\n".format(ipex_git_sha)
    py_buffer += "__torch_gitrev__ = '{}'\n".format(torch_git_sha)
    py_buffer += "__gpu_onednn_gitrev__ = '{}'\n".format(gpu_onednn_sha)
    py_buffer += "__mode__ = '{}'\n".format(build_mode_str)

    c_buffer = '// Autogenerated file, do not edit!\n'
    c_buffer += '// clang-format off\n'
    c_buffer += '// code fingerprint: {}\n'.format(get_code_fingerprint)
    c_buffer += '// clang-format on\n\n'
    c_buffer += '#pragma once\n'
    c_buffer += '#include <string>\n\n'
    c_buffer += 'namespace torch_ipex {\n\n'
    c_buffer += 'const std::string __version__()\n'.format(ipex_build_version)
    c_buffer += '{{ return "{}"; }}\n\n'.format(ipex_build_version)
    c_buffer += 'const std::string __gitrev__()\n'.format(ipex_git_sha)
    c_buffer += '{{ return "{}"; }}\n\n'.format(ipex_git_sha)
    c_buffer += 'const std::string __torch_gitrev__()\n'.format(torch_git_sha)
    c_buffer += '{{ return "{}"; }}\n\n'.format(torch_git_sha)
    c_buffer += 'const std::string __mode__()\n'.format(build_mode_str)
    c_buffer += '{{ return "{}"; }}\n\n'.format(build_mode_str)
    c_buffer += '}  // namespace torch_ipex\n'

    write_buffer_to_file(py_version_path, py_buffer)
    write_buffer_to_file(cpp_version_path, c_buffer)


def get_build_dir():
    project_root_dir = os.path.dirname(__file__)
    return os.path.join(project_root_dir, 'build')


def get_project_dir():
    project_root_dir = os.path.dirname(__file__)
    return os.path.abspath(project_root_dir)


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
    project_root_dir = os.path.dirname(__file__)
    cpu_root_dir = os.path.join(project_root_dir, 'csrc', 'cpu')
    return os.path.abspath(cpu_root_dir)


def get_ipex_cpu_build_dir():
    cpu_build_dir = os.path.join(get_build_type_dir(), 'csrc', 'cpu')
    create_if_not_exist(cpu_build_dir)
    return cpu_build_dir


def get_xpu_project_dir():
    project_root_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(project_root_dir)


def get_xpu_project_build_dir():
    xpu_build_dir = os.path.join(get_build_type_dir(), 'csrc', 'gpu')
    create_if_not_exist(xpu_build_dir)
    return xpu_build_dir


def get_xpu_compliers():
    if shutil.which('icx') is None or shutil.which('dpcpp') is None:
        raise RuntimeError("Failed to find compiler path from OS PATH")
    return "icx", "dpcpp"


def get_ipex_python_dir():
    project_root_dir = os.path.dirname(__file__)
    python_root_dir = os.path.join(project_root_dir, PACKAGE_NAME, 'csrc')
    return os.path.abspath(python_root_dir)


def get_ipex_python_build_dir():
    python_build_dir = os.path.join(get_build_type_dir(), PACKAGE_NAME, 'csrc')
    create_if_not_exist(python_build_dir)
    return python_build_dir


base_dir = os.path.dirname(os.path.abspath(__file__))
# Generate version info (ipex.__version__)
ipex_git_sha, torch_git_sha = get_git_head_sha(base_dir)
ipex_build_version = get_build_version(ipex_git_sha)
ipex_gpu_onednn_git_sha = get_submodule_commit(base_dir, "third_party/oneDNN")
ipex_cpu_ideep_git_sha = get_submodule_commit(base_dir, "third_party/ideep")
create_version_files(base_dir, ipex_build_version, ipex_git_sha, torch_git_sha, ipex_gpu_onednn_git_sha)


# global setup modules
class IPEXClean(distutils.command.clean.clean, object):
    def run(self):
        import glob
        import re
        with open('.gitignore', 'r') as f:
            ignores = f.read()
            pat = re.compile(r'^#( BEGIN NOT-CLEAN-FILES )?')
            for wildcard in filter(None, ignores.split('\n')):
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
    return os.path.join(project_root_dir, 'tests', 'cpu', 'cpp')


def get_cpp_test_build_dir():
    cpp_test_build_dir =  os.path.join(get_build_type_dir(), 'tests', 'cpu', 'cpp')
    create_if_not_exist(cpp_test_build_dir)
    return cpp_test_build_dir


def get_pybind11_abi_compiler_flags():
    pybind11_abi_flags = []
    for pname in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
        pval = getattr(torch._C, f"_PYBIND11_{pname}")
        if pval is not None:
            pybind11_abi_flags.append(f'-DPYBIND11_{pname}=\\"{pval}\\"')
    cl_flags = ""
    for flag in pybind11_abi_flags:
        cl_flags += (flag +' ')
    return cl_flags


def _gen_build_cfg_from_cmake(cmake_exec, project_root_dir, cmake_args, build_dir, build_env):
    check_call([cmake_exec, project_root_dir] + cmake_args, cwd=build_dir, env=build_env)


def _build_project(build_args, build_dir, build_env, use_ninja = False):
    if use_ninja:
        check_call(['ninja'] + build_args, cwd=build_dir, env=build_env)
    else:
        check_call(['make'] + build_args, cwd=build_dir, env=build_env)


class IPEXCPPLibBuild(build_clib, object):
    def run(self):
        self.build_lib = os.path.relpath(get_package_dir())
        self.build_temp = os.path.relpath(get_build_type_dir())

        def defines(args, **kwargs):
            for key, value in sorted(kwargs.items()):
                if value is not None:
                    args.append('-D{}={}'.format(key, value))

        cmake_exec = get_cmake_command()

        if cmake_exec is None:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))
        self.cmake = cmake_exec

        if platform.system() == "Windows":
            raise RuntimeError("Intel Extension for PyTorch only supports Linux now.")

        project_root_dir = get_project_dir()
        ipex_cpu_dir = get_ipex_cpu_dir()
        ipex_cpu_build_dir = get_ipex_cpu_build_dir()
        build_type_dir = get_build_type_dir()
        output_lib_path = get_package_lib_dir()
        ipex_python_dir = get_ipex_python_dir()
        ipex_python_build_dir = get_ipex_python_build_dir()

        ipex_xpu_dir = get_xpu_project_dir()
        ipex_xpu_build_dir = get_xpu_project_build_dir()

        cmake_prefix_path = torch.utils.cmake_prefix_path
        try:
            import pybind11
            cmake_prefix_path = ';'.join([cmake_prefix_path, pybind11.get_cmake_dir()])
        except ImportError as e:
            print("Warning: Cannot find local pybind11 package.")

        build_option_common = {
            'CMAKE_BUILD_TYPE'      : get_build_type(),
            'CMAKE_INSTALL_LIBDIR'  : 'lib',
            'CMAKE_PREFIX_PATH'     : cmake_prefix_path,
            'CMAKE_INSTALL_PREFIX'  : os.path.abspath(get_package_dir()),
            'IPEX_INSTALL_LIBDIR'   : os.path.abspath(output_lib_path),
            'GLIBCXX_USE_CXX11_ABI' : str(int(USE_CXX11_ABI)), # TODO: check if is must need in bdist_cppsdk build.
            'CMAKE_PROJECT_VERSION' : get_version_num(),
            'PYTHON_INCLUDE_DIR'    : sysconfig.get_paths()['include'],
            'PYTHON_EXECUTABLE'     : sys.executable,
            'PYTHON_PLATFORM_INFO'  : platform.platform(),
            'PYTORCH_INSTALL_DIR'   : pytorch_install_dir,
            # 'MKL_INSTALL_DIR'     : mkl_install_dir,
            'PYBIND11_CL_FLAGS'     : get_pybind11_abi_compiler_flags(),
            'IPEX_PROJ_NAME'        : PACKAGE_NAME
        }

        use_ninja = False
        build_with_xpu = False
        sequential_build = False
        my_env = os.environ.copy()
        for var, val in my_env.items():
            if var.startswith(('BUILD_', 'USE_', 'CMAKE_')):
                if var == 'CMAKE_PREFIX_PATH':
                    build_option_common[var] = ';'.join([build_option_common[var], val.replace(':', ';')])
                    continue # XXX: Do NOT overwrite CMAKE_PREFIX_PATH. Append into the list, instead!
                if var == 'BUILD_STATS' and val.upper() in ON_ENV_VAL:
                    sequential_build = True
                if var == 'BUILD_WITH_XPU' and val.upper() in ON_ENV_VAL:
                    build_with_xpu = True
                if var == 'USE_NINJA' and val.upper() in ON_ENV_VAL:
                    use_ninja = True
                build_option_common[var] = val

        cmake_common_args = []
        defines(cmake_common_args, **build_option_common)

        build_nproc = str(os.cpu_count())
        if sequential_build:
            build_nproc = '1'
            print("WARNING: Practice as sequential build with single process !")
        build_args = ['-j', build_nproc, 'install']

        # Build XPU module:
        if(build_with_xpu):
            if os.path.isdir(ipex_xpu_dir) is False:
                raise RuntimeError('It maybe CPU only branch, and it is not contains XPU code.')

            if not torch._C._GLIBCXX_USE_CXX11_ABI:
                print("Intel extension for pytorch only supports _GLIBCXX_USE_CXX11_ABI = 1,\
                     please install pytorch with cxx11abi enabled.")
                sys.exit(1)

            gpu_cc, gpu_cxx = get_xpu_compliers()
            build_option_gpu = {
                'BUILD_MODULE_TYPE'     :'GPU',
                'CMAKE_C_COMPILER'      : gpu_cc,
                'CMAKE_CXX_COMPILER'    : gpu_cxx
            }

            if get_build_type() == 'Debug':
                build_option_gpu = {
                    **build_option_gpu,
                    'BUILD_SEPARATE_OPS'    : 'ON',
                    'USE_SYCL_ASSERT'       : 'ON'
                }

            cmake_args_gpu = copy.deepcopy(cmake_common_args)
            defines(cmake_args_gpu, **build_option_gpu)

            _gen_build_cfg_from_cmake(cmake_exec, project_root_dir, cmake_args_gpu, ipex_xpu_build_dir, my_env)
            _build_project(build_args, ipex_xpu_build_dir, my_env, use_ninja)

        # # Build CPU module:
        # build_option_cpu = {
        #     'BUILD_MODULE_TYPE' : 'CPU',
        #     'IPEX_DISP_OP'      : 1 if _check_env_flag("IPEX_DISP_OP") else 0,
        # }

        # cmake_args_cpu = copy.deepcopy(cmake_common_args)
        # defines(cmake_args_cpu, **build_option_cpu)

        # _gen_build_cfg_from_cmake(cmake_exec, ipex_cpu_dir, cmake_args_cpu, ipex_cpu_build_dir, my_env)
        # _build_project(build_args, ipex_cpu_build_dir, my_env, use_ninja)

        # # Build the CPP UT
        # build_option_cpp_test = {
        #     'PROJECT_DIR'           : project_root_dir,
        #     'CPP_TEST_BUILD_DIR'    : get_cpp_test_build_dir(),
        # }

        # defines(cmake_args_cpu, **build_option_cpp_test)

        # _gen_build_cfg_from_cmake(cmake_exec, get_cpp_test_dir(), cmake_args_cpu, cpp_test_build_dir, my_env)
        # _build_project(build_args, cpp_test_build_dir, my_env, use_ninja)

        # Build common python module:
        build_option_python = {
            'BUILD_MODULE_TYPE' : 'PYTHON',
        }

        if(build_with_xpu):
            gpu_cc, gpu_cxx = get_xpu_compliers()
            build_option_python = {
                **build_option_python,
                'CMAKE_C_COMPILER'      : gpu_cc,
                'CMAKE_CXX_COMPILER'    : gpu_cxx
            }

        cmake_args_python = copy.deepcopy(cmake_common_args)
        defines(cmake_args_python, **build_option_python)

        _gen_build_cfg_from_cmake(cmake_exec, project_root_dir, cmake_args_python, ipex_python_build_dir, my_env)
        _build_project(build_args, ipex_python_build_dir, my_env, use_ninja)


# cppsdk specific setup modules
class IPEXBDistCPPSDK(Command):
    description = "Description of the command"
    user_options = []

    # This method must be implemented
    def initialize_options(self):
        pass

    # This method must be implemented
    def finalize_options(self):
        pass

    def run(self):
        self.run_command('build_clib')

        tmp_dir = 'tmp'
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
        shutil.copyfile(os.path.join('tools', 'install_c++_sdk.sh.in'), os.path.join(tmp_dir, 'install_c++_sdk.sh'))
        shutil.copyfile(os.path.join('cmake', 'Modules', 'FindIPEX.cmake.in'), os.path.join(tmp_dir, 'intel_ext_pt_cpuConfig.cmake'))
        shutil.copyfile(os.path.join('build', 'Release', 'packages', package_name, 'lib', 'libintel-ext-pt-cpu.so'), os.path.join(tmp_dir, 'libintel-ext-pt-cpu.so'))

        if int(USE_CXX11_ABI) == 0:
            run_file_name = 'libintel-ext-pt-{}.run'.format(TORCH_IPEX_VERSION)
        if int(USE_CXX11_ABI) == 1:
            run_file_name = 'libintel-ext-pt-cxx11-abi-{}.run'.format(TORCH_IPEX_VERSION)
        dist_dir = 'dist'
        if not os.path.exists(dist_dir):
            os.makedirs(dist_dir)
        # TODO: check if we need rename 'intel-ext-pt-cpu.run.in' after merge CPU and GPU.
        shutil.copyfile(os.path.join('tools', 'intel-ext-pt-cpu.run.in'), os.path.join(dist_dir, run_file_name))
        subprocess.check_call(['sed', '-i', 's/<IPEX_VERSION>/{}/'.format(TORCH_IPEX_VERSION), os.path.join(dist_dir, run_file_name)])
        subprocess.check_call(['tar', 'czf', '-', '-C', tmp_dir, '.'],
            stdout=open(os.path.join(dist_dir, run_file_name), 'a'))
        shutil.rmtree(tmp_dir)

        if os.path.isfile(os.path.join(dist_dir, run_file_name)):
            print('\n{} is generated in folder "{}"'.format(run_file_name, dist_dir))


def get_src_py_and_dst():
    ret = []
    generated_python_files = glob.glob(
        os.path.join(get_project_dir(), PACKAGE_NAME, '**/*.py'),
        recursive=True)
    for src in generated_python_files:
        dst = os.path.join(
            get_package_base_dir(),
            PACKAGE_NAME,
            os.path.relpath(src, os.path.join(get_project_dir(), PACKAGE_NAME)))
        dst_path = Path(dst)
        if not dst_path.parent.exists():
            Path(dst_path.parent).mkdir(parents=True, exist_ok=True)
        ret.append((src, dst))
    return ret


# python specific setup modules
class IPEXEggInfoBuild(egg_info, object):
    def finalize_options(self):
        self.egg_base = os.path.relpath(get_package_base_dir())
        ret = get_src_py_and_dst()
        for src, dst in ret:
            self.copy_file(src, dst)
        super(IPEXEggInfoBuild, self).finalize_options()


class IPEXInstallCmd(install, object):
    def finalize_options(self) -> None:
        self.build_lib = os.path.relpath(get_package_base_dir())
        return super(IPEXInstallCmd, self).finalize_options()


class IPEXPythonPackageBuild(build_py, object):
    def run(self) -> None:
        ret = get_src_py_and_dst()
        for src, dst in ret:
            self.copy_file(src, dst)
        super(IPEXPythonPackageBuild, self).finalize_options()


class IPEXExtBuild(BuildExtension):
    def run(self):
        self.run_command('build_clib')

        self.build_lib = os.path.relpath(get_package_base_dir())
        self.build_temp = os.path.relpath(get_build_type_dir())
        self.library_dirs.append(os.path.relpath(get_package_lib_dir()))
        super(IPEXExtBuild, self).run()


def make_relative_rpath(path):
    if IS_DARWIN:
        return '-Wl,-rpath,@loader_path/' + path
    elif IS_WINDOWS:
        raise "Windows support is in the plan. Intel Extension for PyTorch supports Linux now."
    else:
        return '-Wl,-rpath,$ORIGIN/' + path

def pyi_module():
    main_libraries = ['intel-ext-pt-python']
    main_sources = [os.path.join(PACKAGE_NAME, "csrc", "_C.cpp")]

    include_dirs = [
        os.path.realpath("."),
        os.path.realpath(os.path.join(PACKAGE_NAME, "csrc")),
        #os.path.join(mkl_include_path),
        os.path.join(pytorch_install_dir, "include"),
        os.path.join(pytorch_install_dir, "include", "torch", "csrc", "api", "include")]

    library_dirs = [
        "lib",
        os.path.join(pytorch_install_dir, "lib")
        ]

    extra_compile_args = [
        '-Wall',
        '-Wextra',
        '-Wno-strict-overflow',
        '-Wno-unused-parameter',
        '-Wno-missing-field-initializers',
        '-Wno-write-strings',
        '-Wno-unknown-pragmas',
        # This is required for Python 2 declarations that are deprecated in 3.
        '-Wno-deprecated-declarations',
        # Python 2.6 requires -fno-strict-aliasing, see
        # http://legacy.python.org/dev/peps/pep-3123/
        # We also depend on it in our code (even Python 3).
        '-fno-strict-aliasing',
        # Clang has an unfixed bug leading to spurious missing
        # braces warnings, see
        # https://bugs.llvm.org/show_bug.cgi?id=21629
        '-Wno-missing-braces']

    C_ext = CppExtension(
        "{}._C".format(PACKAGE_NAME),
        libraries=main_libraries,
        sources=main_sources,
        language='c++',
        extra_compile_args=extra_compile_args,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_link_args=[make_relative_rpath('lib')])
    return C_ext


ext_modules=[]
cmdclass = {
    'build_clib': IPEXCPPLibBuild,
    'clean': IPEXClean,
}

if build_target == 'cppsdk':
    cmdclass['bdist_cppsdk'] = IPEXBDistCPPSDK
elif build_target == 'python':
    cmdclass['build_ext'] = IPEXExtBuild
    cmdclass['build_py'] = IPEXPythonPackageBuild
    cmdclass['egg_info'] = IPEXEggInfoBuild
    cmdclass['install'] = IPEXInstallCmd
    ext_modules.append(pyi_module())

long_description = ''
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

entry_points = {
    'console_scripts': [
        'ipexrun = {}.cpu.launch:main'.format(PACKAGE_NAME),
    ]
}

setup(
    name=PACKAGE_NAME,
    version=ipex_build_version,
    description='Intel® Extension for PyTorch*',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/intel/intel-extension-for-pytorch',
    author='Intel Corp.',
    install_requires=_build_installation_dependency(),
    packages=[PACKAGE_NAME],
    package_data={
        PACKAGE_NAME: [
            "*.so",
            "lib/*.so",
        ]},
    package_dir={'': os.path.relpath(get_package_base_dir())},
    zip_safe=False,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    entry_points=entry_points,
    )
