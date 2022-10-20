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

from __future__ import print_function
from distutils.command.build_py import build_py
from distutils.command.install import install
from distutils.cmd import Command
import pkg_resources
from setuptools.command.build_clib import build_clib
from setuptools.command.egg_info import egg_info

from subprocess import check_call, check_output
from setuptools import setup, distutils
from sysconfig import get_paths

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
from pathlib import Path
import warnings
import urllib.request
import re

try:
    from packaging import version as pkg_ver
except Exception:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'packaging'])
    from packaging import version as pkg_ver


#TORCH_VERSION = '1.14.0'
#TORCH_VERSION = os.getenv('TORCH_VERSION', TORCH_VERSION)

TORCH_IPEX_VERSION = '1.14.0+cpu'
PYTHON_VERSION = sys.version_info

package_name = "intel_extension_for_pytorch"

# build mode
pytorch_install_dir = ''
USE_CXX11_ABI = 0
mode = ''
if len(sys.argv) > 1:
    if sys.argv[1] in ['build_clib', 'bdist_cppsdk']:
        mode = 'cppsdk'
        if len(sys.argv) != 3:
            raise RuntimeError('Please set path of libtorch directory if "build_clib" or "bdist_cppsdk" is applied.\nUsage: python setup.py [build_clib|bdist_cppsdk] <libtorch_path>')
        pytorch_install_dir = sys.argv[2]
        if pytorch_install_dir.startswith('.'):
            pytorch_install_dir = os.path.join(os.getcwd(), pytorch_install_dir)
        sys.argv.pop()

        if not os.path.isfile(os.path.join(pytorch_install_dir, 'build-version')):
            raise RuntimeError('{} doestn\'t seem to be a valid libtorch directory.'.format(pytorch_install_dir))

        out = subprocess.check_output(['grep', 'GLIBCXX_USE_CXX11_ABI', os.path.join(pytorch_install_dir, 'share', 'cmake', 'Torch', 'TorchConfig.cmake')]).decode('ascii').strip()
        if out == '':
            raise RuntimeError('Unable to get GLIBCXX_USE_CXX11_ABI setting from libtorch: 1')
        matches = re.match('.*\"-D_GLIBCXX_USE_CXX11_ABI=(\d)\".*', out)
        if matches:
            USE_CXX11_ABI = int(matches.groups()[0])
        else:
            raise RuntimeError('Unable to get GLIBCXX_USE_CXX11_ABI setting from libtorch: 2')
    elif sys.argv[1] in ['clean']:
        mode = 'clean'
    else:
        mode = 'python'
        try:
            import torch
            from torch.utils.cpp_extension import BuildExtension, CppExtension
        except ImportError as e:
            print("Unable to import torch from the local environment.")
            raise e

        pytorch_install_dir = os.path.dirname(os.path.abspath(torch.__file__))
        USE_CXX11_ABI = torch._C._GLIBCXX_USE_CXX11_ABI


# configure for MKL
mkl_install_dir = os.getenv('MKLROOT', '')
if mode != 'clean':
    if mkl_install_dir:
        mkl_header = glob.glob(f'{mkl_install_dir}/include/**/mkl_version.h', recursive = True)
        if len(mkl_header) == 0:
            raise RuntimeError(f'{mkl_install_dir} doesn\'t seem to be a valid MKL library directory.\n{" ":14}mkl_version.h not found.')
            mkl_install_dir = ''
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
                mkl_install_dir = ''
            mkl_library = glob.glob(f'{mkl_install_dir}/lib/**/libmkl_core.a', recursive = True)
            if len(mkl_library) == 0:
                raise RuntimeError(f'libmkl_core.a not found in {mkl_install_dir}/lib/intel64.')
                mkl_install_dir = ''
    if not mkl_install_dir:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'mkl-include>=2021.0.0'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', 'mkl-static>=2021.0.0'])
        mkl_install_dir = os.path.abspath(os.path.join(os.path.dirname(sys.executable), ".."))


# global supporting functions
def _install_requirements():
    installed_raw = {pkg for pkg in pkg_resources.working_set}
    installed = {}
    for i in installed_raw:
        installed[i.key] = i.version

    requires = {}
    requires_raw = {}
    try:
        with open('requirements.txt', 'r') as reader:
            for line in reader.readlines():
                line_raw = line.replace('\n', '')
                line = line_raw.replace('=', '')
                tmp = re.split('[=<>]', line)
                if len(tmp) == 2:
                    requires[tmp[0]] = tmp[1]
                else:
                    requires[tmp[0]] = ''
                requires_raw[tmp[0]] = line_raw
    except Exception:
        pass

    restart = False
    for k in requires.keys():
        if k in installed.keys():
            if requires[k] != '' and pkg_ver.parse(installed[k]) < pkg_ver.parse(requires[k]):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', requires_raw[k]])
                if k == 'wheel':
                    restart = True
        else:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', requires_raw[k]])
            if k == 'wheel':
                restart = True
        if restart:
            os.execv(sys.executable, ['python'] + sys.argv)
            exit(1)


def _build_installation_dependency():
    install_requires = []
    install_requires.append('psutil')
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


# from https://github.com/pytorch/pytorch/blob/master/tools/setup_helpers/__init__.py
def which(thefile):
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for d in path:
        fname = os.path.join(d, thefile)
        fnames = [fname]
        if sys.platform == 'win32':
            exts = os.environ.get('PATHEXT', '').split(os.pathsep)
            fnames += [fname + ext for ext in exts]
        for name in fnames:
            if os.access(name, os.F_OK | os.X_OK) and not os.path.isdir(name):
                return name
    return None


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
    cmake3 = which('cmake3')
    cmake = which('cmake')
    if cmake3 is not None and _get_version(cmake3) >= pkg_ver.parse("3.13.0"):
        cmake_command = 'cmake3'
        return cmake_command
    elif cmake is not None and _get_version(cmake) >= pkg_ver.parse("3.13.0"):
        return cmake_command
    else:
        raise RuntimeError('no cmake or cmake3 with version >= 3.13.0 found')


def _check_env_flag(name, default=''):
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


def get_git_head_sha(base_dir):
    ipex_git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=base_dir).decode('ascii').strip()
    if os.path.isdir(os.path.join(base_dir, '..', '.git')):
        torch_git_sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.join(base_dir, '..')).decode('ascii').strip()
    else:
        torch_git_sha = ''
    return ipex_git_sha, torch_git_sha


def get_build_version(ipex_git_sha):
    ipex_version = os.getenv('TORCH_IPEX_VERSION', TORCH_IPEX_VERSION)
    if _check_env_flag('VERSIONED_IPEX_BUILD', default='0'):
        try:
            ipex_version += '+' + ipex_git_sha[:7]
        except Exception:
            pass
    return ipex_version


def create_version_files(base_dir, ipex_build_version, ipex_git_sha, torch_git_sha):
    def write_buffer_to_file(file_path, buffer):
        write_buffer_flag = True
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content_org = f.read()
                if buffer == content_org:
                    write_buffer_flag = False
                f.close()

        if write_buffer_flag:
            with open(file_path, 'w') as f:
                f.write(buffer)
                f.close()

    print('Building Intel Extension for PyTorch. Version: {}'.format(ipex_build_version))
    py_version_path = os.path.join(base_dir, package_name, 'version.py')
    cpp_version_path = os.path.join(base_dir, package_name, '..', 'csrc', 'utils', 'version.cpp')

    py_buffer = "# Autogenerated file, do not edit!\n"
    py_buffer += "__version__ = '{}'\n".format(ipex_build_version)
    py_buffer += "__gitrev__ = '{}'\n".format(ipex_git_sha)
    py_buffer += "__torch_gitrev__ = '{}'\n".format(torch_git_sha)
    mode_str = "release"
    if _check_env_flag('DEBUG'):
        mode_str = "debug"
    py_buffer += "__mode__ = '{}'\n".format(mode_str)

    c_buffer = '// Autogenerated file, do not edit!\n'
    # c_buffer += '#include "intel_extension_for_pytorch/csrc/version.h"\n\n'
    c_buffer += '#include "version.h"\n\n'
    c_buffer += 'namespace torch_ipex {\n\n'
    c_buffer += 'const std::string __version__()\n'.format(ipex_build_version)
    c_buffer += '{{ return "{}"; }}\n\n'.format(ipex_build_version)
    c_buffer += 'const std::string __gitrev__()\n'.format(ipex_git_sha)
    c_buffer += '{{ return "{}"; }}\n\n'.format(ipex_git_sha)
    c_buffer += 'const std::string __torch_gitrev__()\n'.format(torch_git_sha)
    c_buffer += '{{ return "{}"; }}\n\n'.format(torch_git_sha)
    c_buffer += 'const std::string __mode__()\n'.format(mode_str)
    c_buffer += '{{ return "{}"; }}\n\n'.format(mode_str)
    c_buffer += '}  // namespace torch_ipex\n'

    write_buffer_to_file(py_version_path, py_buffer)
    write_buffer_to_file(cpp_version_path, c_buffer)


def get_build_dir():
    project_root_dir = os.path.dirname(__file__)
    return os.path.join(project_root_dir, 'build')


def get_project_dir():
    project_root_dir = os.path.dirname(__file__)
    return os.path.abspath(project_root_dir)

def get_build_type():
    build_type = 'Release'
    if _check_env_flag('DEBUG'):
        build_type = 'Debug'

    if _check_env_flag('REL_WITH_DEB_INFO'):
        build_type = 'RelWithDebInfo'

    return build_type


def get_build_type_dir():
    return os.path.join(get_build_dir(), get_build_type())


def get_package_base_dir():
    return os.path.join(get_build_type_dir(), "packages")


def get_package_dir():
    return os.path.join(get_package_base_dir(), package_name)


def get_package_lib_dir():
    return os.path.join(get_package_dir(), "lib")

def get_ipex_cpu_dir():
    project_root_dir = os.path.dirname(__file__)
    cpu_root_dir = os.path.join(project_root_dir, 'csrc', 'cpu')
    return os.path.abspath(cpu_root_dir)

def get_ipex_cpu_build_dir():
    return os.path.join(get_build_type_dir(), 'csrc', 'cpu')

def get_ipex_python_dir():
    project_root_dir = os.path.dirname(__file__)
    python_root_dir = os.path.join(project_root_dir, 'intel_extension_for_pytorch', 'csrc', 'python')
    return os.path.abspath(python_root_dir)

def get_ipex_python_build_dir():
    return os.path.join(get_build_type_dir(), 'csrc', 'python')

# initialize variables for compilation
IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')
IS_LINUX = (platform.system() == 'Linux')

base_dir = os.path.dirname(os.path.abspath(__file__))
python_include_dir = get_paths()['include']

# Generate version info (ipex.__version__)
ipex_git_sha, torch_git_sha = get_git_head_sha(base_dir)
ipex_build_version = get_build_version(ipex_git_sha)
create_version_files(base_dir, ipex_build_version, ipex_git_sha, torch_git_sha)


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
    return os.path.join(get_build_type_dir(), 'tests', 'cpu', 'cpp')

def get_pybind11_abi_compiler_flags():
    import torch
    pybind11_abi_flags = []

    for pname in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
        pval = getattr(torch._C, f"_PYBIND11_{pname}")
        if pval is not None:
            pybind11_abi_flags.append(f'-DPYBIND11_{pname}=\\"{pval}\\"')

    cl_flags = ""
    for flag in pybind11_abi_flags:
        cl_flags += (flag +' ')

    return cl_flags

class IPEXCPPLibBuild(build_clib, object):
    def run(self):
        self.build_lib = os.path.relpath(get_package_dir())
        self.build_temp = os.path.relpath(get_build_type_dir())

        cmake = get_cmake_command()

        if cmake is None:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))
        self.cmake = cmake

        if platform.system() == "Windows":
            raise RuntimeError("Intel Extension for PyTorch only supports Linux now.")

        project_dir = get_project_dir()
        ipex_cpu_dir = get_ipex_cpu_dir()
        ipex_cpu_build_dir = get_ipex_cpu_build_dir()
        build_type_dir = get_build_type_dir()
        output_lib_path = get_package_lib_dir()
        ipex_python_dir = get_ipex_python_dir()
        ipex_python_build_dir = get_ipex_python_build_dir()

        if not os.path.exists(build_type_dir):
            Path(build_type_dir).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(output_lib_path):
            Path(output_lib_path).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(ipex_cpu_build_dir):
            Path(ipex_cpu_build_dir).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(ipex_python_build_dir):
            Path(ipex_python_build_dir).mkdir(parents=True, exist_ok=True)

        cmake_args = [
            '-DCMAKE_BUILD_TYPE=' + get_build_type(),
            '-DCMAKE_INSTALL_PREFIX=' + os.path.abspath(output_lib_path),
            '-DIPEX_INSTALL_LIBDIR=' + os.path.abspath(output_lib_path),
            '-DGLIBCXX_USE_CXX11_ABI=' + str(int(USE_CXX11_ABI)),
            '-DPYTHON_INCLUDE_DIR=' + python_include_dir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DPYTORCH_INSTALL_DIR=' + pytorch_install_dir,
            '-DMKL_INSTALL_DIR=' + mkl_install_dir,
            '-DPYBIND11_CL_FLAGS=' + get_pybind11_abi_compiler_flags()
            ]

        if _check_env_flag("IPEX_DISP_OP"):
            cmake_args += ['-DIPEX_DISP_OP=1']

        if _check_env_flag("USE_SYCL"):
            cmake_args += ['-DUSE_SYCL=1']

        if _check_env_flag("DPCPP_ENABLE_PROFILING"):
            cmake_args += ['-DDPCPP_ENABLE_PROFILING=1']

        use_ninja = False
        if _check_env_flag("USE_NINJA"):
            use_ninja = True
            cmake_args += ['-GNinja']


        build_args = ['-j', str(multiprocessing.cpu_count())]
        # build_args += ['VERBOSE=1']

        env = os.environ.copy()
        if _check_env_flag("USE_SYCL"):
            os.environ['CXX'] = 'compute++'

        check_call([self.cmake, ipex_cpu_dir] + cmake_args, cwd=ipex_cpu_build_dir, env=env)

        if use_ninja:
            check_call(['ninja'] + build_args, cwd=ipex_cpu_build_dir, env=env)
        else:
            check_call(['make'] + build_args, cwd=ipex_cpu_build_dir, env=env)

        # Build python.so
        check_call([self.cmake, ipex_python_dir] + cmake_args, cwd=ipex_python_build_dir, env=env)

        if use_ninja:
            check_call(['ninja'] + build_args, cwd=ipex_python_build_dir, env=env)
        else:
            check_call(['make'] + build_args, cwd=ipex_python_build_dir, env=env)

        # Build the CPP UT
        cpp_test_dir = get_cpp_test_dir()
        cpp_test_build_dir = get_cpp_test_build_dir()
        if not os.path.exists(cpp_test_build_dir):
            Path(cpp_test_build_dir).mkdir(parents=True, exist_ok=True)
        cmake_args += ['-DPROJECT_DIR=' + project_dir]
        cmake_args += ['-DCPP_TEST_BUILD_DIR=' + cpp_test_build_dir]
        check_call([self.cmake, cpp_test_dir] + cmake_args, cwd=cpp_test_build_dir, env=env)
        if use_ninja:
            check_call(['ninja'] + build_args, cwd=cpp_test_build_dir, env=env)
        else:
            check_call(['make'] + build_args, cwd=cpp_test_build_dir, env=env)

cmdclass = {
    'build_clib': IPEXCPPLibBuild,
    'clean': IPEXClean,
}
ext_modules=[]


# cppsdk specific setup modules
if mode == 'cppsdk':
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
            shutil.copyfile(os.path.join('tools', 'intel-ext-pt-cpu.run.in'), os.path.join(dist_dir, run_file_name)) # TODO: check if we need rename 'intel-ext-pt-cpu.run.in' after merge CPU and GPU.
            subprocess.check_call(['sed', '-i', 's/<IPEX_VERSION>/{}/'.format(TORCH_IPEX_VERSION), os.path.join(dist_dir, run_file_name)])
            subprocess.check_call(['tar', 'czf', '-', '-C', tmp_dir, '.'],
                stdout=open(os.path.join(dist_dir, run_file_name), 'a'))
            shutil.rmtree(tmp_dir)

            if os.path.isfile(os.path.join(dist_dir, run_file_name)):
                print('\n{} is generated in folder "{}"'.format(run_file_name, dist_dir))


    cmdclass['bdist_cppsdk'] = IPEXBDistCPPSDK

# python specific setup modules
elif mode == 'python':
    # Install requirements for building
    _install_requirements()

    # Find the oneMKL library path
    mkl_lib_path = mkl_install_dir + "/lib/"
    mkl_include_path = mkl_install_dir + "/include/"

    def get_src_py_and_dst():
        ret = []
        generated_python_files = glob.glob(
            os.path.join(get_project_dir(), package_name, '**/*.py'),
            recursive=True)
        for src in generated_python_files:
            dst = os.path.join(
                get_package_base_dir(),
                package_name,
                os.path.relpath(src, os.path.join(get_project_dir(), package_name)))
            dst_path = Path(dst)
            if not dst_path.parent.exists():
                Path(dst_path.parent).mkdir(parents=True, exist_ok=True)
            ret.append((src, dst))
        return ret

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
        main_sources = [os.path.join("intel_extension_for_pytorch", "csrc", "python", "_C.cpp")]

        include_dirs = [
            os.path.realpath("."),
            os.path.realpath(os.path.join("intel_extension_for_pytorch", "csrc", "python")),
            os.path.join(mkl_include_path),
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
            "intel_extension_for_pytorch._C",
            libraries=main_libraries,
            sources=main_sources,
            language='c++',
            extra_compile_args=extra_compile_args,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_link_args=[make_relative_rpath('lib')])
        return C_ext

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
        'ipexrun = intel_extension_for_pytorch.cpu.launch:main',
    ]
}

setup(
    name='intel_extension_for_pytorch',
    version=ipex_build_version,
    description='Intel® Extension for PyTorch*',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/intel/intel-extension-for-pytorch',
    author='Intel Corp.',
    install_requires=_build_installation_dependency(),
    packages=[
        'intel_extension_for_pytorch'],
    package_data={
        "intel_extension_for_pytorch": [
            "*.so",
            "lib/*.so",
        ]},
    package_dir={'': os.path.relpath(get_package_base_dir())},
    zip_safe=False,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    entry_points=entry_points,
    )
