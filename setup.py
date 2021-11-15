#!/usr/bin/env python
from __future__ import print_function
from distutils.command.build_py import build_py
from distutils.command.install import install
import pkg_resources
from setuptools.command.build_clib import build_clib
from setuptools.command.egg_info import egg_info

from subprocess import check_call, check_output
from setuptools import setup, distutils
from distutils.version import LooseVersion
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

TORCH_VERSION = '1.10.0'
TORCH_VERSION = os.getenv('TORCH_VERSION', TORCH_VERSION)

TORCH_IPEX_VERSION = '1.10.0+cpu'
PYTHON_VERSION = sys.version_info

IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')
IS_LINUX = (platform.system() == 'Linux')

try:
    from packaging import version as pkg_ver
except Exception:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'packaging'])
    from packaging import version as pkg_ver

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension
except ImportError as e:
    print("Unable to import torch from the local environment.")
    raise e

pytorch_install_dir = os.path.dirname(os.path.abspath(torch.__file__))
base_dir = os.path.dirname(os.path.abspath(__file__))
python_include_dir = get_paths()['include']
package_name = "intel_extension_for_pytorch"
short_package_name = "torch_ipex"


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
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', k])
            if k == 'wheel':
                restart = True
        if restart:
            os.execv(sys.executable, ['python'] + sys.argv)
            exit(1)


def _build_installation_dependency():
    if os.getenv('TORCH_VERSION') is None:
        return []

    install_requires = []
    TORCH_URL = 'torch @ https://download.pytorch.org/whl/cpu/torch-{0}%2Bcpu-cp{1}{2}-cp{1}{2}-linux_x86_64.whl'.format(TORCH_VERSION, PYTHON_VERSION.major, PYTHON_VERSION.minor)
    if IS_DARWIN:
        TORCH_URL = 'torch=={}'.format(TORCH_VERSION)
    else:
        OS_VER = 'linux_x86_64'
        if IS_WINDOWS:
            TORCH_URL = 'torch @ https://download.pytorch.org/whl/cpu/torch-{0}%2Bcpu-cp{1}{2}-cp{1}{2}-win_amd64.whl'.format(TORCH_VERSION, PYTHON_VERSION.major, PYTHON_VERSION.minor)
            OS_VER = 'win_amd64'

        try:
            fp = urllib.request.urlopen('https://download.pytorch.org/whl/torch_stable.html', timeout=30)
            cont_bytes = fp.read()
            cont = cont_bytes.decode('utf8').replace('\n', '')
            fp.close()

            lines = re.split(r'<br>', cont)

            for line in lines:
                matches = re.match('<a href="(cpu\/torch-{0}.*cp{1}{2}.*{3}.*)">(.*)<\/a>'.format(TORCH_VERSION, PYTHON_VERSION.major, PYTHON_VERSION.minor, OS_VER), line)
                if matches and len(matches.groups()) == 2:
                    TORCH_URL = 'torch @ https://download.pytorch.org/whl/{}'.format(matches.group(2))
                    break
        except Exception:
            pass

    install_requires.append(TORCH_URL)
    return install_requires


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
                return LooseVersion(line.strip().split(' ')[2])
        raise RuntimeError('no version found')
    "Returns cmake command."
    cmake_command = 'cmake'
    if platform.system() == 'Windows':
        return cmake_command
    cmake3 = which('cmake3')
    cmake = which('cmake')
    if cmake3 is not None and _get_version(cmake3) >= LooseVersion("3.13.0"):
        cmake_command = 'cmake3'
        return cmake_command
    elif cmake is not None and _get_version(cmake) >= LooseVersion("3.13.0"):
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


def create_version_files(base_dir, ipex_build_version, ipex_git_sha, torch_git_sha, ipex_avx_version):
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
    cpp_version_path = os.path.join(base_dir, short_package_name, 'csrc', 'version.cpp')

    py_buffer = "# Autogenerated file, do not edit!\n"
    py_buffer += "__version__ = '{}'\n".format(ipex_build_version)
    py_buffer += "__gitrev__ = '{}'\n".format(ipex_git_sha)
    py_buffer += "__avx_version__ = '{}'\n".format(ipex_avx_version)
    py_buffer += "__torch_gitrev__ = '{}'\n".format(torch_git_sha)
    mode_str = "release"
    if _check_env_flag('DEBUG'):
        mode_str = "debug"
    py_buffer += "__mode__ = '{}'\n".format(mode_str)

    c_buffer = '// Autogenerated file, do not edit!\n'
    c_buffer += '#include "torch_ipex/csrc/version.h"\n\n'
    c_buffer += 'namespace torch_ipex {\n\n'
    c_buffer += 'const std::string __version__ = "{}";\n'.format(ipex_build_version)
    c_buffer += 'const std::string __gitrev__ = "{}";\n'.format(ipex_git_sha)
    c_buffer += 'const std::string __avx_version__ = "{}";\n'.format(ipex_avx_version)
    c_buffer += 'const std::string __torch_gitrev__ = "{}";\n\n'.format(torch_git_sha)
    c_buffer += 'const std::string __mode__ = "{}";\n\n'.format(mode_str)
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


def get_avx_version():
    avx_version = ''
    if _check_env_flag('AVX2'):
        avx_version = 'AVX2'
    elif _check_env_flag('AVX512'):
        avx_version = 'AVX512'

    if avx_version == '':
        avx_version = 'AVX512'

    print("The extension will be built with {}.".format(avx_version))
    return avx_version


def get_build_type_dir():
    return os.path.join(get_build_dir(), get_build_type())


def get_package_base_dir():
    return os.path.join(get_build_type_dir(), "packages")


def get_package_dir():
    return os.path.join(get_package_base_dir(), package_name)


def get_package_lib_dir():
    return os.path.join(get_package_dir(), "lib")

def get_cpp_test_dir():
    project_root_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(project_root_dir, 'tests', 'cpu', 'cpp')

def get_cpp_test_build_dir():
    return os.path.join(get_build_type_dir(), 'tests', 'cpu', 'cpp')

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

        build_dir = get_build_dir()
        project_dir = get_project_dir()
        build_type_dir = get_build_type_dir()
        output_lib_path = get_package_lib_dir()

        if not os.path.exists(build_dir):
            Path(build_dir).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(build_type_dir):
            Path(build_type_dir).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(output_lib_path):
            Path(output_lib_path).mkdir(parents=True, exist_ok=True)

        cmake_args = [
            '-DCMAKE_BUILD_TYPE=' + get_build_type(),
            '-DCMAKE_INSTALL_PREFIX=' + os.path.abspath(output_lib_path),
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + os.path.abspath(output_lib_path),
            '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=' + os.path.abspath(output_lib_path),
            '-DIPEX_INSTALL_LIBDIR=' + os.path.abspath(output_lib_path),
            '-DIPEX_AVX_VERSION=' + get_avx_version(),
            '-DGLIBCXX_USE_CXX11_ABI=' + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
            '-DPYTHON_INCLUDE_DIR=' + python_include_dir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DPYTORCH_INSTALL_DIR=' + pytorch_install_dir,
            '-DPYTORCH_INCLUDE_DIRS=' + pytorch_install_dir + "/include",
            '-DPYTORCH_LIBRARY_DIRS=' + pytorch_install_dir + "/lib"]

        if _check_env_flag("IPEX_DISP_OP"):
            cmake_args += ['-DIPEX_DISP_OP=1']

        if os.getenv("IPEX_PROFILE_OP", "") != "0":
            cmake_args += ['-DIPEX_PROFILE_OP=1']

        if _check_env_flag("USE_SYCL"):
            cmake_args += ['-DUSE_SYCL=1']

        if _check_env_flag("DPCPP_ENABLE_PROFILING"):
            cmake_args += ['-DDPCPP_ENABLE_PROFILING=1']

        use_ninja = False
        if _check_env_flag("USE_NINJA"):
            use_ninja = True
            cmake_args += ['-GNinja']

        if _check_env_flag("ENABLE_AUTOCAST_VERBOSE"):
            cmake_args += ['-DENABLE_AUTOCAST_VERBOSE=1']

        build_args = ['-j', str(multiprocessing.cpu_count())]

        env = os.environ.copy()
        if _check_env_flag("USE_SYCL"):
            os.environ['CXX'] = 'compute++'

        check_call([self.cmake, project_dir] + cmake_args, cwd=build_type_dir, env=env)

        # build_args += ['VERBOSE=1']
        if use_ninja:
            check_call(['ninja'] + build_args, cwd=build_type_dir, env=env)
        else:
            check_call(['make'] + build_args, cwd=build_type_dir, env=env)

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

class IPEXExtBuild(BuildExtension):
    def run(self):
        self.run_command('build_clib')

        self.build_lib = os.path.relpath(get_package_base_dir())
        self.build_temp = os.path.relpath(get_build_type_dir())
        self.library_dirs.append(os.path.relpath(get_package_lib_dir()))
        super(IPEXExtBuild, self).run()

# Install requirements for building
_install_requirements()

# Generate version info (ipex.__version__)
ipex_git_sha, torch_git_sha = get_git_head_sha(base_dir)
ipex_build_version = get_build_version(ipex_git_sha)
ipex_avx_version = get_avx_version()
create_version_files(base_dir, ipex_build_version, ipex_git_sha, torch_git_sha, ipex_avx_version)


def make_relative_rpath(path):
    if IS_DARWIN:
        return '-Wl,-rpath,@loader_path/' + path
    elif IS_WINDOWS:
        raise "Windows support is in the plan. Intel Extension for PyTorch supports Linux now."
    else:
        return '-Wl,-rpath,$ORIGIN/' + path


def pyi_module():
    main_libraries = ['intel-ext-pt-cpu']
    main_sources = [os.path.join("torch_ipex", "csrc", "init_python_bindings.cpp"),
                    os.path.join("torch_ipex", "csrc", "python", "TaskModule.cpp")]

    include_dirs = [
        os.path.realpath("."),
        os.path.realpath(os.path.join("torch_ipex", "csrc")),
        os.path.join(pytorch_install_dir, "include"),
        os.path.join(pytorch_install_dir, "include", "torch", "csrc", "api", "include")]

    library_dirs = [
        "lib",
        os.path.join(pytorch_install_dir, "lib")]

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


setup(
    name='intel_extension_for_pytorch',
    version=ipex_build_version,
    description='Intel Extension for PyTorch',
    url='https://github.com/intel/intel-extension-for-pytorch',
    author='Intel/PyTorch Dev Team',
    install_requires=_build_installation_dependency(),
    libraries=[('intel-ext-pt-cpu', {'sources': list()})],
    packages=[
        'intel_extension_for_pytorch'],
    package_data={
        "intel_extension_for_pytorch": [
            "*.so",
            "lib/*.so",
        ]},
    package_dir={'': os.path.relpath(get_package_base_dir())},
    zip_safe=False,
    ext_modules=[pyi_module()],
    cmdclass={
        'build_py': IPEXPythonPackageBuild,
        'build_clib': IPEXCPPLibBuild,
        'build_ext': IPEXExtBuild,
        'egg_info': IPEXEggInfoBuild,
        'install': IPEXInstallCmd,
        'clean': IPEXClean,
    })
