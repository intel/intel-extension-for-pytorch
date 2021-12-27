#!/usr/bin/env python

##############################################################
# Build options:
#
# USE_ONEDPL            - to use oneDPL in operators
# USE_ONEMKL            - to use oneMKL in operators
# USE_MICROBENCH        - to build with Microbench
# USE_CHANNELS_LAST_1D  - to use channels last 1d feature
# USE_LEVEL_ZERO_ONLY   - to enumerate devices only with Level Zero
# USE_PERSIST_STREAM    - to use persistent oneDNN stream
# USE_PRIMITIVE_CACHE   - to Cache oneDNN primitives by framework
# USE_QUEUE_BARRIER     - to use queue submit_barrier API
# USE_SCRATCHPAD_MODE   - to trun on oneDNN scratchpad user mode
# USE_MULTI_CONTEXT     - to create DPC++ runtime context per device
# USE_ITT               - to Use Intel(R) VTune Profiler ITT functionality
# USE_AOT_DEVLIST       - to set device list for AOT build option, for example, bdw,tgl,ats,..."
# BUILD_BY_PER_KERNEL   - to build by DPC++ per_kernel option (exclusive with USE_AOT_DEVLIST)
# BUILD_NO_L0_ONEDNN    - to build oneDNN without LevelZero support
# BUILD_STRIPPED_BIN    - to strip all symbols after build
# BUILD_INTERNAL_DEBUG  - to build internal debug code path
# BUILD_DOUBLE_KERNEL   - to build double data type kernel (if BUILD_INTERNAL_DEBUG==ON)
# BUILD_NO_CLANGFORMAT  - to build without force clang-format
# BUILD_SEPARATE_OPS    - to build operators in separate libraries
#
##############################################################

from __future__ import print_function

import distutils.command.clean
import os
import pathlib
import platform
import shutil
import subprocess
import sys
from distutils.spawn import find_executable
from subprocess import check_call

import setuptools.command.build_ext
import setuptools.command.install
from setuptools import Extension, distutils, setup

from scripts.tools.setup.cmake import CMake

try:
    import torch
    from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                           include_paths)
except ImportError as e:
    print('Unable to import torch. Error:')
    print('\t', e)
    print('You need to install pytorch first.')
    sys.exit(1)

os.environ.setdefault('IPEX_BACKEND', 'gpu')
base_dir = os.path.dirname(os.path.abspath(__file__))
ipex_pydir = os.path.join(base_dir, 'ipex')
ipex_scripts = os.path.join(base_dir, 'scripts')
ipex_examples = os.path.join(base_dir, 'tests/gpu/examples')


def _get_complier():
    if shutil.which('icx') is None or shutil.which('dpcpp') is None:
        raise RuntimeError("Failed to find compiler path from OS PATH")
    # dpcpp build
    return "icx", "dpcpp"


def _check_env_flag(name, default=''):
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


def _get_env_backend():
    env_backend_var_name = 'IPEX_BACKEND'
    env_backend_options = ['xpu', 'cpu', 'gpu']
    env_backend_val = os.getenv(env_backend_var_name)
    if env_backend_val is None or env_backend_val.strip() == '':
        return env_backend_options[0]
    else:
        if env_backend_val not in env_backend_options:
            print("Intel PyTorch Extension only supports CPU and GPU now.")
            sys.exit(1)
        else:
            return env_backend_val


def get_git_head_sha(base_dir):
    git_sha = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'], cwd=base_dir).decode('ascii').strip()
    return git_sha


def get_submodule_commit(base_dir, submodule_dir):
    return subprocess.check_output(
        ['git', 'submodule', 'status', submodule_dir], cwd=base_dir).decode('ascii').strip().split()[0]


def check_flake8_errors(base_dir, filepath):
    if shutil.which('flake8') is None:
        print("WARNING: Please install flake8 by pip!")
    flak8_cmd = ['flake8']  # '--quiet'
    if os.path.isdir(filepath):
        for root, dirs, files in os.walk(filepath):
            for file in files:
                if(file.endswith('.py')):
                    flak8_cmd.append(os.path.join(root, file))
    elif os.path.isfile(filepath):
        flak8_cmd.append(filepath)
    ret = subprocess.call(flak8_cmd, cwd=base_dir)
    if ret != 0:
        print("ERROR: flake8 found format errors in", filepath, "!")
        sys.exit(1)


def get_build_version(ipex_git_sha):
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
    version_sha = version + (('+' + ipex_git_sha) if (ipex_git_sha != 'Unknown') else '')
    return version, version_sha


def create_version_files(base_dir, version, git_sha_dict):
    print('Building ipex version: {}'.format(version))
    py_version_path = os.path.join(base_dir, 'ipex', '_version.py')

    with open(py_version_path, 'w') as f:
        f.write('# Autogenerated file, do not edit!\n# Build versions for ipex and torch.\n\n')
        f.write("__version__ = '{}'\n".format(version + _get_env_backend()))
        f.write("__torch_version__ = '{}'\n".format(torch.__version__))
        f.write("\n")
        for k, v in git_sha_dict.items():
            f.write("{key} = '{value}'\n".format(key=k, value=v))


check_flake8_errors(base_dir, os.path.abspath(__file__))
check_flake8_errors(base_dir, ipex_pydir)
check_flake8_errors(base_dir, ipex_scripts)
check_flake8_errors(base_dir, ipex_examples)

git_sha_dict = {
    "__ipex_git_sha__": get_git_head_sha(base_dir),
    "__torch_git_sha__": torch.version.git_version,
    "__ittapi_git_sha__": get_submodule_commit(base_dir, "third_party/ittapi"),
    "__onednn_git_sha__": get_submodule_commit(base_dir, "third_party/oneDNN"),
    "__onedpl_git_sha__": get_submodule_commit(base_dir, "third_party/oneDPL")
}

version, version_sha = get_build_version(git_sha_dict.get('__ipex_git_sha__', 'Unknown'))

# Generate version info (ipex.__version__)
create_version_files(base_dir, version, git_sha_dict)

class DPCPPExt(Extension, object):
    def __init__(self, name, project_dir=os.path.dirname(__file__)):
        Extension.__init__(self, name, sources=[])
        self.project_dir = os.path.abspath(project_dir)
        self.build_dir = os.path.join(project_dir, 'build')


class DPCPPInstall(setuptools.command.install.install):
    def run(self):
        self.run_command("build_ext")
        setuptools.command.install.install.run(self)


class DPCPPClean(distutils.command.clean.clean, object):
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


class DPCPPBuild(BuildExtension, object):
    def run(self):
        if platform.system() == "Windows":
            raise RuntimeError("Does not support windows")

        shutil.copy("README.md", "ipex/README.md")
        if os.path.exists("requirements.txt"):
            shutil.copy("requirements.txt", "ipex/requirements.txt")

        dpcpp_exts = [ext for ext in self.extensions if isinstance(ext, DPCPPExt)]
        for ext in dpcpp_exts:
            self.build_extension(ext)
        self.extensions = [ext for ext in self.extensions if not isinstance(ext, DPCPPExt)]
        super(DPCPPBuild, self).run()
        build_py = self.get_finalized_command('build_py')
        build_py.data_files = build_py._get_data_files()
        build_py.run()

    def build_extension(self, ext):
        if not isinstance(ext, DPCPPExt):
            return super(DPCPPBuild, self).build_extension(ext)
        ext_dir = pathlib.Path(ext.project_dir)
        if not os.path.exists(ext.build_dir):
            os.mkdir(ext.build_dir)
        cmake = CMake(ext.build_dir)
        if not os.path.isfile(cmake._cmake_cache_file):
            build_type = 'Release'

            if _check_env_flag('DEBUG'):
                build_type = 'Debug'

            def convert_cmake_dirs(paths):
                def converttostr(input_seq, seperator):
                    # Join all the strings in list
                    final_str = seperator.join(input_seq)
                    return final_str
                try:
                    return converttostr(paths, ";")
                except BaseException:
                    return paths

            def defines(args, **kwargs):
                for key, value in sorted(kwargs.items()):
                    if value is not None:
                        args.append('-D{}={}'.format(key, value))

            cmake_args = []
            try:
                import pybind11
            except ImportError as e:
                cmake_prefix_path = torch.utils.cmake_prefix_path
            else:
                cmake_prefix_path = ';'.join([torch.utils.cmake_prefix_path, pybind11.get_cmake_dir()])

            build_options = {
                # The default value cannot be easily obtained in CMakeLists.txt. We set it here.
                'CMAKE_BUILD_TYPE': build_type,
                'CMAKE_PREFIX_PATH': cmake_prefix_path,
                'CMAKE_INSTALL_PREFIX': '/'.join([str(ext_dir.absolute()), "ipex"]),
                'CMAKE_INSTALL_LIBDIR': 'lib',
                'CMAKE_PROJECT_VERSION': version,
                'PYTHON_EXECUTABLE': sys.executable,
                'PYTHON_INCLUDE_DIR': distutils.sysconfig.get_python_inc(),
                'LIB_NAME': ext.name,
            }

            my_env = os.environ.copy()
            for var, val in my_env.items():
                if var.startswith(('BUILD_', 'USE_', 'CMAKE_')):
                    if var == 'CMAKE_PREFIX_PATH':
                        # Do NOT overwrite this path. Append into the list, instead.
                        build_options[var] += ';' + val
                    else:
                        build_options[var] = val

            cc, cxx = _get_complier()
            defines(cmake_args, CMAKE_C_COMPILER=cc)
            defines(cmake_args, CMAKE_CXX_COMPILER=cxx)
            defines(cmake_args, **build_options)

            cmake = find_executable('cmake3') or find_executable('cmake')
            if cmake is None:
                raise RuntimeError(
                    "CMake must be installed to build the following extensions: " +
                    ", ".join(e.name for e in self.extensions))
            command = [cmake, ext.project_dir] + cmake_args
            print(' '.join(command))

            env = os.environ.copy()
            check_call(command, cwd=ext.build_dir, env=env)

        env = os.environ.copy()
        build_args = ['-j', str(os.cpu_count()), 'install']
        # build_args += ['VERBOSE=1']

        gen_exec = 'make'
        print("build args: {}".format(build_args))
        check_call([gen_exec] + build_args, cwd=ext.build_dir, env=env)


def get_c_module():
    main_compile_args = []
    main_libraries = ['ipex_python']
    main_link_args = []
    main_sources = ["ipex/csrc/_C.cpp"]
    cwd = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(cwd, "ipex", "lib")
    library_dirs = [lib_path]
    extra_link_args = []
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
        '-Wno-missing-braces',
    ]

    def make_relative_rpath(path):
        return '-Wl,-rpath,$ORIGIN/' + path

    include_dirs = include_paths()

    try:
        import pybind11
    except ImportError as e:
        pass
    else:
        include_dirs.append(pybind11.get_include())

    C_ext = CppExtension(
        "ipex._C",
        libraries=main_libraries,
        sources=main_sources,
        language='c++',
        extra_compile_args=main_compile_args + extra_compile_args,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_link_args=extra_link_args + main_link_args + [make_relative_rpath('lib')])
    return C_ext


with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='ipex',
    version=version_sha,
    description='Intel Extension for PyTorch',
    author='Intel PyTorch Team',
    url='https://github.com/intel/intel-extension-for-pytorch',
    # Exclude the build files.
    packages=['ipex',
              'ipex.xpu',
              'ipex.autograd',
              'ipex.xpu.intrinsic',
              'ipex.xpu.intrinsic.modules',
              'ipex.optim'],
    package_data={
        'ipex': [
            'README.md',
            'requirements.txt',
            'lib/*.so',
            'include/*.h',
            'include/core/*.h',
            'include/utils/*.h',
            'share/cmake/Ipex/*']
    },
    long_description=long_description,
    long_description_content_type='test/markdown',
    zip_safe=False,
    ext_modules=[DPCPPExt('ipex'), get_c_module()],
    cmdclass={
        'install': DPCPPInstall,
        'build_ext': DPCPPBuild,
        'clean': DPCPPClean,
    })
