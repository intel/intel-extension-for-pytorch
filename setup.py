#!/usr/bin/env python
from __future__ import print_function

TORCH_VERSION = '1.8.0'
TORCH_IPEX_VERSION = '1.3.0'

# import torch
import platform
import pkg_resources
import re
from socket import timeout
import subprocess
import sys
import os
import urllib.request

try:
    from packaging import version
except Exception:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'packaging'])
    from packaging import version

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
        if requires[k] != '' and version.parse(installed[k]) < version.parse(requires[k]):
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

TORCH_VERSION = os.getenv('TORCH_VERSION', TORCH_VERSION)

try:
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths
except ImportError as e:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch=='+TORCH_VERSION+'+cpu', '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
    import torch

PYTHON_VERSION = sys.version_info
IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')
IS_LINUX = (platform.system() == 'Linux')

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

from subprocess import check_call, check_output
from setuptools import setup, Extension, find_packages, distutils
from setuptools.command.build_ext import build_ext
from distutils.spawn import find_executable
from distutils.version import LooseVersion
from sysconfig import get_paths

import distutils.ccompiler
import distutils.command.clean
import glob
import inspect
import multiprocessing
import multiprocessing.pool
import os
import platform
import re
import shutil
import subprocess
import sys
import pathlib


pytorch_install_dir = os.path.dirname(os.path.abspath(torch.__file__))
base_dir = os.path.dirname(os.path.abspath(__file__))
python_include_dir = get_paths()['include']

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


def _get_env_backend():
  env_backend_var_name = 'IPEX_BACKEND'
  env_backend_options = ['cpu', 'gpu']
  env_backend_val = os.getenv(env_backend_var_name)
  if env_backend_val is None or env_backend_val.strip() == '':
    return 'cpu'
  else:
    if env_backend_val not in env_backend_options:
      print("Intel PyTorch Extension only supports CPU and GPU now.")
      sys.exit(1)
    else:
      return env_backend_val


def get_git_head_sha(base_dir):
  ipex_git_sha = ''
  torch_git_sha = ''
  try:
    ipex_git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                          cwd=base_dir).decode('ascii').strip()
    if os.path.isdir(os.path.join(base_dir, '..', '.git')):
      torch_git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                            cwd=os.path.join(
                                                base_dir,
                                                '..')).decode('ascii').strip()
  except Exception:
    pass
  return ipex_git_sha, torch_git_sha


def get_build_version(ipex_git_sha):
  version = os.getenv('TORCH_IPEX_VERSION', TORCH_IPEX_VERSION)
  if _check_env_flag('VERSIONED_IPEX_BUILD', default='0'):
    try:
      version += '+' + ipex_git_sha[:7]
    except Exception:
      pass
  return version


def create_version_files(base_dir, version, ipex_git_sha, torch_git_sha):
  print('Building torch_ipex version: {}'.format(version))
  py_version_path = os.path.join(base_dir, 'torch_ipex', 'version.py')
  with open(py_version_path, 'w') as f:
    f.write('# Autogenerated file, do not edit!\n')
    f.write("__version__ = '{}'\n".format(version))
    f.write("__ipex_gitrev__ = '{}'\n".format(ipex_git_sha))
    f.write("__torch_gitrev__ = '{}'\n".format(torch_git_sha))

  cpp_version_path = os.path.join(base_dir, 'torch_ipex', 'csrc', 'version.cpp')
  with open(cpp_version_path, 'w') as f:
    f.write('// Autogenerated file, do not edit!\n')
    f.write('#include "torch_ipex/csrc/version.h"\n\n')
    f.write('namespace torch_ipex {\n\n')
    f.write('const char IPEX_GITREV[] = {{"{}"}};\n'.format(ipex_git_sha))
    f.write('const char TORCH_GITREV[] = {{"{}"}};\n\n'.format(torch_git_sha))
    f.write('}  // namespace torch_ipex\n')


def generate_ipex_cpu_aten_code(base_dir):
  cur_dir = os.path.abspath(os.path.curdir)

  os.chdir(os.path.join(base_dir, 'scripts', 'cpu'))

  cpu_ops_path = os.path.join(base_dir, 'torch_ipex', 'csrc', 'cpu')
  sparse_dec_file_path = os.path.join(base_dir, 'scripts', 'cpu', 'pytorch_headers')
  generate_code_cmd = ['./gen-sparse-cpu-ops.sh', cpu_ops_path, pytorch_install_dir, sparse_dec_file_path]
  if subprocess.call(generate_code_cmd) != 0:
    print("Failed to run '{}'".format(generate_code_cmd), file=sys.stderr)
    os.chdir(cur_dir)
    sys.exit(1)

  generate_code_cmd = ['./gen-dense-cpu-ops.sh', cpu_ops_path, pytorch_install_dir]
  if subprocess.call(generate_code_cmd) != 0:
    print("Failed to run '{}'".format(generate_code_cmd), file=sys.stderr)
    os.chdir(cur_dir)
    sys.exit(1)

  os.chdir(cur_dir)


class IPEXExt(Extension, object):
  def __init__(self, name, project_dir=os.path.dirname(__file__)):
    Extension.__init__(self, name, sources=[])
    self.project_dir = os.path.abspath(project_dir)
    self.build_dir = os.path.join(project_dir, 'build')


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


class IPEXBuild(build_ext, object):
  def run(self):
    print("run")

    # Generate the code before globbing!
    generate_ipex_cpu_aten_code(base_dir)

    cmake = get_cmake_command()

    if cmake is None:
      raise RuntimeError(
          "CMake must be installed to build the following extensions: " +
              ", ".join(e.name for e in self.extensions))
    self.cmake = cmake

    if platform.system() == "Windows":
      raise RuntimeError("Does not support windows")

    ipex_exts = [ext for ext in self.extensions if isinstance(ext, IPEXExt)]
    for ext in ipex_exts:
      self.build_ipex_extension(ext)
    
    self.extensions = [ext for ext in self.extensions if not isinstance(ext, IPEXExt)]
    super(IPEXBuild, self).run()

  def build_ipex_extension(self, ext):
    if not isinstance(ext, IPEXExt):
      return super(IPEXBuild, self).build_extension(ext)
    ext_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
    if not os.path.exists(ext.build_dir):
      os.mkdir(ext.build_dir)

    build_type = 'Release'
    use_ninja = False

    if _check_env_flag('DEBUG'):
      build_type = 'Debug'

    # install _torch_ipex.so as python module
    if ext.name == 'torch_ipex' and _check_env_flag("USE_SYCL"):
      ext_dir = ext_dir + '/torch_ipex'

    cmake_args = [
            '-DCMAKE_BUILD_TYPE=' + build_type,
            '-DPYTORCH_INSTALL_DIR=' + pytorch_install_dir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DCMAKE_INSTALL_PREFIX=' + ext_dir,
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + ext_dir,
            '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=' + ext_dir,
            '-DPYTHON_INCLUDE_DIR=' + python_include_dir,
            '-DPYTORCH_INCLUDE_DIRS=' + pytorch_install_dir + "/include",
            '-DPYTORCH_LIBRARY_DIRS=' + pytorch_install_dir + "/lib",
        ]

    if _check_env_flag("IPEX_DISP_OP"):
      cmake_args += ['-DIPEX_DISP_OP=1']

    if _check_env_flag("IPEX_PROFILE_OP"):
      cmake_args += ['-DIPEX_PROFILE_OP=1']

    if _check_env_flag("USE_SYCL"):
      cmake_args += ['-DUSE_SYCL=1']

    if _check_env_flag("DPCPP_ENABLE_PROFILING"):
      cmake_args += ['-DDPCPP_ENABLE_PROFILING=1']

    if _check_env_flag("USE_NINJA"):
      use_ninja = True
      cmake_args += ['-GNinja']

    build_args = ['-j', str(multiprocessing.cpu_count())]

    env = os.environ.copy()
    if _check_env_flag("USE_SYCL"):
      os.environ['CXX'] = 'compute++'
      check_call([self.cmake, ext.project_dir] + cmake_args, cwd=ext.build_dir, env=env)
    else:
      check_call([self.cmake, ext.project_dir] + cmake_args, cwd=ext.build_dir, env=env)

    # build_args += ['VERBOSE=1']
    if use_ninja:
      check_call(['ninja'] + build_args, cwd=ext.build_dir, env=env)
    else:
      check_call(['make'] + build_args, cwd=ext.build_dir, env=env)
    check_call(['make', 'install'] + build_args, cwd=ext.build_dir, env=env)

ipex_git_sha, torch_git_sha = get_git_head_sha(base_dir)
version = get_build_version(ipex_git_sha)

# Generate version info (torch_xla.__version__)
create_version_files(base_dir, version, ipex_git_sha, torch_git_sha)


# Constant known variables used throughout this file


def make_relative_rpath(path):
  if IS_DARWIN:
    return '-Wl,-rpath,@loader_path/' + path
  elif IS_WINDOWS:
    return ''
  else:
    return '-Wl,-rpath,$ORIGIN/' + path

install_requires=[
        TORCH_URL,
]
def get_c_module():
    main_compile_args = []
    main_libraries = ['torch_ipex']
    main_link_args = []
    main_sources = ["torch_ipex/csrc/_C.cpp"]
    cwd = os.path.dirname(os.path.abspath(__file__))
    # lib_path = os.path.join(cwd, "torch_ipex", "lib")
    lib_path = os.path.join(cwd, "build")
    lib_path_1 = os.path.join(cwd, "build", "lib.linux-x86_64-3.8")
    library_dirs = [lib_path, lib_path_1]
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

    C_ext = Extension("torch_ipex._C",
                  libraries=main_libraries,
                  sources=main_sources,
                  language='c',
                  extra_compile_args=main_compile_args + extra_compile_args,
                  include_dirs=include_paths(),
                  library_dirs=library_dirs,
                  extra_link_args=extra_link_args + main_link_args + [make_relative_rpath('lib')])
    return C_ext

setup(
    name='torch_ipex',
    version=version,
    description='Intel PyTorch Extension',
    url='https://github.com/intel/intel-extension-for-pytorch',
    author='Intel/PyTorch Dev Team',
    install_requires=install_requires,
    # Exclude the build files.
    #packages=find_packages(exclude=['build']),
    packages=[
      'torch_ipex',
      'torch_ipex.ops',
      'torch_ipex.optim'],
    package_data={
        'torch_ipex':[
            'README.md',
            'requirements.txt',
            '*.py',
            'lib/*.so',
            'include/*.h',
            'include/core/*.h',
            'include/utils/*.h']
        },
    zip_safe=False,
    ext_modules=[IPEXExt('torch_ipex'), get_c_module()],
    cmdclass={
        'build_ext': IPEXBuild,
        'clean': IPEXClean,
    })
