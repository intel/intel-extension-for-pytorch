import copy
import os
import setuptools
import subprocess
import shutil
import re
import shlex
import sys
import torch
import warnings
from torch.utils.cpp_extension import _TORCH_PATH, CppExtension

from typing import List, Optional, Union, Tuple
from torch.torch_version import TorchVersion

from setuptools.command.build_ext import build_ext
from pkg_resources import packaging

IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform.startswith('darwin')
IS_LINUX = sys.platform.startswith('linux')
LIB_EXT = '.pyd' if IS_WINDOWS else '.so'
EXEC_EXT = '.exe' if IS_WINDOWS else ''
CLIB_PREFIX = '' if IS_WINDOWS else 'lib'
CLIB_EXT = '.dll' if IS_WINDOWS else '.so'
SHARED_FLAG = '/DLL' if IS_WINDOWS else '-shared'

MINIMUM_GCC_VERSION = (5, 0, 0)
MINIMUM_MSVC_VERSION = (19, 0, 24215)

COMMON_DPCPP_FLAGS = ['-fPIC']

def get_dpcpp_complier():
    # build cxx via dpcpp    
    dpcpp_cmp = shutil.which('dpcpp')
    if dpcpp_cmp is None:
        raise RuntimeError("Failed to find compiler path from OS PATH")
    _cxxbin = os.getenv("CXX")
    if _cxxbin is not None:
        dpcpp_cmp = _cxxbin
    return dpcpp_cmp

def get_icx_complier():
    # build cc via icx    
    icx_cmp = shutil.which('icx')
    if icx_cmp is None:
        raise RuntimeError("Failed to find compiler path from OS PATH")
    _ccbin = os.getenv("CC")
    if _ccbin is not None:
        dpcpp_cmp = _ccbin    
    return icx_cmp

def is_ninja_available():
    r'''
    Returns ``True`` if the `ninja <https://ninja-build.org/>`_ build system is
    available on the system, ``False`` otherwise.
    '''
    try:
        subprocess.check_output('ninja --version'.split())
    except Exception:
        return False
    else:
        return True


def verify_ninja_availability():
    r'''
    Raises ``RuntimeError`` if `ninja <https://ninja-build.org/>`_ build system is not
    available on the system, does nothing otherwise.
    '''
    if not is_ninja_available():
        raise RuntimeError("Ninja is required to load C++ extensions")

def _is_cpp_file(path: str) -> bool:
    valid_ext = ['.cpp', '.hpp']
    return os.path.splitext(path)[1] in valid_ext

def _is_c_file(path: str) -> bool:
    valid_ext = ['.c', '.h']
    return os.path.splitext(path)[1] in valid_ext

class DpcppBuildExtension(build_ext, object):
    r'''
    A custom :mod:`setuptools` build extension .
    This :class:`setuptools.build_ext` subclass takes care of passing the
    minimum required compiler flags (e.g. ``-std=c++17``) as well as DPCPP
    compilation.
    When using :class:`DpcppBuildExtension`, it is allowed to supply a dictionary
    for ``extra_compile_args`` (rather than the usual list) that maps from
    languages (``cxx``) to a list of additional compiler flags to supply to the
    compiler.

    ``use_ninja`` (bool): If ``use_ninja`` is ``True`` (default), then we
    attempt to build using the Ninja backend. Ninja greatly speeds up
    compilation compared to the standard ``setuptools.build_ext``.
    Fallbacks to the standard distutils backend if Ninja is not available.

    ``no_python_abi_suffix`` (bool): If ``no_python_abi_suffix`` is ``False`` (default),
    then we attempt to build module with python abi suffix, example:
    output module name: module_name.cpython-37m-x86_64-linux-gnu.so, the 
    ``cpython-37m-x86_64-linux-gnu`` is append python abi suffix.

    .. note::
        By default, the Ninja backend uses #CPUS + 2 workers to build the
        extension. This may use up too many resources on some systems. One
        can control the number of workers by setting the `MAX_JOBS` environment
        variable to a non-negative number.
    '''
    @classmethod
    def with_options(cls, **options):
        r'''
        Returns a subclass with alternative constructor that extends any original keyword
        arguments to the original constructor with the given options.
        '''
        class cls_with_options(cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                kwargs.update(options)
                super().__init__(*args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs) -> None:
        super(DpcppBuildExtension, self).__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)

        self.use_ninja = kwargs.get('use_ninja', True)
        if self.use_ninja:
            # Test if we can use ninja. Fallback otherwise.
            msg = ('Attempted to use ninja as the BuildExtension backend but '
                   '{}. Falling back to using the slow distutils backend.')
            if not is_ninja_available():
                warnings.warn(msg.format('we could not find ninja.'))
                self.use_ninja = False

    def finalize_options(self) -> None:
        super().finalize_options()
        if self.use_ninja:
            self.force = True

    def build_extensions(self) -> None:
        dpcpp_ext = False
        extension_iter = iter(self.extensions)
        extension = next(extension_iter, None)
        while not dpcpp_ext and extension:
            extension = next(extension_iter, None)

        for extension in self.extensions:
            # Ensure at least an empty list of flags for 'cxx' when
            # extra_compile_args is a dict. Otherwise, default torch
            # flags do not get passed. Necessary when only one of 'cxx' is
            # passed to extra_compile_args in DPCPPExtension, i.e.
            #   DPCPPExtension(..., extra_compile_args={'cxx': [...]})
            if isinstance(extension.extra_compile_args, dict):
                for ext in ['cxx']:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []

            self._add_compile_flag(extension, '-DTORCH_API_INCLUDE_EXTENSION_H')
            # See note [Pybind11 ABI constants]
            for name in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
                val = getattr(torch._C, f"_PYBIND11_{name}")
                if val is not None and not IS_WINDOWS:
                    self._add_compile_flag(extension, f'-DPYBIND11_{name}="{val}"')
            self._define_torch_extension_name(extension)
            self._add_gnu_cpp_abi_flag(extension)

        # Save the original _compile method for later.
        if self.compiler.compiler_type == 'msvc':
            original_compile = self.compiler.compile
            original_spawn = self.compiler.spawn
        else:
            original_compile = self.compiler._compile

        def append_std17_if_no_std_present(cflags) -> None:
            cpp_format_prefix = '/{}:' if self.compiler.compiler_type == 'msvc' else '-{}='
            cpp_flag_prefix = cpp_format_prefix.format('std')
            cpp_flag = cpp_flag_prefix + 'c++17'
            if not any(flag.startswith(cpp_flag_prefix) for flag in cflags):
                cflags.append(cpp_flag)

        def unix_dpcpp_flags(cflags):
            cflags = (COMMON_DPCPP_FLAGS + cflags)
            return cflags

        def convert_to_absolute_paths_inplace(paths):
            # Helper function. See Note [Absolute include_dirs]
            if paths is not None:
                for i in range(len(paths)):
                    if not os.path.isabs(paths[i]):
                        paths[i] = os.path.abspath(paths[i])

        def unix_wrap_single_compile(obj, src, ext, cc_args, extra_postargs, pp_opts) -> None:
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_cpp_file(src):
                    _cxxbin = get_dpcpp_complier()
                    self.compiler.set_executable('compiler_so', _cxxbin)
                    if isinstance(cflags, dict):
                        cflags = cflags['cxx']
                    else:
                        cflags = unix_dpcpp_flags(cflags)
                elif _is_c_file(src):
                    _ccbin = get_icx_complier()
                    self.compiler.set_executable('compiler_so', _ccbin)
                    if isinstance(cflags, dict):
                        cflags = cflags['cxx']
                    else:
                        cflags = unix_dpcpp_flags(cflags)                    
                elif isinstance(cflags, dict):
                    cflags = cflags['cxx']
                append_std17_if_no_std_present(cflags)

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable('compiler_so', original_compiler)

        def unix_wrap_ninja_compile(sources,
                                    output_dir=None,
                                    macros=None,
                                    include_dirs=None,
                                    debug=0,
                                    extra_preargs=None,
                                    extra_postargs=None,
                                    depends=None):
            r"""Compiles sources by outputting a ninja file and running it."""
            # NB: I copied some lines from self.compiler (which is an instance
            # of distutils.UnixCCompiler). See the following link.
            # https://github.com/python/cpython/blob/f03a8f8d5001963ad5b5b28dbd95497e9cc15596/Lib/distutils/ccompiler.py#L564-L567
            # This can be fragile, but a lot of other repos also do this
            # (see https://github.com/search?q=_setup_compile&type=Code)
            # so it is probably OK; we'll also get CI signal if/when
            # we update our python version (which is when distutils can be
            # upgraded)

            # Use absolute path for output_dir so that the object file paths
            # (`objects`) get generated with absolute paths.
            output_dir = os.path.abspath(output_dir)

            # See Note [Absolute include_dirs]
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)

            _, objects, extra_postargs, pp_opts, _ = \
                self.compiler._setup_compile(output_dir, macros,
                                             include_dirs, sources,
                                             depends, extra_postargs)
            common_cflags = self.compiler._get_cc_args(pp_opts, debug, extra_preargs)
            extra_cc_cflags = self.compiler.compiler_so[1:]

            # extra_postargs can be either:
            # - a dict mapping cxx to extra flags
            # - a list of extra flags.
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs['cxx']
            else:
                post_cflags = list(extra_postargs)
            append_std17_if_no_std_present(post_cflags)

            _write_ninja_file_and_compile_objects(
                sources=sources,
                objects=objects,
                cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags],
                post_cflags=[shlex.quote(f) for f in post_cflags],
                build_directory=output_dir,
                verbose=True)

            # Return *all* object filenames, not just the ones we just built.
            return objects            

        if self.compiler.compiler_type == 'msvc':
            raise 'Not implemented'
        else:
            if self.use_ninja:
                self.compiler.compile = unix_wrap_ninja_compile
            else:
                self.compiler._compile = unix_wrap_single_compile

        build_ext.build_extensions(self)

    def _add_compile_flag(self, extension, flag):
        extension.extra_compile_args = copy.deepcopy(extension.extra_compile_args)
        if isinstance(extension.extra_compile_args, dict):
            for args in extension.extra_compile_args.values():
                args.append(flag)
        else:
            extension.extra_compile_args.append(flag)

    def _define_torch_extension_name(self, extension):
        # pybind11 doesn't support dots in the names
        # so in order to support extensions in the packages
        # like torch._C, we take the last part of the string
        # as the library name
        names = extension.name.split('.')
        name = names[-1]
        define = f'-DTORCH_EXTENSION_NAME={name}'
        self._add_compile_flag(extension, define)

    def _add_gnu_cpp_abi_flag(self, extension):
        # use the same CXX ABI as what PyTorch was compiled with
        self._add_compile_flag(extension, '-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)))

SUBPROCESS_DECODE_ARGS = ('oem',) if IS_WINDOWS else ()

ABI_INCOMPATIBILITY_WARNING = '''
                               !! WARNING !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Your compiler ({}) may be ABI-incompatible with PyTorch!
Please use a compiler that is ABI-compatible with GCC 5.0 and above.
See https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html.
See https://gist.github.com/goldsborough/d466f43e8ffc948ff92de7486c5216d6
for instructions on how to install GCC 5 or higher.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                              !! WARNING !!
'''
WRONG_COMPILER_WARNING = '''
                               !! WARNING !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Your compiler ({user_compiler}) is not compatible with the compiler Pytorch was
built with for this platform, which is {pytorch_compiler} on {platform}. Please
use {pytorch_compiler} to to compile your extension. Alternatively, you may
compile PyTorch from source using {user_compiler}, and then you can also use
{user_compiler} to compile your extension.
See https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md for help
with compiling PyTorch from source.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                              !! WARNING !!
'''

BUILT_FROM_SOURCE_VERSION_PATTERN = re.compile(r'\d+\.\d+\.\d+\w+\+\w+')

def _is_binary_build() -> bool:
    return not BUILT_FROM_SOURCE_VERSION_PATTERN.match(torch.version.__version__)


def _accepted_compilers_for_platform() -> List[str]:
    # gnu-c++ and gnu-cc are the conda gcc compilers
    return ['clang++', 'clang'] if IS_MACOS else ['dpcpp', 'icx']

def check_compiler_ok_for_platform(compiler: str) -> bool:
    r'''
    Verifies that the compiler is the expected one for the current platform.
    Args:
        compiler (str): The compiler executable to check.
    Returns:
        True if the compiler is gcc/g++ on Linux or clang/clang++ on macOS,
        and always True for Windows.
    '''

    if IS_WINDOWS:
        return True
    which = subprocess.check_output(['which', compiler], stderr=subprocess.STDOUT)
    # Use os.path.realpath to resolve any symlinks, in particular from 'c++' to e.g. 'g++'.
    compiler_path = os.path.realpath(which.decode(*SUBPROCESS_DECODE_ARGS).strip())

    # Check the compiler name
    if any(name in compiler_path for name in _accepted_compilers_for_platform()):
        return True
    # If compiler wrapper is used try to infer the actual compiler by invoking it with -v flag
    version_string = subprocess.check_output([compiler, '-v'], stderr=subprocess.STDOUT).decode(*SUBPROCESS_DECODE_ARGS)
    if IS_LINUX:
        # Check for 'gcc' or 'g++' for sccache warpper
        pattern = re.compile("^COLLECT_GCC=(.*)$", re.MULTILINE)
        results = re.findall(pattern, version_string)
        if len(results) != 1:
            return False
        compiler_path = os.path.realpath(results[0].strip())
        # On RHEL/CentOS c++ is a gcc compiler wrapper
        if os.path.basename(compiler_path) == 'c++' and 'gcc version' in version_string:
            return True
        return any(name in compiler_path for name in _accepted_compilers_for_platform())
    if IS_MACOS:
        # Check for 'clang' or 'clang++'
        return version_string.startswith("Apple clang")
    return False

def get_compiler_abi_compatibility_and_version(compiler) -> Tuple[bool, TorchVersion]:
    r'''
    Determine if the given compiler is ABI-compatible with PyTorch alongside
    its version.
    Args:
        compiler (str): The compiler executable name to check (e.g. ``g++``).
            Must be executable in a shell process.
    Returns:
        A tuple that contains a boolean that defines if the compiler is (likely) ABI-incompatible with PyTorch,
        followed by a `TorchVersion` string that contains the compiler version separated by dots.
    '''
    if not _is_binary_build():
        return (True, TorchVersion('0.0.0'))
    if os.environ.get('TORCH_DONT_CHECK_COMPILER_ABI') in ['ON', '1', 'YES', 'TRUE', 'Y']:
        return (True, TorchVersion('0.0.0'))

    # First check if the compiler is one of the expected ones for the particular platform.
    if not check_compiler_ok_for_platform(compiler):
        warnings.warn(WRONG_COMPILER_WARNING.format(
            user_compiler=compiler,
            pytorch_compiler=_accepted_compilers_for_platform()[0],
            platform=sys.platform))
        return (False, TorchVersion('0.0.0'))

    if IS_MACOS:
        # There is no particular minimum version we need for clang, so we're good here.
        return (True, TorchVersion('0.0.0'))
    try:
        if IS_LINUX:
            minimum_required_version = MINIMUM_GCC_VERSION
            versionstr = subprocess.check_output([compiler, '-dumpfullversion', '-dumpversion'])
            version = versionstr.decode(*SUBPROCESS_DECODE_ARGS).strip().split('.')
        else:
            minimum_required_version = MINIMUM_MSVC_VERSION
            compiler_info = subprocess.check_output(compiler, stderr=subprocess.STDOUT)
            match = re.search(r'(\d+)\.(\d+)\.(\d+)', compiler_info.decode(*SUBPROCESS_DECODE_ARGS).strip())
            version = ['0', '0', '0'] if match is None else list(match.groups())
    except Exception:
        _, error, _ = sys.exc_info()
        warnings.warn(f'Error checking compiler version for {compiler}: {error}')
        return (False, TorchVersion('0.0.0'))

    if tuple(map(int, version)) >= minimum_required_version:
        return (True, TorchVersion('.'.join(version)))

    compiler = f'{compiler} {".".join(version)}'
    warnings.warn(ABI_INCOMPATIBILITY_WARNING.format(compiler))

    return (False, TorchVersion('.'.join(version)))

def _write_ninja_file_and_compile_objects(
        sources: List[str],
        objects,
        cflags,
        post_cflags,
        build_directory: str,
        verbose: bool) -> None:
    verify_ninja_availability()
    if IS_WINDOWS:
        compiler = os.environ.get('CXX', 'cl')
    else:
        compiler = get_dpcpp_complier()
    get_compiler_abi_compatibility_and_version(compiler)

    build_file_path = os.path.join(build_directory, 'build.ninja')
    if verbose:
        print(f'Emitting ninja build file {build_file_path}...')
    _write_ninja_file(
        path=build_file_path,
        cflags=cflags,
        post_cflags=post_cflags,
        sources=sources,
        objects=objects,
        ldflags=None,
        library_target=None)
    if verbose:
        print('Compiling objects...')
    _run_ninja_build(
        build_directory,
        verbose,
        # It would be better if we could tell users the name of the extension
        # that failed to build but there isn't a good way to get it here.
        error_prefix='Error compiling objects for extension')

PLAT_TO_VCVARS = {
    'win32' : 'x86',
    'win-amd64' : 'x86_amd64',
}

def _get_num_workers(verbose: bool) -> Optional[int]:
    max_jobs = os.environ.get('MAX_JOBS')
    if max_jobs is not None and max_jobs.isdigit():
        if verbose:
            print(f'Using envvar MAX_JOBS ({max_jobs}) as the number of workers...')
        return int(max_jobs)
    if verbose:
        print('Allowing ninja to set a default number of workers... '
              '(overridable by setting the environment variable MAX_JOBS=N)')
    return None

def _run_ninja_build(build_directory: str, verbose: bool, error_prefix: str) -> None:
    command = ['ninja', '-v']
    num_workers = _get_num_workers(verbose)
    if num_workers is not None:
        command.extend(['-j', str(num_workers)])
    env = os.environ.copy()
    # Try to activate the vc env for the users
    if IS_WINDOWS and 'VSCMD_ARG_TGT_ARCH' not in env:
        from setuptools import distutils

        plat_name = distutils.util.get_platform()
        plat_spec = PLAT_TO_VCVARS[plat_name]

        vc_env = distutils._msvccompiler._get_vc_env(plat_spec)
        vc_env = {k.upper(): v for k, v in vc_env.items()}
        for k, v in env.items():
            uk = k.upper()
            if uk not in vc_env:
                vc_env[uk] = v
        env = vc_env
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        # Warning: don't pass stdout=None to subprocess.run to get output.
        # subprocess.run assumes that sys.__stdout__ has not been modified and
        # attempts to write to it by default.  However, when we call _run_ninja_build
        # from ahead-of-time cpp extensions, the following happens:
        # 1) If the stdout encoding is not utf-8, setuptools detachs __stdout__.
        #    https://github.com/pypa/setuptools/blob/7e97def47723303fafabe48b22168bbc11bb4821/setuptools/dist.py#L1110
        #    (it probably shouldn't do this)
        # 2) subprocess.run (on POSIX, with no stdout override) relies on
        #    __stdout__ not being detached:
        #    https://github.com/python/cpython/blob/c352e6c7446c894b13643f538db312092b351789/Lib/subprocess.py#L1214
        # To work around this, we pass in the fileno directly and hope that
        # it is valid.
        stdout_fileno = 1
        subprocess.run(
            command,
            stdout=stdout_fileno if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=build_directory,
            check=True,
            env=env)
    except subprocess.CalledProcessError as e:
        # Python 2 and 3 compatible way of getting the error object.
        _, error, _ = sys.exc_info()
        # error.output contains the stdout and stderr of the build attempt.
        message = error_prefix
        # `error` is a CalledProcessError (which has an `ouput`) attribute, but
        # mypy thinks it's Optional[BaseException] and doesn't narrow
        if hasattr(error, 'output') and error.output:  # type: ignore[union-attr]
            message += f": {error.output.decode(*SUBPROCESS_DECODE_ARGS)}"  # type: ignore[union-attr]
        raise RuntimeError(message) from e

def _write_ninja_file(path,
                      cflags,
                      post_cflags,
                      sources,
                      objects,
                      ldflags,
                      library_target) -> None:
    r"""Write a ninja file that does the desired compiling and linking.
    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case,
                      we do no linking.
    """
    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    ldflags = sanitize_flags(ldflags)

    # Sanity checks...
    assert len(sources) == len(objects)
    assert len(sources) > 0

    if IS_WINDOWS:
        compiler = os.environ.get('CXX', 'cl')
    else:
        compiler = get_dpcpp_complier()

    # Version 1.3 is required for the `deps` directive.
    config = ['ninja_required_version = 1.3']
    config.append(f'cxx = {compiler}')

    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ['rule compile']
    if IS_WINDOWS:
        compile_rule.append(
            '  command = cl /showIncludes $cflags -c $in /Fo$out $post_cflags')
        compile_rule.append('  deps = msvc')
    else:
        compile_rule.append(
            '  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags')
        compile_rule.append('  depfile = $out.d')
        compile_rule.append('  deps = gcc')

    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects):
        rule = 'compile'
        if IS_WINDOWS:
            source_file = source_file.replace(':', '$:')
            object_file = object_file.replace(':', '$:')
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f'build {object_file}: {rule} {source_file}')

    if library_target is not None:
        link_rule = ['rule link']
        if IS_WINDOWS:
            cl_paths = subprocess.check_output(['where',
                                                'cl']).decode(*SUBPROCESS_DECODE_ARGS).split('\r\n')
            if len(cl_paths) >= 1:
                cl_path = os.path.dirname(cl_paths[0]).replace(':', '$:')
            else:
                raise RuntimeError("MSVC is required to load C++ extensions")
            link_rule.append(f'  command = "{cl_path}/link.exe" $in /nologo $ldflags /out:$out')
        else:
            link_rule.append('  command = $cxx $in $ldflags -o $out')

        link = [f'build {library_target}: link {" ".join(objects)}']

        default = [f'default {library_target}']
    else:
        link_rule, link, default = [], [], []

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]

    blocks += [link_rule, build, link, default]
    with open(path, 'w') as build_file:
        for block in blocks:
            lines = '\n'.join(block)
            build_file.write(f'{lines}\n\n')

def _get_dpcpp_root():
    # TODO: Need to decouple with toolchain env scripts
    dpcpp_root = os.getenv('CMPLR_ROOT')
    return dpcpp_root

def _get_onemkl_root():
    # TODO: Need to decouple with toolchain env scripts
    path = os.getenv('MKLROOT')
    return path

def _get_onednn_root():
    # TODO: Need to decouple with toolchain env scripts
    path = os.getenv('DNNLROOT')
    return path

class _one_api_help:
    __dpcpp_root = None
    __onemkl_root = None
    __onednn_root = None
    __default_root = None

    def __init__(self):
        self.__dpcpp_root = _get_dpcpp_root()
        self.__onemkl_root = _get_onemkl_root()
        self.__onednn_root = _get_onednn_root()

        CUR_DIR = os.path.dirname(__file__)
        self.__default_root = os.path.dirname(CUR_DIR)

        self.check_onednn_cfg()
        self.check_dpcpp_cfg()
        self.check_onemkl_cfg()

    def check_onemkl_cfg(self):
        if self.__onemkl_root is None:
            raise 'Didn\'t detect mkl root. Please source <oneapi_dir>/mkl/<version>/env/vars.sh '

    def check_onednn_cfg(self):
        if self.__onednn_root is None:
            raise 'Didn\'t detect dnnl root. Please source <oneapi_dir>/dnnl/<version>/env/vars.sh '
        else:
            warnings.warn("This extension has static linked onednn library. Please attaction to that, this path of onednn version maybe not match with the built-in version.")

    def check_dpcpp_cfg(self):
        if self.__dpcpp_root is None:
            raise 'Didn\'t detect dpcpp root. Please source <oneapi_dir>/compiler/<version>/env/vars.sh '        

    def get_default_include_dir(self):
        return [os.path.join(self.__default_root, 'include')]

    def get_default_lib_dir(self):
        return [os.path.join(self.__default_root, 'lib')]

    def get_dpcpp_include_dir(self):
        return [
            os.path.join(self.__dpcpp_root, 'linux', 'include'),
            os.path.join(self.__dpcpp_root, 'linux', 'include', 'sycl')
        ]

    def get_onemkl_include_dir(self):
        return [os.path.join(self.__onemkl_root, 'include')]

    def get_onednn_include_dir(self):
        return [os.path.join(self.__onednn_root, 'include')]

    def get_onednn_lib_dir(self):
        return [os.path.join(self.__onednn_root, 'lib')]

    def is_onemkl_ready(self):
        if self.__onemkl_root is None:
            return False
        return True

    def is_onednn_ready(self):
        if self.__onednn_root is None:
            return False
        return True

    def get_library_dirs(self):
        library_dirs = []
        library_dirs += [f'{x}' for x in self.get_default_lib_dir()]
        library_dirs += [f'{x}' for x in self.get_onednn_lib_dir()]
        return library_dirs

    def get_include_dirs(self):
        include_dirs = []
        include_dirs += [f'{x}' for x in self.get_dpcpp_include_dir()]
        include_dirs += [f'{x}' for x in self.get_onemkl_include_dir()]
        include_dirs += [f'{x}' for x in self.get_onednn_include_dir()]
        include_dirs += [f'{x}' for x in self.get_default_include_dir()]
        return include_dirs

    def get_onemkl_libraries(self):
        MKLROOT = self.__onemkl_root
        return [
            f'{MKLROOT}/lib/intel64/libmkl_sycl.a',
            f'{MKLROOT}/lib/intel64/libmkl_intel_ilp64.a',
            f'{MKLROOT}/lib/intel64/libmkl_sequential.a',
            f'{MKLROOT}/lib/intel64/libmkl_core.a',
        ]

def get_pytorch_include_dir():
    lib_include = os.path.join(_TORCH_PATH, 'include')
    paths = [
        lib_include,
        # Remove this once torch/torch.h is officially no longer supported for C++ extensions.
        os.path.join(lib_include, 'torch', 'csrc', 'api', 'include'),
        # Some internal (old) Torch headers don't properly prefix their includes,
        # so we need to pass -Itorch/lib/include/TH as well.
        os.path.join(lib_include, 'TH')
    ]
    return paths

def get_pytorch_lib_dir():
    return [os.path.join(_TORCH_PATH, 'lib')]


def DPCPPExtension(name, sources, *args, **kwargs):
    r'''
    Creates a :class:`setuptools.Extension` for DPCPP/C++.
    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a DPCPP/C++
    extension.
    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.
    Example:
        >>> from intel_extension_for_pytorch.xpu.utils import DpcppBuildExtension, DPCPPExtension
        >>> setup(
                name='dpcpp_extension',
                ext_modules=[
                    DPCPPExtension(
                            name='dpcpp_extension',
                            sources=['extension.cpp', 'extension_kernel.cpp'],
                            extra_compile_args={'cxx': ['-g', '-std=c++20', '-fPIC']})
                ],
                cmdclass={
                    'build_ext': DpcppBuildExtension
                })
    '''

    oneAPI = _one_api_help()

    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += get_pytorch_lib_dir()
    library_dirs += oneAPI.get_library_dirs()
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    libraries.append('torch_python')

    # Append oneDNN link parameters.
    libraries.append('dnnl')
    kwargs['libraries'] = libraries

    include_dirs = kwargs.get('include_dirs', [])
    include_dirs += get_pytorch_include_dir()
    include_dirs += oneAPI.get_include_dirs()
    kwargs['include_dirs'] = include_dirs

    kwargs['language'] = 'c++'

    extra_compile_args = kwargs.get('extra_compile_args', {})
    extra_link_args = kwargs.get('extra_link_args', [])    

    # Append oneMKL link parameters, detailed please reference:
    # https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html
    extra_link_args += [f'-L{x}' for x in library_dirs]
    # extra_link_args += ['-fsycl-device-code-split=per_kernel']
    extra_link_args += ['-Wl,--start-group']
    extra_link_args += [f'{x}' for x in oneAPI.get_onemkl_libraries()]
    extra_link_args += ['-Wl,--end-group']
    extra_link_args += ['-lsycl', '-lOpenCL', '-lpthread', '-lm', '-ldl']

    # Append IPEX link parameters.
    extra_link_args += [f'-L{x}' for x in oneAPI.get_default_lib_dir()]
    extra_link_args += ['-lintel-ext-pt-gpu']

    # todo: add dpcpp parameter support.
    kwargs['extra_link_args'] = extra_link_args
    kwargs['extra_compile_args'] = extra_compile_args

    return setuptools.Extension(name, sources, *args, **kwargs)
