import os
import setuptools
import torch
import warnings
from torch.utils.cpp_extension import _TORCH_PATH, BuildExtension, CppExtension
from pkg_resources import packaging

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

    def is_xpu_compiler_ready(self):
        # TODO: Need to decouple with toolchain env scripts
        # and according to OS to find real path of DPCPP compiler

        cc_cfg = os.getenv('CC')
        cxx_cfg = os.getenv('CXX')
        if cc_cfg is None or cxx_cfg is None:
            raise 'Please use corrent compiler for XPU build, via CC=icx CXX=dpcpp.'
        # Intel XPU use DPCPP as compiler.
        if cc_cfg == 'icx' and cxx_cfg == 'dpcpp':
            return True
        return False

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
    oneAPI = _one_api_help()
    if oneAPI.is_xpu_compiler_ready() is not True:
        raise 'Please use dpcpp for XPU build.'

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
    extra_link_args += ['-fsycl-device-code-split=per_kernel']
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
