import os
import torch
from . import _C
from ._version import (__version__, __ipex_git_sha__,
                       __torch_version__, __torch_git_sha__,
                       __ittapi_git_sha__, __onednn_git_sha__)

from . import optim


def version():
    print("intel_extension_for_pytorch gpu version:          {}".format(__version__))
    print("intel_extension_for_pytorch gpu git sha:          {}".format(__ipex_git_sha__))
    print("private gpu torch version: {}".format(__torch_version__))
    print("private gpu torch sha:     {}".format(__torch_git_sha__))
    print("submodule ittapi sha:      {}".format(__ittapi_git_sha__))
    print("submodule oneDNN sha:      {}".format(__onednn_git_sha__))


def _find_dpcpp_home():
    pass


def _here_paths():
    here = os.path.abspath(__file__)
    ipex_path = os.path.dirname(here)
    return ipex_path


def include_paths():
    '''
    Get the include paths required to build a C++ extension.

    Returns:
        A list of include path strings.
    '''
    ipex_path = _here_paths()
    lib_include = os.path.join(ipex_path, 'include')
    paths = [
        lib_include,
        # os.path.join(lib_include, 'more')
    ]
    return paths


def library_paths():
    '''
    Get the library paths required to build a C++ extension.

    Returns:
        A list of library path strings.
    '''
    ipex_path = _here_paths()
    lib_path = os.path.join(ipex_path, 'lib')
    paths = [
        lib_path,
    ]
    return paths


# Path to folder containing CMake definitions for torch ipex package
cmake_prefix_path = os.path.join(os.path.dirname(__file__), 'share', 'cmake')

import intel_extension_for_pytorch.xpu
