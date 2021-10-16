import os
import torch
from ._utils import _get_device_index  # , _dummy_type
from . import _C
from .version import __version__, __ipex_gitrev__

from . import itt as itt
from . import optim
from . import profiler as profiler
from .autograd import inference_mode


def version():
    version = __version__.split('+')[0]
    print("ipex gpu version: {}".format(version))
    print("ipex gpu git sha: {}".format(__ipex_gitrev__))


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

import ipex.xpu
