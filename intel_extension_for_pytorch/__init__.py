# This Python file uses the following encoding: utf-8
import re

import torch


try:
    import torchvision
except ImportError:
    pass  # skip if torchvision is not available


from ._version import (
    __version__,
    __ipex_gitrev__,
    __torch_gitrev__,
    __gpu_onednn_gitrev__,
    __cpu_ideep_gitrev__,
    __build_type__,
)

torch_version = ""
ipex_version = ""
matches = re.match(r"(\d+\.\d+).*", torch.__version__)
if matches and len(matches.groups()) == 1:
    torch_version = matches.group(1)
matches = re.match(r"(\d+\.\d+).*", __version__)
if matches and len(matches.groups()) == 1:
    ipex_version = matches.group(1)
if torch_version == "" or ipex_version == "" or torch_version != ipex_version:
    print(
        "ERROR! Intel® Extension for PyTorch* needs to work with PyTorch "
        + f"{ipex_version}.*, but PyTorch {torch.__version__} is found. "
        + "Please switch to the matching version and run again."
    )
    exit(127)


import os
import sys
import glob
import ctypes
import platform

################################################################################
# Load the extension module
################################################################################

if sys.platform == "win32":
    pfiles_path = os.getenv("ProgramFiles", "C:\\Program Files")
    py_dll_path = os.path.join(sys.exec_prefix, "Library", "bin")
    th_dll_path = os.path.join(os.path.dirname(__file__), "bin")

    # When users create a virtualenv that inherits the base environment,
    # we will need to add the corresponding library directory into
    # DLL search directories. Otherwise, it will rely on `PATH` which
    # is dependent on user settings.
    if sys.exec_prefix != sys.base_exec_prefix:
        base_py_dll_path = os.path.join(sys.base_exec_prefix, "Library", "bin")
    else:
        base_py_dll_path = ""

    dll_paths = list(
        filter(os.path.exists, [th_dll_path, py_dll_path, base_py_dll_path])
    )

    kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    with_load_library_flags = hasattr(kernel32, "AddDllDirectory")
    prev_error_mode = kernel32.SetErrorMode(0x0001)

    kernel32.LoadLibraryW.restype = ctypes.c_void_p
    if with_load_library_flags:
        kernel32.AddDllDirectory.restype = ctypes.c_void_p
        kernel32.LoadLibraryExW.restype = ctypes.c_void_p

    for dll_path in dll_paths:
        if sys.version_info >= (3, 8):
            os.add_dll_directory(dll_path)
        elif with_load_library_flags:
            res = kernel32.AddDllDirectory(dll_path)
            if res is None:
                err = ctypes.WinError(ctypes.get_last_error())
                err.strerror += f' Error adding "{dll_path}" to the DLL directories.'
                raise err

    try:
        ctypes.CDLL("vcruntime140.dll")
        ctypes.CDLL("msvcp140.dll")
        ctypes.CDLL("vcruntime140_1.dll")
    except OSError:
        print(
            """Microsoft Visual C++ Redistributable is not installed, this may lead to the DLL load failure.
                 It can be downloaded at https://aka.ms/vs/16/release/vc_redist.x64.exe"""
        )

    dlls = glob.glob(os.path.join(th_dll_path, "*.dll"))
    path_patched = False
    for dll in dlls:
        is_loaded = False
        if with_load_library_flags:
            res = kernel32.LoadLibraryExW(dll, None, 0x00001100)
            last_error = ctypes.get_last_error()
            if res is None and last_error != 126:
                err = ctypes.WinError(last_error)
                err.strerror += f' Error loading "{dll}" or one of its dependencies.'
                raise err
            elif res is not None:
                is_loaded = True
        if not is_loaded:
            if not path_patched:
                os.environ["PATH"] = ";".join(dll_paths + [os.environ["PATH"]])
                path_patched = True
            res = kernel32.LoadLibraryW(dll)
            if res is None:
                err = ctypes.WinError(ctypes.get_last_error())
                err.strerror += f' Error loading "{dll}" or one of its dependencies.'
                raise err

    kernel32.SetErrorMode(prev_error_mode)
from . import llm
from . import cpu
from . import xpu
from . import quantization
from . import nn
from . import jit
from . import optim
from . import fx
from . import _dynamo
from . import _meta_registrations
from ._init_on_device import OnDevice
from .utils._logger import logger, WarningType

try:
    from .cpu import tpp
except BaseException:
    logger.warn(
        "Please install transformers repo when you want to use fast_bert API.",
        _type=WarningType.MissingArgument,
    )

from .frontend import optimize
from .transformers import (
    optimize_transformers,
    _set_optimized_model_for_generation,
)
from . import distributed
from .frontend import enable_auto_channels_last, disable_auto_channels_last
from .frontend import set_fp32_math_mode, get_fp32_math_mode, FP32MathMode
from .cpu._auto_kernel_selection import _enable_dnnl, _disable_dnnl, _using_dnnl
from .cpu.utils.verbose import verbose, VERBOSE_OFF, VERBOSE_ON, VERBOSE_ON_CREATION
from .cpu.tpp.fused_bert import fast_bert
from ._inductor.compiler import _set_compiler_backend, _get_compiler_backend, compile
from .cpu.onednn_fusion import enable_onednn_fusion

from . import _C

# Path to folder containing CMake definitions for torch ipex package
cmake_prefix_path = os.path.join(os.path.dirname(__file__), "share", "cmake")


from .cpu.utils import _cpu_isa, _custom_fx_tracer

_cpu_isa.check_minimal_isa_support()


def version():
    print("intel_extension_for_pytorch version:          {}".format(__version__))
    print("intel_extension_for_pytorch git sha:          {}".format(__ipex_gitrev__))
    if len(__torch_gitrev__) != 0:
        print("torch version and sha:     {}".format(__torch_gitrev__))
    print("submodule oneDNN sha:      {}".format(__gpu_onednn_gitrev__))
    print("submodule ideep sha:      {}".format(__cpu_ideep_gitrev__))
