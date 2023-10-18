# This Python file uses the following encoding: utf-8

import intel_extension_for_pytorch._isa_help as isa
import sys


def check_avx2_support():
    return isa._check_isa_avx2()


def check_minimal_isa_support():
    err_msg = "ERROR! Intel® Extension for PyTorch* only works on machines with instruction sets equal or newer \
        than AVX2, which are not detected on the current machine."
    if not check_avx2_support():
        sys.exit(err_msg)
