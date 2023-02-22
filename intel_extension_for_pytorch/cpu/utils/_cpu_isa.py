# coding: utf-8
import intel_extension_for_pytorch
import subprocess
import platform
import sys
import os

def check_avx2_support():
    def get_normalized_bit(value, bit_index):
        return (value >> bit_index) & 1

    try:
        # https://pypi.org/project/cpuid/
        import cpuid
    except ImportError:
        raise Exception(f"unable to import cpuid, please install it via pypi.")

    eax, ebx, ecx, edx = cpuid.cpuid(0)
    max_basic_id =eax
    if max_basic_id >= 7:
        # https://github.com/fpelliccioni/cpuid-py/blob/master/cpuid.py#L8
        eax, ebx, ecx, edx = cpuid.cpuid_count(7, 0)
        support_avx2 = get_normalized_bit(ebx, 5)

        if support_avx2 == 0:
            return False
        
        xcrFeatureMask = cpuid.xgetbv(0)
        '''
        Intel® 64 and IA-32 Architectures
        Software Developer’s Manual
        Combined Volumes:
        1, 2A, 2B, 2C, 2D, 3A, 3B, 3C, 3D and 4
        Order Number: 325462-075US
        June 2021
        ----------------------------------------------------
        13.1 XSAVE-SUPPORTED FEATURES AND STATE-COMPONENT BITMAPS
        ......
        Bit 1 corresponds to the state component used for registers used by the
        streaming SIMD extensions (SSE state). See Section 13.5.2. 
        Bit 2 corresponds to the state component used for the additional register 
        state used by the Intel® Advanced Vector Extensions (AVX state). 
        See Section 13.5.3
        '''
        if get_normalized_bit(xcrFeatureMask, 1) and get_normalized_bit(xcrFeatureMask, 2):
            return True

    return False

def check_minimal_isa_support():
    if not intel_extension_for_pytorch._C._has_cpu():
        return
    err_msg = "ERROR! Intel® Extension for PyTorch* only works on machines with instruction sets equal or newer than AVX2, which are not detected on the current machine."
    if not check_avx2_support():
        sys.exit(err_msg)
