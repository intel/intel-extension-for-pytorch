ISA Dynamic Dispatching
=======================

This document explains the dynamic kernel dispatch mechanism for Intel® Extension for PyTorch\* (Intel® Extension for PyTorch\*) based on CPU ISA. It is an extension to the similar mechanism in PyTorch.

## Overview

Forked from PyTorch, Intel® Extension for PyTorch\* adds additional CPU ISA level support, such as `AVX512_VNNI`, `AVX512_BF16` and `AMX`.

PyTorch & Intel® Extension for PyTorch\* CPU ISA support statement:

 | | DEFAULT | AVX2 | AVX2_VNNI | AVX512 | AVX512_VNNI | AVX512_BF16 | AMX |
 | ---- | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
 | PyTorch | ✔ | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
 | Intel® Extension for PyTorch\* 1.11 | ✘ | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
 | Intel® Extension for PyTorch\* 1.12 | ✘ | ✔ | ✘ | ✔ | ✔ | ✔ | ✔ |

\* `DEFAULT` in Intel® Extension for PyTorch\* 1.12 implies `AVX2`.

### CPU ISA build compiler requirement

 | ISA Level | GCC requirement |
 | ---- | :----: |
 | AVX2 | Any |
 | AVX512 | GCC 9.2+ |
 | AVX512_VNNI | GCC 9.2+ |
 | AVX512_BF16 | GCC 10.3+ |
 | AVX2_VNNI | GCC 11.2+ |
 | AMX | GCC 11.2+ |

\* Check with `cmake/Modules/FindAVX.cmake` for detailed compiler checks.

## Select ISA Level

By default, Intel® Extension for PyTorch\* dispatches to kernels with the maximum ISA level supported on the underlying CPU hardware. This ISA level can be overridden by an environment variable `ATEN_CPU_CAPABILITY` (same environment variable as PyTorch). Available values are {`avx2`, `avx512`, `avx512_vnni`, `avx512_bf16`, `amx`}. The effective ISA level would be the minimal level between `ATEN_CPU_CAPABILITY` and the maximum level supported by the hardware.

### Example:

```bash
$ python -c 'import intel_extension_for_pytorch._C as core;print(core._get_current_isa_level())'
AMX
$ ATEN_CPU_CAPABILITY=avx2 python -c 'import intel_extension_for_pytorch._C as core;print(core._get_current_isa_level())'
AVX2
```
>**Note:**
>
>`core._get_current_isa_level()` is an Intel® Extension for PyTorch\* internal function used for checking the current effective ISA level. It is used for debugging purpose only and subject to change.

## CPU feature check

An addtional CPU feature check tool in the subfolder: `tests/cpu/isa`

```bash
$ cmake .
-- The C compiler identification is GNU 11.2.1
-- The CXX compiler identification is GNU 11.2.1
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /opt/rh/gcc-toolset-11/root/usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /opt/rh/gcc-toolset-11/root/usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: tests/cpu/isa

$ make
[ 33%] Building CXX object CMakeFiles/cpu_features.dir/intel_extension_for_pytorch/csrc/cpu/isa/cpu_feature.cpp.o
[ 66%] Building CXX object CMakeFiles/cpu_features.dir/intel_extension_for_pytorch/csrc/cpu/isa/cpu_feature_main.cpp.o
[100%] Linking CXX executable cpu_features
[100%] Built target cpu_features

$ ./cpu_features
XCR0: 00000000000602e7
os --> avx: true
os --> avx2: true
os --> avx512: true
os --> amx: true
mmx:                    true
sse:                    true
sse2:                   true
sse3:                   true
ssse3:                  true
sse4_1:                 true
sse4_2:                 true
aes_ni:                 true
sha:                    true
xsave:                  true
fma:                    true
f16c:                   true
avx:                    true
avx2:                   true
avx_vnni:                       true
avx512_f:                       true
avx512_cd:                      true
avx512_pf:                      false
avx512_er:                      false
avx512_vl:                      true
avx512_bw:                      true
avx512_dq:                      true
avx512_ifma:                    true
avx512_vbmi:                    true
avx512_vpopcntdq:               true
avx512_4fmaps:                  false
avx512_4vnniw:                  false
avx512_vbmi2:                   true
avx512_vpclmul:                 true
avx512_vnni:                    true
avx512_bitalg:                  true
avx512_fp16:                    true
avx512_bf16:                    true
avx512_vp2intersect:            true
amx_bf16:                       true
amx_tile:                       true
amx_int8:                       true
prefetchw:                      true
prefetchwt1:                    false
```
