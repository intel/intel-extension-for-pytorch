# Intel® Extension for PyTorch\* CPU ISA Dynamic Dispatch Design Doc

This document explains the dynamic kernel dispatch mechanism for Intel® Extension for PyTorch\* (IPEX) based on CPU ISA. It is an extension to the similar mechanism in PyTorch.

## Overview

IPEX dyndisp is forked from **PyTorch:** `ATen/native/DispatchStub.h` and `ATen/native/DispatchStub.cpp`. IPEX adds additional CPU ISA level support, such as `AVX512_VNNI`, `AVX512_BF16` and `AMX`.

PyTorch & IPEX CPU ISA support statement:

 | | DEFAULT | AVX2 | AVX2_VNNI | AVX512 | AVX512_VNNI | AVX512_BF16 | AMX | AVX512_FP16
 | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
 | PyTorch | ✔ | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ | ✘ |
 | IPEX-1.11 | ✘ | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ | ✘ |
 | IPEX-1.12 | ✘ | ✔ | ✘ | ✔ | ✔ | ✔ | ✔ | ✘ |
 | IPEX-1.13 | ✘ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✘ |
 | IPEX-2.1 | ✘ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
 | IPEX-2.2 | ✘ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |

\* Current IPEX DEFAULT level implemented as same as AVX2 level.

### CPU ISA build compiler requirement
 | ISA Level | GCC requirement |
 | ---- | ---- |
 | AVX2 | Any |
 | AVX512 | GCC 9.2+ |
 | AVX512_VNNI | GCC 9.2+ |
 | AVX512_BF16 | GCC 10.3+ |
 | AVX2_VNNI | GCC 11.2+ |
 | AMX | GCC 11.2+ |
 | AVX512_FP16 | GCC 12.1+ |

\* Check with `cmake/Modules/FindAVX.cmake` for detailed compiler checks.

## Dynamic Dispatch Design

Dynamic dispatch copies the kernel implementation source files to multiple folders for each ISA level. It then builds each file using its ISA specific parameters. Each generated object file will contain its function body (**Kernel Implementation**).

Kernel Implementation uses an anonymous namespace so that different CPU versions won't conflict.

**Kernel Stub** is a "virtual function" with polymorphic kernel implementations pertaining to ISA levels.

At the runtime, **Dispatch Stub implementation** will check CPUIDs and OS status to determins which ISA level pointer best matches the function body.

### Code Folder Struct
>#### **Kernel implementation:** `csrc/cpu/aten/kernels/xyzKrnl.cpp`
>#### **Kernel Stub:** `csrc/cpu/aten/xyz.cpp` and `csrc/cpu/aten/xyz.h`
>#### **Dispatch Stub implementation:** `csrc/cpu/dyndisp/DispatchStub.cpp` and `csrc/cpu/dyndisp/DispatchStub.h`

### CodeGen Process
IPEX build system will generate code for each ISA level with specifiy complier parameters. The CodeGen script is located at `cmake/cpu/IsaCodegen.cmake`.

The CodeGen will copy each cpp files from **Kernel implementation**, and then add ISA level as new file suffix.

> **Sample:**
>
> ----
>
> **Origin file:**
>
> `csrc/cpu/aten/kernels/AdaptiveAveragePoolingKrnl.cpp`
>
> **Generate files:**
>
> DEFAULT: `build/Release/csrc/isa_codegen/cpu/aten/kernels/AdaptiveAveragePoolingKrnl.cpp.DEFAULT.cpp -O3 -D__AVX__ -DCPU_CAPABILITY_AVX2 -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT`
>
> AVX2: `build/Release/csrc/isa_codegen/cpu/aten/kernels/AdaptiveAveragePoolingKrnl.cpp.AVX2.cpp -O3 -D__AVX__ -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2`
>
> AVX512: `build/Release/csrc/isa_codegen/cpu/aten/kernels/AdaptiveAveragePoolingKrnl.cpp.AVX512.cpp -O3 -D__AVX512F__ -mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma -DCPU_CAPABILITY=AVX512 -DCPU_CAPABILITY_AVX512`
>
> AVX512_VNNI: `build/Release/csrc/isa_codegen/cpu/aten/kernels/AdaptiveAveragePoolingKrnl.cpp.AVX512_VNNI.cpp -O3 -D__AVX512F__ -DCPU_CAPABILITY_AVX512 -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mfma -DCPU_CAPABILITY=AVX512_VNNI -DCPU_CAPABILITY_AVX512_VNNI`
>
> AVX512_BF16: `build/Release/csrc/isa_codegen/cpu/aten/kernels/AdaptiveAveragePoolingKrnl.cpp.AVX512_BF16.cpp -O3 -D__AVX512F__ -DCPU_CAPABILITY_AVX512 -DCPU_CAPABILITY_AVX512_VNNI -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512bf16 -mfma -DCPU_CAPABILITY=AVX512_BF16 -DCPU_CAPABILITY_AVX512_BF16`
>
> AMX: `build/Release/csrc/isa_codegen/cpu/aten/kernels/AdaptiveAveragePoolingKrnl.cpp.AMX.cpp -O3  -D__AVX512F__ -DCPU_CAPABILITY_AVX512 -DCPU_CAPABILITY_AVX512_VNNI -DCPU_CAPABILITY_AVX512_BF16 -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512bf16 -mfma -mamx-tile -mamx-int8 -mamx-bf16 -DCPU_CAPABILITY=AMX -DCPU_CAPABILITY_AMX`
>
> AVX512_FP16: `build/Release/csrc/isa_codegen/cpu/aten/kernels/AdaptiveAveragePoolingKrnl.cpp.AVX512_FP16.cpp -O3  -D__AVX512F__ -DCPU_CAPABILITY_AVX512 -DCPU_CAPABILITY_AVX512_VNNI -DCPU_CAPABILITY_AVX512_BF16 -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512bf16 -mfma -mamx-tile -mamx-int8 -mamx-bf16 -mavx512fp16 -DCPU_CAPABILITY_AMX -DCPU_CAPABILITY=AVX512_FP16 -DCPU_CAPABILITY_AVX512_FP16`
---

>**Note:**
>1. DEFAULT level kernels is not fully implemented in IPEX. In order to align to PyTorch, we build default use AVX2 parameters in stead of that. So, IPEX minimal required executing machine support AVX2.
>2. `-D__AVX__` and `-D__AVX512F__` is defined for depends library [sleef](https://sleef.org/) .
>3. `-DCPU_CAPABILITY_AVX512` and `-DCPU_CAPABILITY_AVX2` are must to be defined for **PyTorch:** `aten/src/ATen/cpu/vec`, it determins vec register width.
>4. `-DCPU_CAPABILITY=[ISA_NAME]` is must to be defined for **PyTorch:** `aten/src/ATen/cpu/vec`, it is used as inline namespace name.
>5. Higher ISA level is compatible to lower ISA levels, so it needs to contains level ISA feature definitions. Such as AVX512_BF16 need contains `-DCPU_CAPABILITY_AVX512` `-DCPU_CAPABILITY_AVX512_VNNI`. But AVX512 don't contains AVX2 definitions, due to there are different vec register width.

## Add Custom Kernel

If you want to add a new custom kernel, and the kernel uses CPU ISA instructions, refer to these tips:

1. Add CPU ISA related kernel implementation to the folder:  `csrc/cpu/aten/kernels/NewKernelKrnl.cpp`
2. Add kernel stub to the folder: `csrc/cpu/aten/NewKernel.cpp`
3. Include header file: `csrc/cpu/dyndisp/DispatchStub.h`, and reference to the comment in the header file.
```c++
// Implements instruction set specific function dispatch.
//
// Kernels that may make use of specialized instruction sets (e.g. AVX2) are
// compiled multiple times with different compiler flags (e.g. -mavx2). A
// DispatchStub contains a table of function pointers for a kernel. At runtime,
// the fastest available kernel is chosen based on the features reported by
// cpuinfo.
//
// Example:
//
// In csrc/cpu/aten/MyKernel.h:
//   using fn_type = void(*)(const Tensor& x);
//   IPEX_DECLARE_DISPATCH(fn_type, stub);
//
// In csrc/cpu/aten/MyKernel.cpp
//   IPEX_DEFINE_DISPATCH(stub);
//
// In csrc/cpu/aten/kernels/MyKernel.cpp:
//   namespace {
//     // use anonymous namespace so that different cpu versions won't conflict
//     void kernel(const Tensor& x) { ... }
//   }
//   IPEX_REGISTER_DISPATCH(stub, &kernel);
//
// To call:
//   stub(kCPU, tensor);
```
4. Write the kernel follow the guide. It contains: declare function type, register stub, call stub, etc.

>**Note:**
>
>1. Some kernels only call **oneDNN** or **iDeep** implementation, or other backend implementation, which is not needed to add kernel implementations. (Refer: `BatchNorm.cpp`)
>2. Vec related header file must be included in kernel implementation files, but can not be included in kernel stub. Kernel stub is common code for all ISA level, and can't pass ISA related compiler parameters.
>3. For more intrinsics, check the [Intel® Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html).

### ISA intrinics specific kernel example:

This is a FP32 convert to BF16 function example, and it is implemented for `AVX512_BF16`, `AVX512` and `DEFAULT` ISA levels.

```c++
//csrc/cpu/aten/CvtFp32ToBf16.h

#pragma once

#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

void cvt_fp32_to_bf16(at::BFloat16* dst, const float* src, int len);

namespace {

void cvt_fp32_to_bf16_kernel_impl(at::BFloat16* dst, const float* src, int len);

}

using cvt_fp32_to_bf16_kernel_fn = void (*)(at::BFloat16*, const float*, int);
IPEX_DECLARE_DISPATCH(cvt_fp32_to_bf16_kernel_fn, cvt_fp32_to_bf16_kernel_stub);
} // namespace cpu
} // namespace torch_ipex

```
```c++
//csrc/cpu/aten/CvtFp32ToBf16.cpp

#include "CvtFp32ToBf16.h"

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(cvt_fp32_to_bf16_kernel_stub);

void cvt_fp32_to_bf16(at::BFloat16* dst, const float* src, int len) {
  return cvt_fp32_to_bf16_kernel_stub(kCPU, dst, src, len);
}

} // namespace cpu
} // namespace torch_ipex

```
Macro `CPU_CAPABILITY_AVX512` and `CPU_CAPABILITY_AVX512_BF16` are defined by compiler check, it is means that current compiler havs capability to generate defined ISA level code.

Because of `AVX512_BF16` is higher level than `AVX512`, and it compatible to `AVX512`. `CPU_CAPABILITY_AVX512_BF16` can be contained in `CPU_CAPABILITY_AVX512` region.
```c++
//csrc/cpu/aten/kernels/CvtFp32ToBf16Krnl.cpp

#include <ATen/cpu/vec/vec.h>
#include "csrc/aten/cpu/CvtFp32ToBf16.h"

namespace torch_ipex {
namespace cpu {

namespace {

#if defined(CPU_CAPABILITY_AVX512)
#include <ATen/cpu/vec/vec512/vec512.h>
#else
#include <ATen/cpu/vec/vec256/vec256.h>
#endif
using namespace at::vec;

#if defined(CPU_CAPABILITY_AVX512)
#include <immintrin.h>

inline __m256i _cvt_fp32_to_bf16(const __m512 src) {
#if (defined CPU_CAPABILITY_AVX512_BF16) // AVX512_BF16 ISA implementation.
  return reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(src));
#else  // AVX512 ISA implementation.
  __m512i value = _mm512_castps_si512(src);
  __m512i nan = _mm512_set1_epi32(0xffff);
  auto mask_value = _mm512_cmp_ps_mask(src, src, _CMP_ORD_Q);
  __m512i ones = _mm512_set1_epi32(0x1);
  __m512i vec_bias = _mm512_set1_epi32(0x7fff);
  // uint32_t lsb = (input >> 16) & 1;
  auto t_value = _mm512_and_si512(_mm512_srli_epi32(value, 16), ones);
  // uint32_t rounding_bias = 0x7fff + lsb;
  t_value = _mm512_add_epi32(t_value, vec_bias);
  // input += rounding_bias;
  t_value = _mm512_add_epi32(t_value, value);
  // input = input >> 16;
  t_value = _mm512_srli_epi32(t_value, 16);
  // Check NaN before converting back to bf16
  t_value = _mm512_mask_blend_epi32(mask_value, nan, t_value);
  return _mm512_cvtusepi32_epi16(t_value);
#endif
}

void cvt_fp32_to_bf16_kernel_impl(
    at::BFloat16* dst,
    const float* src,
    int len) {
  int i = 0;
  for (; i < len - 15; i += 16) {
    auto f32 = _mm512_loadu_ps(src + i);
    _mm256_storeu_si256((__m256i*)(dst + i), _cvt_fp32_to_bf16(f32));
  }
  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto f32 = _mm512_maskz_loadu_ps(mask, src + i);
    _mm256_mask_storeu_epi16(dst + i, mask, _cvt_fp32_to_bf16(f32));
  }
}

#else // DEFAULT ISA implementation.

void cvt_fp32_to_bf16_kernel_impl(
    at::BFloat16* dst,
    const float* src,
    int len) {
  for (int j = 0; j < len; j++) {
    *(dst + j) = *(src + j);
  }
}

#endif

} // anonymous namespace

IPEX_REGISTER_DISPATCH(cvt_fp32_to_bf16_kernel_stub, &cvt_fp32_to_bf16_kernel_impl);

} // namespace cpu
} // namespace torch_ipex

```

### Vec specific kernel example:
This example shows how to get the data type size and its Vec size. In different ISA, Vec has a different register width and a different Vec size.

```c++
//csrc/cpu/aten/GetVecLength.h
#pragma once

#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

std::tuple<int, int> get_cpp_typesize_and_vecsize(at::ScalarType dtype);

namespace {

std::tuple<int, int> get_cpp_typesize_and_vecsize_kernel_impl(
    at::ScalarType dtype);
}

using get_cpp_typesize_and_vecsize_kernel_fn =
    std::tuple<int, int> (*)(at::ScalarType);
IPEX_DECLARE_DISPATCH(
    get_cpp_typesize_and_vecsize_kernel_fn,
    get_cpp_typesize_and_vecsize_kernel_stub);

} // namespace cpu
} // namespace torch_ipex

```

```c++
//csrc/cpu/aten/GetVecLength.cpp

#include "GetVecLength.h"

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(get_cpp_typesize_and_vecsize_kernel_stub);

// get cpp typesize and vectorsize by at::ScalarType
std::tuple<int, int> get_cpp_typesize_and_vecsize(at::ScalarType dtype) {
  return get_cpp_typesize_and_vecsize_kernel_stub(kCPU, dtype);
}

} // namespace cpu
} // namespace torch_ipex

```

```c++
//csrc/cpu/aten/kernels/GetVecLengthKrnl.cpp

#include <ATen/cpu/vec/vec.h>
#include "csrc/cpu/aten/GetVecLength.h"

namespace torch_ipex {
namespace cpu {

namespace {

std::tuple<int, int> get_cpp_typesize_and_vecsize_kernel_impl(
    at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Double:
      return std::make_tuple(
          sizeof(double), at::vec::Vectorized<double>::size());
    case at::ScalarType::Float:
      return std::make_tuple(sizeof(float), at::vec::Vectorized<float>::size());
    case at::ScalarType::ComplexDouble:
      return std::make_tuple(
          sizeof(c10::complex<double>),
          at::vec::Vectorized<c10::complex<double>>::size());
    case at::ScalarType::ComplexFloat:
      return std::make_tuple(
          sizeof(c10::complex<float>),
          at::vec::Vectorized<c10::complex<float>>::size());
    case at::ScalarType::BFloat16:
      return std::make_tuple(
          sizeof(decltype(
              c10::impl::ScalarTypeToCPPType<at::ScalarType::BFloat16>::t)),
          at::vec::Vectorized<decltype(c10::impl::ScalarTypeToCPPType<
                                       at::ScalarType::BFloat16>::t)>::size());
    case at::ScalarType::Half:
      return std::make_tuple(
          sizeof(decltype(
              c10::impl::ScalarTypeToCPPType<at::ScalarType::Half>::t)),
          at::vec::Vectorized<decltype(c10::impl::ScalarTypeToCPPType<
                                       at::ScalarType::Half>::t)>::size());
    default:
      TORCH_CHECK(
          false,
          "Currently only floating and complex ScalarType are supported.");
  }
}

} // anonymous namespace

IPEX_REGISTER_DISPATCH(
    get_cpp_typesize_and_vecsize_kernel_stub,
    &get_cpp_typesize_and_vecsize_kernel_impl);

} // namespace cpu
} // namespace torch_ipex

```
## Private Debug APIs

Here are three ISA-related private APIs that can help debugging::
1. Query current ISA level.
2. Query max CPU supported ISA level.
3. Query max binary supported ISA level.
>**Note:**
>
>1. Max CPU supported ISA level only depends on CPU features.
>2. Max binary supported ISA level only depends on built complier version.
>3. Current ISA level, it is the smaller of `max CPU ISA level` and `max binary ISA level`.

### Example:
```bash
python
Python 3.9.7 (default, Sep 16 2021, 13:09:58)
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import intel_extension_for_pytorch._C as core
>>> core._get_current_isa_level()
'AMX'
>>> core._get_highest_cpu_support_isa_level()
'AMX'
>>> core._get_highest_binary_support_isa_level()
'AMX'
>>> quit()
```

## Select ISA level manually.

By default, IPEX dispatches to the kernels with the maximum ISA level supported by the underlying CPU hardware. This ISA level can be overridden by the environment variable `ATEN_CPU_CAPABILITY` (same environment variable as PyTorch). The available values are {`avx2`, `avx512`, `avx512_vnni`, `avx512_bf16`, `amx`, `avx512_fp16`}. The effective ISA level would be the minimal level between `ATEN_CPU_CAPABILITY` and the maximum level supported by the hardware.
### Example:
```bash
$ python -c 'import intel_extension_for_pytorch._C as core;print(core._get_current_isa_level())'
AMX
$ ATEN_CPU_CAPABILITY=avx2 python -c 'import intel_extension_for_pytorch._C as core;print(core._get_current_isa_level())'
AVX2
```
>**Note:**
>
>`core._get_current_isa_level()` is an IPEX internal function used for checking the current effective ISA level. It is used for debugging purpose only and subject to change.

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
avx512_vpopcntdq:                       true
avx512_4fmaps:                  false
avx512_4vnniw:                  false
avx512_vbmi2:                   true
avx512_vpclmul:                 true
avx512_vnni:                    true
avx512_bitalg:                  true
avx512_fp16:                    true
avx512_bf16:                    true
avx512_vp2intersect:                    true
amx_bf16:                       true
amx_tile:                       true
amx_int8:                       true
prefetchw:                      true
prefetchwt1:                    false
```
