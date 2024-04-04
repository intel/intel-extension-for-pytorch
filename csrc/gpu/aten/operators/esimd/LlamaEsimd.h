#pragma once

#define DPCT_COMPAT_RT_VERSION 12010
#include <assert.h>
#include <dpct/blas_utils.hpp>
#include <dpct/dpct.hpp>
#include <dpct/lib_common_utils.hpp>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace sycl;
using namespace std;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;

#ifdef EXPORT_KERNEL
extern "C" bool __declspec(dllexport) runLinear(
    queue& q,
    unsigned m,
    unsigned n,
    unsigned k,
    uint8_t* a,
    uint8_t* b,
    uint8_t* c);
extern "C" bool __declspec(dllexport) runLinearAdjustable(
    queue& q,
    unsigned m,
    unsigned n,
    unsigned k,
    uint32_t pixelPerGroupCommonDim4096,
    uint32_t pixelPerGroupCommonDim11008,
    uint32_t* dispatchPattern /*0: Nx8, 1: Nx16, 2: Nx4, 3: Nx2*/,
    uint8_t* a,
    uint8_t* b,
    uint8_t* c);
extern "C" bool __declspec(dllexport) runLinearAdjustableQ6k(
    queue& q,
    unsigned m,
    unsigned n,
    unsigned k,
    uint32_t* pixelPerGroup,
    uint32_t* dispatchPattern /*0: Nx8, 1: Nx16, 2: Nx4, 3: Nx2*/,
    uint8_t* a,
    uint8_t* b,
    uint8_t* c);
extern "C" bool __declspec(dllexport) runGemm(
    queue& q,
    unsigned m,
    unsigned n,
    unsigned k,
    unsigned internalPrecision /*0: fp32, 1: fp16*/,
    uint8_t* a,
    uint8_t* b,
    uint8_t* c);
extern "C" bool __declspec(dllexport) runGemmPartition(
    queue& q,
    unsigned
        m /* !! m must be the original input row dimension before partition!! */
    ,
    unsigned n,
    unsigned k,
    unsigned startRow,
    unsigned rowsToProcess,
    unsigned internalPrecision /*0: fp32, 1: fp16*/,
    uint8_t* a,
    uint8_t* b,
    uint8_t* c);
extern "C" bool __declspec(dllexport) runGemmDequantQ40(
    queue& q,
    unsigned m,
    unsigned n,
    unsigned k,
    unsigned internalPrecision /*0: fp32, 1: fp16*/,
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* shuffleTt);
extern "C" bool __declspec(dllexport) runInt4Dequant0(
    queue& q,
    unsigned size,
    unsigned outputPrecision /*0: fp32, 1: fp16*/,
    uint8_t* input,
    uint8_t* output);
extern "C" bool __declspec(dllexport) runDequantQ6k(
    queue& q,
    unsigned size,
    unsigned outputPrecision /*0: fp32, 1: fp16*/,
    uint8_t* input,
    uint8_t* output);
extern "C" bool __declspec(dllexport) runFfnFusion(
    queue& q,
    unsigned m,
    uint8_t* gate,
    uint8_t* up,
    uint8_t* curr,
    uint8_t* out);
extern "C" bool __declspec(dllexport) runSdpFusion(
    queue& q,
    uint8_t* qState,
    uint8_t* kState,
    uint8_t* vState,
    uint8_t* qkvOut,
    unsigned batch_size,
    unsigned kv_len,
    unsigned vCacheStride,
    unsigned precision /*0: fp32, 1: fp16*/);
#else
extern "C" bool __declspec(dllimport) runLinear(
    queue& q,
    unsigned m,
    unsigned n,
    unsigned k,
    uint8_t* a,
    uint8_t* b,
    uint8_t* c);
extern "C" bool __declspec(dllimport) runLinearAdjustable(
    queue& q,
    unsigned m,
    unsigned n,
    unsigned k,
    uint32_t pixelPerGroupCommonDim4096,
    uint32_t pixelPerGroupCommonDim11008,
    uint32_t* dispatchPattern /*0: Nx8, 1: Nx16, 2: Nx4, 3: Nx2*/,
    uint8_t* a,
    uint8_t* b,
    uint8_t* c);
extern "C" bool __declspec(dllimport) runLinearAdjustableQ6k(
    queue& q,
    unsigned m,
    unsigned n,
    unsigned k,
    uint32_t* pixelPerGroup,
    uint32_t* dispatchPattern /*0: Nx8, 1: Nx16, 2: Nx4, 3: Nx2*/,
    uint8_t* a,
    uint8_t* b,
    uint8_t* c);
extern "C" bool __declspec(dllimport) runGemm(
    queue& q,
    unsigned m,
    unsigned n,
    unsigned k,
    unsigned internalPrecision /*0: fp32, 1: fp16*/,
    uint8_t* a,
    uint8_t* b,
    uint8_t* c);
extern "C" bool __declspec(dllimport) runGemmPartition(
    queue& q,
    unsigned
        m /* !! m must be the original input row dimension before partition!! */
    ,
    unsigned n,
    unsigned k,
    unsigned startRow,
    unsigned rowsToProcess,
    unsigned internalPrecision /*0: fp32, 1: fp16*/,
    uint8_t* a,
    uint8_t* b,
    uint8_t* c);
extern "C" bool __declspec(dllimport) runGemmDequantQ40(
    queue& q,
    unsigned m,
    unsigned n,
    unsigned k,
    unsigned internalPrecision /*0: fp32, 1: fp16*/,
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* shuffleTt);
extern "C" bool __declspec(dllimport) runInt4Dequant0(
    queue& q,
    unsigned size,
    unsigned outputPrecision /*0: fp32, 1: fp16*/,
    uint8_t* input,
    uint8_t* output);
extern "C" bool __declspec(dllimport) runDequantQ6k(
    queue& q,
    unsigned size,
    unsigned outputPrecision /*0: fp32, 1: fp16*/,
    uint8_t* input,
    uint8_t* output);
extern "C" bool __declspec(dllimport) runFfnFusion(
    queue& q,
    unsigned m,
    uint8_t* gate,
    uint8_t* up,
    uint8_t* curr,
    uint8_t* out);
extern "C" bool __declspec(dllimport) runSdpFusion(
    queue& q,
    uint8_t* qState,
    uint8_t* kState,
    uint8_t* vState,
    uint8_t* qkvOut,
    unsigned batch_size,
    unsigned kv_len,
    unsigned vCacheStride,
    unsigned precision /*0: fp32, 1: fp16*/);
#endif