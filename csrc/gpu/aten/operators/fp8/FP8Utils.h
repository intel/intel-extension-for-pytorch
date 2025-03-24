/*******************************************************************************
 * Copyright (C) 2024 Intel Corporation
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission. This software and the related documents are provided as is,
 * with no express or implied warranties, other than those that are expressly
 * stated in the License.
 *******************************************************************************
 */
#ifndef IPEX_CORE_KERNELS_GPU_FP8_UTILS_H_
#define IPEX_CORE_KERNELS_GPU_FP8_UTILS_H_

#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <string>

namespace at {
namespace AtenIpexTypeXPU {
using fp8e5m2 = at::Float8_e5m2;
using fp8e4m3 = at::Float8_e4m3fn;

enum Float8Format {
  NOT_VALID = 0,
  kFloat8_E5M2 = static_cast<int>(at::kFloat8_e5m2),
  kFloat8_E4M3 = static_cast<int>(at::kFloat8_e4m3fn),
};

// Each tensor here is shape (N, ) holding all scaling
// data for a single FP8 block, e.g. LayerNormLinear
class FP8TensorMeta {
 public:
  at::Tensor scale;
  at::Tensor scale_inv;
  at::Tensor amax_history;
};

// Used as named indices on the `scale`, `scale_inv`,
// and `amax` tensors in the `FP8TensorMeta` class.
enum FP8FwdTensors {
  GEMM1_INPUT = 0,
  GEMM1_WEIGHT = 1,
  GEMM1_OUTPUT = 2,
  GEMM2_INPUT = 3,
  GEMM2_WEIGHT = 4,
  GEMM2_OUTPUT = 5
};

// Used as named indices on the `scale`, `scale_inv`,
// and `amax` tensors in the `FP8TensorMeta` class.
enum FP8BwdTensors {
  GRAD_OUTPUT1 = 0,
  GRAD_INPUT1 = 1,
  GRAD_OUTPUT2 = 2,
  GRAD_INPUT2 = 3
};

template <typename T>
struct is_fp8 : std::false_type {};

template <>
struct is_fp8<fp8e4m3> : std::true_type {};

template <>
struct is_fp8<fp8e5m2> : std::true_type {};

#define IPEX_PRIVATE_CASE_TYPE_ESIMD(enum_type, type, ...) \
  case enum_type: {                                        \
    using type_in = type;                                  \
    return __VA_ARGS__();                                  \
  }

#define IPEX_TYPE_SWITCH_ESIMD(TYPE, NAME, ...)                                \
  [&] {                                                                        \
    constexpr const char* at_dispatch_name = NAME;                             \
    switch (TYPE) {                                                            \
      IPEX_PRIVATE_CASE_TYPE_ESIMD(at::ScalarType::Float, float, __VA_ARGS__)  \
      IPEX_PRIVATE_CASE_TYPE_ESIMD(at::ScalarType::Double, float, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE_ESIMD(                                            \
          at::ScalarType::Half, sycl::half, __VA_ARGS__)                       \
      IPEX_PRIVATE_CASE_TYPE_ESIMD(                                            \
          at::ScalarType::BFloat16, sycl::ext::oneapi::bfloat16;, __VA_ARGS__) \
      default:                                                                 \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");        \
    }                                                                          \
  }()

#define IPEX_TYPE_SWITCH_FP8ONLY_ESIMD(dtype, type_out, ...)       \
  switch (dtype) {                                                 \
    case Float8Format::kFloat8_E5M2: {                             \
      using type_out = sycl::ext::intel::experimental::esimd::bf8; \
      { __VA_ARGS__ }                                              \
    } break;                                                       \
    case Float8Format::kFloat8_E4M3: {                             \
      using type_out = sycl::ext::intel::experimental::esimd::hf8; \
      { __VA_ARGS__ }                                              \
    } break;                                                       \
    default:                                                       \
      TORCH_CHECK(false, "invalid dtype!!\n");                     \
      break;                                                       \
  }

#define FP8_CHECK(stat, msg) stat ? Status::OK() : errors::InvalidArgument(msg)

#define IPEX_TYPE_SWITCH_FP8ONLY(dtype, type, ...) \
  switch (dtype) {                                 \
    case Float8Format::kFloat8_E5M2: {             \
      using type = fp8e5m2;                        \
      { __VA_ARGS__ }                              \
    } break;                                       \
    case Float8Format::kFloat8_E4M3: {             \
      using type = fp8e4m3;                        \
      { __VA_ARGS__ }                              \
    } break;                                       \
    default:                                       \
      TORCH_CHECK(false, "invalid type!!\n");      \
      break;                                       \
  }

#define FP8_PTR(PTR, TYPE) reinterpret_cast<TYPE*>(PTR)

}; // namespace AtenIpexTypeXPU
}; // namespace at
#endif
