#pragma once

#include <sycl/sycl.hpp>

#include <c10/util/Float8_e5m2.h>
#include <torch/all.h>

#define AT_DISPATCH_FP8_CASE(enum_type, ...) \
  AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, fp8_t, __VA_ARGS__)

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#define VLLM_DISPATCH_CASE_FP8_TYPES(...) \
  AT_DISPATCH_FP8_CASE(at::ScalarType::Float8_e5m2, __VA_ARGS__)

#define VLLM_DISPATCH_FP8_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FP8_TYPES(__VA_ARGS__))

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
struct __align__(8) vec4_t {
  scalar_t x;
  scalar_t y;
  scalar_t z;
  scalar_t w;
};

template <typename quant_type_t>
struct __align__(4) q8x4_t {
  static_assert(std::is_same_v<quant_type_t, c10::Float8_e5m2>);
  quant_type_t x;
  quant_type_t y;
  quant_type_t z;
  quant_type_t w;
};

} // namespace AtenIpexTypeXPU
} // namespace at

