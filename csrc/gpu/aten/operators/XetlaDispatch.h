#pragma once
#include <ATen/Dispatch.h>
#include "xetla/xetla_kernel_api.h"

#define XETLA_CASE_TYPE(enum_type, type, xe_type, ...) \
  case enum_type: {                                    \
    using scalar_t = type;                             \
    constexpr XetlaType xetla_t = xe_type;             \
    return __VA_ARGS__();                              \
  }

#define XETLA_ALL_TYPES(TYPE, NAME, ...)                                     \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op  */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    constexpr const char* at_dispatch_name = NAME;                           \
    switch (_st) {                                                           \
      XETLA_CASE_TYPE(                                                       \
          at::ScalarType::Half,                                              \
          at::Half,                                                          \
          xpu::xetla::XetlaType::fp16,                                       \
          __VA_ARGS__)                                                       \
      XETLA_CASE_TYPE(                                                       \
          at::ScalarType::BFloat16,                                          \
          at::BFloat16,                                                      \
          xpu::xetla::XetlaType::bf16,                                       \
          __VA_ARGS__)                                                       \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()
