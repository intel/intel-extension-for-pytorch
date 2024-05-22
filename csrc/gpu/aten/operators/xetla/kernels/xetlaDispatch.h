#pragma once
#include "../xetla_kernel_api.h"
#include "xetla.h"

#define XETLA_KERNEL_TYPE(enum_type, type, ...) \
  case enum_type: {                             \
    using scalar_t = type;                      \
    return __VA_ARGS__();                       \
  }

#define XETLA_ALL_KERNEL_TYPES(TYPE, ...)                               \
  [&] {                                                                 \
    const auto& the_type = TYPE;                                        \
    switch (the_type) {                                                 \
      XETLA_KERNEL_TYPE(XetlaType::fp16, gpu::xetla::fp16, __VA_ARGS__) \
      XETLA_KERNEL_TYPE(XetlaType::bf16, gpu::xetla::bf16, __VA_ARGS__) \
      default: {                                                        \
        std::cerr << "hgemm kernel not implemented for XetlaType '"     \
                  << int(the_type) << "'" << std::endl;                 \
        exit(1);                                                        \
      }                                                                 \
    }                                                                   \
  }()
