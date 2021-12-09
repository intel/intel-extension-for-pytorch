#pragma once

#ifdef BUILD_INTERNAL_DEBUG
#define IPEX_PRIVATE_CASE_TYPE(enum_type, type, ...) \
  case enum_type: {                                  \
    using scalar_t = type;                           \
    return __VA_ARGS__();                            \
  }

#define IPEX_QINT_PRIVATE_CASE_TYPE(                          \
    enum_type, type, underlying_enum, underlying_type, ...)   \
  case enum_type: {                                           \
    const auto& UNDERLYING_TYPE C10_UNUSED = underlying_enum; \
    using scalar_t C10_UNUSED = type;                         \
    using underlying_t C10_UNUSED = underlying_type;          \
    return __VA_ARGS__();                                     \
  }

#else // NOT BUILD_INTERNAL_DEBUG

#define IPEX_PRIVATE_CASE_TYPE(enum_type, type, ...) \
  IPEX_AT_PRIVATE_CASE_TYPE(enum_type, type, __VA_ARGS__)

#endif // BUILD_INTERNAL_DEBUG

#if defined(BUILD_INTERNAL_DEBUG) && !defined(BUILD_DOUBLE_KERNEL)
#define IPEX_PRIVATE_CASE_DOUBLE(enum_type, type, ...)
#else // Always BUILD_DOUBLE_KERNEL
#define IPEX_PRIVATE_CASE_DOUBLE(enum_type, type, ...) \
  IPEX_PRIVATE_CASE_TYPE(enum_type, type, __VA_ARGS__)
#endif

#define IPEX_DISPATCH_ATOMIC_ALL_TYPES(TYPE, NAME, ...)                      \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op  */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(                                                \
          at::ScalarType::BFloat16, at::BFloat16, __VA_ARGS__)               \
      IPEX_PRIVATE_CASE_DOUBLE(at::ScalarType::Double, double, __VA_ARGS__)  \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, long, __VA_ARGS__)        \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define IPEX_DISPATCH_ATOMIC_FLOATING_TYPES(TYPE, NAME, ...)                 \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op  */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(                                                \
          at::ScalarType::BFloat16, at::BFloat16, __VA_ARGS__)               \
      IPEX_PRIVATE_CASE_DOUBLE(at::ScalarType::Double, double, __VA_ARGS__)  \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define IPEX_DISPATCH_BYTE_ALL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)        \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op  */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::kQInt8, int8_t, __VA_ARGS__)                \
      IPEX_PRIVATE_CASE_TYPE(at::kQUInt8, uint8_t, __VA_ARGS__)              \
      IPEX_PRIVATE_CASE_TYPE(                                                \
          SCALARTYPE,                                                        \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),           \
          __VA_ARGS__)                                                       \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

// Some kernels must support full data types, for example, fill and copy.
// This dispatch macro helps to keep all data types support in any build config.
#define IPEX_DISPATCH_ALL_TYPES_ALWAYS_AND(SCALARTYPE, TYPE, NAME, ...) \
  IPEX_AT_DISPATCH_ALL_TYPES_AND(SCALARTYPE, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_ALWAYS_AND2(   \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...) \
  IPEX_AT_DISPATCH_ALL_TYPES_AND2(             \
      SCALARTYPE1, SCALARTYPE2, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_ALWAYS_AND3(                \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  IPEX_AT_DISPATCH_ALL_TYPES_AND3(                          \
      SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, __VA_ARGS__)

// Redefine DISPATCH macro for debug usage only
// For example, remove useless data type support for multiple kernels to reduce
// JIT time

#ifdef BUILD_INTERNAL_DEBUG

#define IPEX_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                       \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      IPEX_PRIVATE_CASE_DOUBLE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...)              \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      IPEX_PRIVATE_CASE_DOUBLE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)   \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define IPEX_DISPATCH_FLOATING_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)       \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      IPEX_PRIVATE_CASE_DOUBLE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),          \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");     \
    }                                                                       \
  }()

#define IPEX_DISPATCH_FLOATING_TYPES_AND2(                                  \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)                              \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      IPEX_PRIVATE_CASE_DOUBLE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE1,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),         \
          __VA_ARGS__)                                                      \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE2,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),         \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");     \
    }                                                                       \
  }()

#define IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, ...)           \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      IPEX_PRIVATE_CASE_DOUBLE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_DOUBLE(                                             \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)   \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(                      \
    SCALARTYPE, TYPE, NAME, ...)                                            \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      IPEX_PRIVATE_CASE_DOUBLE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_DOUBLE(                                             \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),          \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                      \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)                              \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      IPEX_PRIVATE_CASE_DOUBLE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_DOUBLE(                                             \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE1,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),         \
          __VA_ARGS__)                                                      \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE2,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),         \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define IPEX_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...)                       \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)   \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define IPEX_DISPATCH_INTEGRAL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)     \
  [&] {                                                                   \
    switch (TYPE) {                                                       \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)  \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE,                                                     \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),        \
          __VA_ARGS__)                                                    \
      default:                                                            \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");   \
    }                                                                     \
  }()

#define IPEX_DISPATCH_ALL_TYPES(TYPE, NAME, ...)                             \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op  */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_DOUBLE(at::ScalarType::Double, double, __VA_ARGS__)  \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)    \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define IPEX_DISPATCH_COMPLEX_TYPES(TYPE, NAME, ...)                        \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_DOUBLE(                                             \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__) \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define IPEX_DISPATCH_QINT_TYPES(TYPE, NAME, ...)                       \
  [&] {                                                                 \
    const auto& SCALAR_TYPE C10_UNUSED = TYPE;                          \
    switch (TYPE) {                                                     \
      IPEX_QINT_PRIVATE_CASE_TYPE(                                      \
          at::kQInt8, at::qint8, at::kChar, int8_t, __VA_ARGS__)        \
      IPEX_QINT_PRIVATE_CASE_TYPE(                                      \
          at::kQUInt8, at::quint8, at::kByte, uint8_t, __VA_ARGS__)     \
      IPEX_QINT_PRIVATE_CASE_TYPE(                                      \
          at::kQInt32, at::qint32, at::kInt, int, __VA_ARGS__)          \
      default:                                                          \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
    }                                                                   \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX(TYPE, NAME, ...)                \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op*/  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_DOUBLE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_DOUBLE(                                             \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__) \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)            \
  [&] {                                                                     \
    switch (TYPE) {                                                         \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_DOUBLE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),          \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");     \
    }                                                                       \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(SCALARTYPE, TYPE, NAME, ...) \
  [&] {                                                                      \
    switch (TYPE) {                                                          \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_DOUBLE(at::ScalarType::Double, double, __VA_ARGS__)  \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)      \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(                                                \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_DOUBLE(                                              \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)  \
      IPEX_PRIVATE_CASE_TYPE(                                                \
          SCALARTYPE,                                                        \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),           \
          __VA_ARGS__)                                                       \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");      \
    }                                                                        \
  }()

#define IPEX_DISPATCH_COMPLEX_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)        \
  [&] {                                                                     \
    switch (TYPE) {                                                         \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_DOUBLE(                                             \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),          \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");     \
    }                                                                       \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND2(                                       \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)                              \
  [&] {                                                                     \
    switch (TYPE) {                                                         \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_DOUBLE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE1,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),         \
          __VA_ARGS__)                                                      \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE2,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),         \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");     \
    }                                                                       \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(                           \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)                              \
  [&] {                                                                     \
    switch (TYPE) {                                                         \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_DOUBLE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_DOUBLE(                                             \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE1,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),         \
          __VA_ARGS__)                                                      \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE2,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),         \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", TYPE, "'");               \
    }                                                                       \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND3(                                       \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...)                 \
  [&] {                                                                     \
    switch (TYPE) {                                                         \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_DOUBLE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE1,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),         \
          __VA_ARGS__)                                                      \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE2,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),         \
          __VA_ARGS__)                                                      \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE3,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE3>::t),         \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");     \
    }                                                                       \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(                           \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...)                 \
  [&] {                                                                     \
    switch (TYPE) {                                                         \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_DOUBLE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)     \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)    \
      IPEX_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)   \
      IPEX_PRIVATE_CASE_DOUBLE(                                             \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__) \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE1,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),         \
          __VA_ARGS__)                                                      \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE2,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),         \
          __VA_ARGS__)                                                      \
      IPEX_PRIVATE_CASE_TYPE(                                               \
          SCALARTYPE3,                                                      \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE3>::t),         \
          __VA_ARGS__)                                                      \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", TYPE, "'");               \
    }                                                                       \
  }()

#else // Reuse DISPATCH macro in ATen

#define IPEX_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  IPEX_AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...) \
  IPEX_AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_FLOATING_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  IPEX_AT_DISPATCH_FLOATING_TYPES_AND(SCALARTYPE, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_FLOATING_TYPES_AND2(     \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...) \
  IPEX_AT_DISPATCH_FLOATING_TYPES_AND2(        \
      SCALARTYPE1, SCALARTYPE2, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, ...) \
  IPEX_AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1( \
    SCALARTYPE, TYPE, NAME, ...)                       \
  IPEX_AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(    \
      SCALARTYPE, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2( \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)         \
  IPEX_AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(    \
      SCALARTYPE1, SCALARTYPE2, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  IPEX_AT_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_INTEGRAL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  IPEX_AT_DISPATCH_INTEGRAL_TYPES_AND(SCALARTYPE, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES(TYPE, NAME, ...) \
  IPEX_AT_DISPATCH_ALL_TYPES(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_COMPLEX_TYPES(TYPE, NAME, ...) \
  IPEX_AT_DISPATCH_COMPLEX_TYPES(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_QINT_TYPES(TYPE, NAME, ...) \
  IPEX_AT_DISPATCH_QINT_TYPES(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX(TYPE, NAME, ...) \
  IPEX_AT_DISPATCH_ALL_TYPES_AND_COMPLEX(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  IPEX_AT_DISPATCH_ALL_TYPES_AND(SCALARTYPE, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(SCALARTYPE, TYPE, NAME, ...) \
  IPEX_AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(                                \
      SCALARTYPE, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_COMPLEX_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  IPEX_AT_DISPATCH_COMPLEX_TYPES_AND(SCALARTYPE, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND2(          \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...) \
  IPEX_AT_DISPATCH_ALL_TYPES_AND2(             \
      SCALARTYPE1, SCALARTYPE2, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2( \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)    \
  IPEX_AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(    \
      SCALARTYPE1, SCALARTYPE2, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND3(                       \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  IPEX_AT_DISPATCH_ALL_TYPES_AND3(                          \
      SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(           \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  IPEX_AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(              \
      SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_INDEX_TYPES(TYPE, NAME, ...) \
  IPEX_AT_DISPATCH_INDEX_TYPES(TYPE, NAME, __VA_ARGS__)

#endif
#pragma once

#include <ATen/core/DeprecatedTypeProperties.h>
#include <ATen/core/Tensor.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#define IPEX_AT_PRIVATE_CASE_TYPE(enum_type, type, ...) \
  case enum_type: {                                     \
    using scalar_t = type;                              \
    return __VA_ARGS__();                               \
  }

// Workaround for C10_UNUSED because CUDA 10.1 and below fails to handle unused
// attribute in the type aliasing context. Keep name long and verbose to avoid
// macro collisions.
#if defined(__CUDACC__) && CUDA_VERSION <= 10100
#define C10_UNUSED_DISPATCH_CUDA_WORKAROUND
#else
#define C10_UNUSED_DISPATCH_CUDA_WORKAROUND C10_UNUSED
#endif // defined(__CUDACC__) && CUDA_VERSION <= 10100

#define IPEX_AT_QINT_PRIVATE_CASE_TYPE(                                      \
    enum_type, type, underlying_enum, underlying_type, ...)                  \
  case enum_type: {                                                          \
    using scalar_t = type;                                                   \
    using underlying_t C10_UNUSED_DISPATCH_CUDA_WORKAROUND =                 \
        scalar_t::underlying;                                                \
    const auto& SCALAR_TYPE C10_UNUSED_DISPATCH_CUDA_WORKAROUND = enum_type; \
    const auto& UNDERLYING_TYPE C10_UNUSED_DISPATCH_CUDA_WORKAROUND =        \
        toUnderlying(enum_type);                                             \
    return __VA_ARGS__();                                                    \
  }

// This macro should be used to skip bfloat16 dispatch on non-ROCm platforms and
// should be removed once the bfloat16 bringup is complete on other platforms.
// This is supposed to be used as a wrapper around the lambda function passed to
// the dispatch macro and will conditionally dispatch ops with bfloat16 type
// only on ROCm.
#if !defined(__HIP_PLATFORM_HCC__)
#define AT_SKIP_BFLOAT16_IF_NOT_ROCM(SCALARTYPE, NAME, ...) \
  if (std::is_same<SCALARTYPE, at::BFloat16>::value) {      \
    AT_ERROR(                                               \
        #NAME,                                              \
        " not implemented for '",                           \
        toString(at::ScalarType::BFloat16),                 \
        "'");                                               \
  } else {                                                  \
    return __VA_ARGS__();                                   \
  }
#else
#define AT_SKIP_BFLOAT16_IF_NOT_ROCM(SCALARTYPE, NAME, ...) return __VA_ARGS__()
#endif

namespace detail {

inline at::ScalarType ipex_scalar_type(at::ScalarType s) {
  return s;
}

C10_DEPRECATED_MESSAGE(
    "passing at::DeprecatedTypeProperties to an IPEX_AT_DISPATCH macro is deprecated, "
    "pass an at::ScalarType instead")
inline at::ScalarType ipex_scalar_type(const at::DeprecatedTypeProperties& t) {
  return t.scalarType();
}

C10_DEPRECATED_MESSAGE(
    "IPEX_AT_DISPATCH_ALL_TYPES_AND_HALF is deprecated, "
    "use IPEX_AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, ...) instead")
inline void deprecated_IPEX_AT_DISPATCH_ALL_TYPES_AND_HALF() {}

C10_DEPRECATED_MESSAGE(
    "IPEX_AT_DISPATCH_ALL_TYPES_AND_HALF_AND_COMPLEX is deprecated, "
    "use IPEX_AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(at::ScalarType::Half, ...) "
    "instead")
inline void deprecated_IPEX_AT_DISPATCH_ALL_TYPES_AND_HALF_AND_COMPLEX() {}

} // namespace detail

// The IPEX_AT_DISPATCH_* family of macros provides the ability to
// conveniently generate specializations of a kernel over all of the
// dtypes we care about in PyTorch.  We call it "dispatch" because
// we are "dispatching" to the correct, dtype-specific kernel.
//
// A standard usage looks like:
//
//      IPEX_AT_DISPATCH_ALL_TYPES(self.ipex_scalar_type(), "op_name", [&] {
//          // Your code here, with 'scalar_t' now defined to
//          // be the dtype in question
//      })
//
// There are many variations of this macro, so it's important to
// understand exactly /which/ dtypes you want to get instantiated, as
// well as what the "default" set is.
//
// The default set of dtypes that are instantiated (e.g., by
// IPEX_AT_DISPATCH_ALL_TYPES) are floating point types (float, double),
// and integral types (int32_t, int64_t, int16_t, int8_t, uint8_t),
// but NOT booleans (bool), half-precision floats (Half) or
// complex number (c10::complex<float>, c10::complex<double>).
// This "cut" is somewhat historical (the default types are the
// ones that TH historically supported), but it also reflects the
// fact that the non-default types are "poorly" behaved (booleans
// are NOT integers mod 2, half precision operations ~essentially
// don't exist on CPU, complex numbers are an experimental application).
//
// Here are the questions you should generally ask to decide which
// dispatch you want:
//
// 1. Is this an integral or floating point specific operation?
//    (If so, you'll want one of the FLOATING or INTEGRAL macros.)
//
// 2. Should half be supported?  (If you're on CPU, the answer is almost
//    definitely no.  If you do want support, use one of the AND_HALF
//    macros)
//
// Much rarer situations:
//
// 3. Should bool be supported?  (You often have to write your kernel
//    differently if arithmetic operations are involved.)  If so,
//    Use IPEX_AT_DISPATCH_ALL_TYPES_AND along with ScalarType::Bool
//
// 4. Should complex be supported?  The answer is almost always no,
//    unless you are working on "generic" code that should work on
//    all dtypes.

// NB: the the_type variable is not used, but we have kept it for
// backwards compatibility.  It's probably not used by anyone though;
// but we're just being safe (and it doesn't hurt.)  Note we must
// use it to shut up warnings about unused store.

#define IPEX_AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                     \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::ipex_scalar_type(the_type);               \
    switch (_st) {                                                           \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)   \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define IPEX_AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...)            \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::ipex_scalar_type(the_type);               \
    switch (_st) {                                                           \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__) \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define IPEX_AT_DISPATCH_FLOATING_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)     \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::ipex_scalar_type(the_type);               \
    switch (_st) {                                                           \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE,                                                        \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),           \
          __VA_ARGS__)                                                       \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");      \
    }                                                                        \
  }()

#define IPEX_AT_DISPATCH_FLOATING_TYPES_AND2(                                \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)                               \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::ipex_scalar_type(the_type);               \
    switch (_st) {                                                           \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE1,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),          \
          __VA_ARGS__)                                                       \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE2,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),          \
          __VA_ARGS__)                                                       \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");      \
    }                                                                        \
  }()

#define IPEX_AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, ...)         \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::ipex_scalar_type(the_type);               \
    switch (_st) {                                                           \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          at::ScalarType::ComplexDouble, c10::complex<double>, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          at::ScalarType::ComplexFloat, c10::complex<float>, __VA_ARGS__)    \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define IPEX_AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(                    \
    SCALARTYPE, TYPE, NAME, ...)                                             \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::ipex_scalar_type(the_type);               \
    switch (_st) {                                                           \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          at::ScalarType::ComplexDouble, c10::complex<double>, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          at::ScalarType::ComplexFloat, c10::complex<float>, __VA_ARGS__)    \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE,                                                        \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),           \
          __VA_ARGS__)                                                       \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define IPEX_AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                    \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)                               \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::ipex_scalar_type(the_type);               \
    switch (_st) {                                                           \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          at::ScalarType::ComplexDouble, c10::complex<double>, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          at::ScalarType::ComplexFloat, c10::complex<float>, __VA_ARGS__)    \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE1,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),          \
          __VA_ARGS__)                                                       \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE2,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),          \
          __VA_ARGS__)                                                       \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define IPEX_AT_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...)                     \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::ipex_scalar_type(the_type);               \
    switch (_st) {                                                           \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define IPEX_AT_DISPATCH_INTEGRAL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)     \
  [&] {                                                                      \
    switch (TYPE) {                                                          \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE,                                                        \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),           \
          __VA_ARGS__)                                                       \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");      \
    }                                                                        \
  }()

#define IPEX_AT_DISPATCH_ALL_TYPES(TYPE, NAME, ...)                          \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op  */ \
    at::ScalarType _st = ::detail::ipex_scalar_type(the_type);               \
    switch (_st) {                                                           \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define IPEX_AT_DISPATCH_COMPLEX_TYPES(TYPE, NAME, ...)                     \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::ipex_scalar_type(the_type);              \
    switch (_st) {                                                          \
      IPEX_AT_PRIVATE_CASE_TYPE(                                            \
          at::ScalarType::ComplexFloat, c10::complex<float>, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(                                            \
          at::ScalarType::ComplexDouble, c10::complex<double>, __VA_ARGS__) \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

#define IPEX_AT_DISPATCH_QINT_TYPES(TYPE, NAME, ...)                        \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::ipex_scalar_type(the_type);              \
    switch (_st) {                                                          \
      IPEX_AT_QINT_PRIVATE_CASE_TYPE(                                       \
          at::kQInt8, at::qint8, at::kChar, int8_t, __VA_ARGS__)            \
      IPEX_AT_QINT_PRIVATE_CASE_TYPE(                                       \
          at::kQUInt8, at::quint8, at::kByte, uint8_t, __VA_ARGS__)         \
      IPEX_AT_QINT_PRIVATE_CASE_TYPE(                                       \
          at::kQInt32, at::qint32, at::kInt, int, __VA_ARGS__)              \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");     \
    }                                                                       \
  }()

#define IPEX_AT_DISPATCH_ALL_TYPES_AND_COMPLEX(TYPE, NAME, ...)              \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op*/   \
    at::ScalarType _st = ::detail::ipex_scalar_type(the_type);               \
    switch (_st) {                                                           \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          at::ScalarType::ComplexFloat, c10::complex<float>, __VA_ARGS__)    \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          at::ScalarType::ComplexDouble, c10::complex<double>, __VA_ARGS__)  \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define IPEX_AT_DISPATCH_ALL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)          \
  [&] {                                                                      \
    switch (TYPE) {                                                          \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE,                                                        \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),           \
          __VA_ARGS__)                                                       \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");      \
    }                                                                        \
  }()

#define IPEX_AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(                          \
    SCALARTYPE, TYPE, NAME, ...)                                             \
  [&] {                                                                      \
    switch (TYPE) {                                                          \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          at::ScalarType::ComplexFloat, c10::complex<float>, __VA_ARGS__)    \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          at::ScalarType::ComplexDouble, c10::complex<double>, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE,                                                        \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t),           \
          __VA_ARGS__)                                                       \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");      \
    }                                                                        \
  }()

#define IPEX_AT_DISPATCH_ALL_TYPES_AND2(                                     \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)                               \
  [&] {                                                                      \
    switch (TYPE) {                                                          \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE1,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),          \
          __VA_ARGS__)                                                       \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE2,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),          \
          __VA_ARGS__)                                                       \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");      \
    }                                                                        \
  }()

#define IPEX_AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(                         \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)                               \
  [&] {                                                                      \
    switch (TYPE) {                                                          \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          at::ScalarType::ComplexFloat, c10::complex<float>, __VA_ARGS__)    \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          at::ScalarType::ComplexDouble, c10::complex<double>, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE1,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),          \
          __VA_ARGS__)                                                       \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE2,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),          \
          __VA_ARGS__)                                                       \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", TYPE, "'");                \
    }                                                                        \
  }()

#define IPEX_AT_DISPATCH_ALL_TYPES_AND3(                                     \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...)                  \
  [&] {                                                                      \
    switch (TYPE) {                                                          \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE1,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),          \
          __VA_ARGS__)                                                       \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE2,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),          \
          __VA_ARGS__)                                                       \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE3,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE3>::t),          \
          __VA_ARGS__)                                                       \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");      \
    }                                                                        \
  }()

#define IPEX_AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(                         \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...)                  \
  [&] {                                                                      \
    switch (TYPE) {                                                          \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          at::ScalarType::ComplexFloat, c10::complex<float>, __VA_ARGS__)    \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          at::ScalarType::ComplexDouble, c10::complex<double>, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE1,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t),          \
          __VA_ARGS__)                                                       \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE2,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t),          \
          __VA_ARGS__)                                                       \
      IPEX_AT_PRIVATE_CASE_TYPE(                                             \
          SCALARTYPE3,                                                       \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE3>::t),          \
          __VA_ARGS__)                                                       \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", TYPE, "'");                \
    }                                                                        \
  }()

// ----------------------------------------------------------------------------
// DEPRECATED MACROS, DON'T USE THESE
// ----------------------------------------------------------------------------

#define IPEX_AT_DISPATCH_ALL_TYPES_AND_HALF(TYPE, NAME, ...)                 \
  [&] {                                                                      \
    detail::deprecated_IPEX_AT_DISPATCH_ALL_TYPES_AND_HALF();                \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::ipex_scalar_type(the_type);               \
    switch (_st) {                                                           \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      IPEX_AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__) \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define IPEX_AT_DISPATCH_INDEX_TYPES(TYPE, NAME, ...)                       \
  [&] {                                                                     \
    const auto& the_index_type = TYPE;                                      \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _it = ::detail::scalar_type(the_index_type);             \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _it)                                 \
    switch (_it) {                                                          \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                                      \
          NAME, at::ScalarType::Int, int32_t, index_t, __VA_ARGS__)         \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                                      \
          NAME, at::ScalarType::Long, int64_t, index_t, __VA_ARGS__)        \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_it), "'");      \
    }                                                                       \
  }()
