#pragma once

#include <ATen/Dispatch.h>

// Dispatch atomic-avaliable data types
#define IPEX_DISPATCH_ATOMIC_ALL_TYPES(TYPE, NAME, ...)                           \
  [&] {                                                                           \
    const auto& the_type = TYPE;                                                  \
    /* don't use TYPE again in case it is an expensive or side-effect op  */      \
    at::ScalarType _st = ::detail::scalar_type(the_type);                         \
    switch (_st) {                                                                \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)             \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)             \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::BFloat16, at::BFloat16, __VA_ARGS__)   \
      default:                                                                    \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");            \
    }                                                                             \
  }()

#define IPEX_DISPATCH_ATOMIC_FLOATING_TYPES(TYPE, NAME, ...)                      \
  [&] {                                                                           \
    const auto& the_type = TYPE;                                                  \
    /* don't use TYPE again in case it is an expensive or side-effect op  */      \
    at::ScalarType _st = ::detail::scalar_type(the_type);                         \
    switch (_st) {                                                                \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)             \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::BFloat16, at::BFloat16, __VA_ARGS__)   \
      default:                                                                    \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");            \
    }                                                                             \
  }()

// Redefine DISPATCH macro for debug usage only
// For example, remove useless data type support for multiple kernels to reduce JIT time
#if 0

#define IPEX_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                        \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...)               \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)      \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define IPEX_DISPATCH_FLOATING_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)                   \
  [&] {                                                                                 \
    const auto& the_type = TYPE;                                                        \
    /* don't use TYPE again in case it is an expensive or side-effect op */             \
    at::ScalarType _st = ::detail::scalar_type(the_type);                               \
    switch (_st) {                                                                      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)                 \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)                   \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE,                                                  \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t), __VA_ARGS__)         \
      default:                                                                          \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");                 \
    }                                                                                   \
  }()

#define IPEX_DISPATCH_FLOATING_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)    \
  [&] {                                                                                 \
    const auto& the_type = TYPE;                                                        \
    /* don't use TYPE again in case it is an expensive or side-effect op */             \
    at::ScalarType _st = ::detail::scalar_type(the_type);                               \
    switch (_st) {                                                                      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)                 \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)                   \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE1,                                                 \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t), __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE2,                                                 \
          decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t), __VA_ARGS__)        \
      default:                                                                          \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");                 \
    }                                                                                   \
  }()

#define IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, ...)                                                \
  [&] {                                                                                                          \
    const auto& the_type = TYPE;                                                                                 \
    /* don't use TYPE again in case it is an expensive or side-effect op */                                      \
    at::ScalarType _st = ::detail::scalar_type(the_type);                                                        \
    switch (_st) {                                                                                               \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)                                            \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)                     \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)                       \
      default:                                                                                                   \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");                                           \
    }                                                                                                            \
  }()

#define IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(SCALARTYPE, TYPE, NAME, ...)                               \
  [&] {                                                                                                          \
    const auto& the_type = TYPE;                                                                                 \
    /* don't use TYPE again in case it is an expensive or side-effect op */                                      \
    at::ScalarType _st = ::detail::scalar_type(the_type);                                                        \
    switch (_st) {                                                                                               \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)                                            \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)                     \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)                       \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE, decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t), __VA_ARGS__)     \
      default:                                                                                                   \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");                                           \
    }                                                                                                            \
  }()

#define IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)                 \
  [&] {                                                                                                          \
    const auto& the_type = TYPE;                                                                                 \
    /* don't use TYPE again in case it is an expensive or side-effect op */                                      \
    at::ScalarType _st = ::detail::scalar_type(the_type);                                                        \
    switch (_st) {                                                                                               \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)                                            \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)                     \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)                       \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE1, decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t), __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE2, decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t), __VA_ARGS__)   \
      default:                                                                                                   \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");                                           \
    }                                                                                                            \
  }()

#define IPEX_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...)                        \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)      \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define IPEX_DISPATCH_INTEGRAL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)                                          \
  [&] {                                                                                                        \
    switch (TYPE) {                                                                                            \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)                                         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)                                         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)                                        \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE, decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t), __VA_ARGS__)   \
      default:                                                                                                 \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");                                        \
    }                                                                                                          \
  }()

#define IPEX_DISPATCH_ALL_TYPES(TYPE, NAME, ...)                             \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op  */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)      \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define IPEX_DISPATCH_COMPLEX_TYPES(TYPE, NAME, ...)                                                \
  [&] {                                                                                             \
    const auto& the_type = TYPE;                                                                    \
    /* don't use TYPE again in case it is an expensive or side-effect op */                         \
    at::ScalarType _st = ::detail::scalar_type(the_type);                                           \
    switch (_st) {                                                                                  \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)        \
      default:                                                                                      \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");                              \
    }                                                                                               \
  }()

#define IPEX_DISPATCH_QINT_TYPES(TYPE, NAME, ...)                       \
  [&] {                                                                 \
    const auto& SCALAR_TYPE C10_UNUSED = TYPE;                          \
    switch (TYPE) {                                                     \
      AT_QINT_PRIVATE_CASE_TYPE(                                        \
          at::kQInt8, at::qint8, at::kChar, int8_t, __VA_ARGS__)        \
      AT_QINT_PRIVATE_CASE_TYPE(                                        \
          at::kQUInt8, at::quint8, at::kByte, uint8_t, __VA_ARGS__)     \
      AT_QINT_PRIVATE_CASE_TYPE(                                        \
          at::kQInt32, at::qint32, at::kInt, int, __VA_ARGS__)          \
      default:                                                          \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
    }                                                                   \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX(TYPE, NAME, ...)                                        \
  [&] {                                                                                             \
    const auto& the_type = TYPE;                                                                    \
    /* don't use TYPE again in case it is an expensive or side-effect op*/                          \
    at::ScalarType _st = ::detail::scalar_type(the_type);                                           \
    switch (_st) {                                                                                  \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)                              \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)                               \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)                             \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)                               \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)                               \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)                              \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)                             \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)        \
      default:                                                                                      \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");                              \
    }                                                                                               \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)                                               \
  [&] {                                                                                                        \
    switch (TYPE) {                                                                                            \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)                                         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)                                        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)                                         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)                                        \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE, decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t), __VA_ARGS__)   \
      default:                                                                                                 \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");                                        \
    }                                                                                                          \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(SCALARTYPE, TYPE, NAME, ...)                                   \
  [&] {                                                                                                        \
    switch (TYPE) {                                                                                            \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)                                         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)                                        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)                                         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)                                        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)                     \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)                   \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE, decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t), __VA_ARGS__)   \
      default:                                                                                                 \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");                                        \
    }                                                                                                          \
  }()

#define IPEX_DISPATCH_COMPLEX_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)                                             \
  [&] {                                                                                                          \
    switch (TYPE) {                                                                                              \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)                       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)                     \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE, decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t), __VA_ARGS__)     \
      default:                                                                                                   \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");                                          \
    }                                                                                                            \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)                                  \
  [&] {                                                                                                          \
    switch (TYPE) {                                                                                              \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)                                            \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)                                            \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)                                            \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)                                          \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE1, decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t), __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE2, decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t), __VA_ARGS__)   \
      default:                                                                                                   \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");                                          \
    }                                                                                                            \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)                      \
  [&] {                                                                                                          \
    switch (TYPE) {                                                                                              \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)                                            \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)                                            \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)                                            \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)                       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)                     \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE1, decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t), __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE2, decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t), __VA_ARGS__)   \
      default:                                                                                                   \
        AT_ERROR(#NAME, " not implemented for '", TYPE, "'");                                                    \
    }                                                                                                            \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...)                     \
  [&] {                                                                                                          \
    switch (TYPE) {                                                                                              \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)                                            \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)                                            \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)                                            \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)                                          \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE1, decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t), __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE2, decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t), __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE3, decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE3>::t), __VA_ARGS__)   \
      default:                                                                                                   \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");                                          \
    }                                                                                                            \
  }()

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...)         \
  [&] {                                                                                                          \
    switch (TYPE) {                                                                                              \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)                                            \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)                                            \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)                                            \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)                       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)                     \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE1, decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE1>::t), __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE2, decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE2>::t), __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE3, decltype(c10::impl::ScalarTypeToCPPType<SCALARTYPE3>::t), __VA_ARGS__)   \
      default:                                                                                                   \
        AT_ERROR(#NAME, " not implemented for '", TYPE, "'");                                                    \
    }                                                                                                            \
  }()

#else // Reuse DISPATCH macro in ATen

#define IPEX_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                   \
  AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...)          \
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_FLOATING_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)   \
  AT_DISPATCH_FLOATING_TYPES_AND(SCALARTYPE, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_FLOATING_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)    \
  AT_DISPATCH_FLOATING_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, ...)                       \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(SCALARTYPE, TYPE, NAME, ...)      \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(SCALARTYPE, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)    \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...)                   \
  AT_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_INTEGRAL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)   \
  AT_DISPATCH_INTEGRAL_TYPES_AND(SCALARTYPE, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES(TYPE, NAME, ...)                        \
  AT_DISPATCH_ALL_TYPES(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_COMPLEX_TYPES(TYPE, NAME, ...)                    \
  AT_DISPATCH_COMPLEX_TYPES(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_QINT_TYPES(TYPE, NAME, ...)                       \
  AT_DISPATCH_QINT_TYPES(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX(TYPE, NAME, ...)            \
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)        \
  AT_DISPATCH_ALL_TYPES_AND(SCALARTYPE, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(SCALARTYPE, TYPE, NAME, ...)    \
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(SCALARTYPE, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_COMPLEX_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)    \
  AT_DISPATCH_COMPLEX_TYPES_AND(SCALARTYPE, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)   \
  AT_DISPATCH_ALL_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)     \
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...)    \
  AT_DISPATCH_ALL_TYPES_AND3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, __VA_ARGS__)

#define IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...)  \
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, __VA_ARGS__)

#endif
