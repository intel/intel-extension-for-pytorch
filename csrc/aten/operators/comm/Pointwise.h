#pragma once

#include <ATen/ATen.h>

#include <core/TensorImplUtils.h>
#include <utils/DPCPP.h>
#include "ATDispatch.h"
#include "ApplyUtils.h"
#include "Numerics.h"

namespace at {
namespace AtenIpexTypeXPU {

const Tensor& resize_as_(
    const Tensor& self,
    const Tensor& the_template,
    c10::optional<MemoryFormat> memory_format);

#define IMPLEMENT_POINTWISE_FUNC_(NAME, CFUNC, REAL)             \
  template <typename scalar_t>                                   \
  struct Tensor_##NAME##_##REAL##_Op {                           \
    inline void operator()(scalar_t& out, scalar_t& in) const {  \
      out = CFUNC(in);                                           \
    }                                                            \
                                                                 \
    inline void operator()(scalar_t& v) const {                  \
      v = CFUNC(v);                                              \
    }                                                            \
  };                                                             \
                                                                 \
  template <typename scalar_t>                                   \
  void NAME(Tensor& self_, const Tensor& src) {                  \
    if (TensorImpl_Unwrap(self_) == TensorImpl_Unwrap(src)) {    \
      DPCPP_tensor_apply1<scalar_t>(                             \
          self_, Tensor_##NAME##_##REAL##_Op<scalar_t>());       \
    } else {                                                     \
      at::AtenIpexTypeXPU::resize_as_(self_, src, c10::nullopt); \
      DPCPP_tensor_apply2<scalar_t, scalar_t>(                   \
          self_, src, Tensor_##NAME##_##REAL##_Op<scalar_t>());  \
    }                                                            \
  }

#define IMPLEMENT_POINTWISE_1_FUNC(NAME, CFUNC, REAL) \
  IMPLEMENT_POINTWISE_FUNC_(NAME, CFUNC, REAL)

// Customized Callable Ops
#define POINTWISE_ARGS_1 arg1 // out
#define POINTWISE_ARGS_2 POINTWISE_ARGS_1, arg2
#define POINTWISE_ARGS_3 POINTWISE_ARGS_2, arg3
#define POINTWISE_ARGS_11 POINTWISE_ARGS_1, POINTWISE_ARGS_1

// oprand: arg1(self), arg2(self), arg3(other)
#define POINTWISE_OPR_ARGS_1 arg1
#define POINTWISE_OPR_ARGS_2 POINTWISE_ARGS_1, arg3
#define POINTWISE_OPR_ARGS_3 POINTWISE_ARGS_1, arg3, arg4

#define POINTWISE_ARGS_DECL_1 Tensor& arg1 // out
#define POINTWISE_ARGS_DECL_2 POINTWISE_ARGS_DECL_1, const Tensor& arg2
#define POINTWISE_ARGS_DECL_3 POINTWISE_ARGS_DECL_2, const Tensor& arg3
#define POINTWISE_ARGS_DECL_11 POINTWISE_ARGS_DECL_1

#define POINTWISE_CONST_ARGS_DECL_2 const Tensor& arg2
#define POINTWISE_CONST_ARGS_DECL_3 \
  POINTWISE_CONST_ARGS_DECL_2, const Tensor& arg3
#define POINTWISE_CONST_ARGS_DECL_11 POINTWISE_CONST_ARGS_DECL_1

#define POINTWISE_ARG_FOR_TYPE_1 arg1
#define POINTWISE_ARG_FOR_TYPE_2 arg2
#define POINTWISE_ARG_FOR_TYPE_3 POINTWISE_ARG_FOR_TYPE_2
#define POINTWISE_ARG_FOR_TYPE_11 arg1

#define CALLABLE_INIT_ARGS_0
#define CALLABLE_INIT_ARGS_1 val1
#define CALLABLE_INIT_ARGS_2 CALLABLE_INIT_ARGS_1, val2
#define CALLABLE_INIT_ARGS_3 CALLABLE_INIT_ARGS_2, val3

#define CALLABLE_INIT_ARGS_DECL_0
#define CALLABLE_INIT_ARGS_DECL_1 scalar_t val1
#define CALLABLE_INIT_ARGS_DECL_2 CALLABLE_INIT_ARGS_DECL_1, scalar_t val2
#define CALLABLE_INIT_ARGS_DECL_3 CALLABLE_INIT_ARGS_DECL_2, scalar_t val3

#define CHECK_SAME_TENSOR() (TensorImpl_Unwrap(arg1) == TensorImpl_Unwrap(arg2))

#define COMMA_0
#define COMMA_1 ,
#define COMMA_2 COMMA_1
#define COMMA_3 COMMA_2

#define ARGS_OUT_0
#define ARGS_OUT_1 Tensor& arg1

#define REPEAT_AS_ARGLIST_1(r) r
#define REPEAT_AS_ARGLIST_2(r) REPEAT_AS_ARGLIST_1(r), r
#define REPEAT_AS_ARGLIST_3(r) REPEAT_AS_ARGLIST_2(r), r

#define IMPLEMENT_POINTWISE_CALLABLE_(                                        \
    NAME, APPLY_NUM, APPLY_NUM_EXT, CALLABLE, CALLABLE_ARGS_NUM)              \
  template <typename scalar_t>                                                \
  void NAME(POINTWISE_ARGS_DECL_##APPLY_NUM_EXT COMMA_##CALLABLE_ARGS_NUM     \
                CALLABLE_INIT_ARGS_DECL_##CALLABLE_ARGS_NUM) {                \
    if (CHECK_SAME_TENSOR()) {                                                \
      DPCPP_tensor_apply##APPLY_NUM<REPEAT_AS_ARGLIST_##APPLY_NUM(scalar_t)>( \
          POINTWISE_OPR_ARGS_##APPLY_NUM,                                     \
          CALLABLE<scalar_t>(CALLABLE_INIT_ARGS_##CALLABLE_ARGS_NUM));        \
    } else {                                                                  \
      at::AtenIpexTypeXPU::resize_as_(POINTWISE_ARGS_2, c10::nullopt);        \
      DPCPP_tensor_apply##APPLY_NUM_EXT<REPEAT_AS_ARGLIST_##APPLY_NUM_EXT(    \
          scalar_t)>(                                                         \
          POINTWISE_ARGS_##APPLY_NUM_EXT,                                     \
          CALLABLE<scalar_t>(CALLABLE_INIT_ARGS_##CALLABLE_ARGS_NUM));        \
    }                                                                         \
  }

#define IMPLEMENT_POINTWISE_1_CALLABLE_0(NAME, CALLABLE) \
  IMPLEMENT_POINTWISE_CALLABLE_(NAME, 1, 2, CALLABLE, 0)

#define IMPLEMENT_POINTWISE_2_CALLABLE_0(NAME, CALLABLE) \
  IMPLEMENT_POINTWISE_CALLABLE_(NAME, 2, 3, CALLABLE, 0)

#define IMPLEMENT_POINTWISE_1_CALLABLE_1(NAME, CALLABLE) \
  IMPLEMENT_POINTWISE_CALLABLE_(NAME, 1, 2, CALLABLE, 1)

#define IMPLEMENT_POINTWISE_2_CALLABLE_1(NAME, CALLABLE) \
  IMPLEMENT_POINTWISE_CALLABLE_(NAME, 2, 3, CALLABLE, 1)

#define IMPLEMENT_POINTWISE_1_CALLABLE_2(NAME, CALLABLE) \
  IMPLEMENT_POINTWISE_CALLABLE_(NAME, 1, 2, CALLABLE, 2)

#define IMPLEMENT_POINTWISE_2_CALLABLE_2(NAME, CALLABLE) \
  IMPLEMENT_POINTWISE_CALLABLE_(NAME, 2, 3, CALLABLE, 2)

// AT Dispatch
#define IPEX_FUNC_OPS(op, func, real, types)       \
  namespace impl {                                 \
  IMPLEMENT_POINTWISE_1_FUNC(op, func, real)       \
  }                                                \
                                                   \
  Tensor& op(const Tensor& self, Tensor& out) {    \
    IPEX_DISPATCH_##types(                         \
        at::ScalarType::Half,                      \
        at::ScalarType::BFloat16,                  \
        self.scalar_type(),                        \
        #op,                                       \
        [&]() { impl::op<scalar_t>(out, self); }); \
    return out;                                    \
  }

#define IPEX_OUT_INPLACE_UNARY_FUNC_OPS(op, func, real, types) \
  IPEX_FUNC_OPS(op##_out, func, real, types)

#define IPEX_OUT_ALL_UNARY_FUNC_OPS(op, func, real) \
  IPEX_FUNC_OPS(op, func, real, ALL_TYPES_AND2)

#define IPEX_OUT_FLOAT_UNARY_FUNC_OPS(op, func, real) \
  IPEX_FUNC_OPS(op, func, real, FLOATING_TYPES_AND2)

#define IPEX_OUT_INPLACE_ALL_UNARY_FUNC_OPS(op, func, real) \
  IPEX_OUT_INPLACE_UNARY_FUNC_OPS(op, func, real, ALL_TYPES_AND2)

#define IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(op, func, real) \
  IPEX_OUT_INPLACE_UNARY_FUNC_OPS(op, func, real, FLOATING_TYPES_AND2)

// Customized Callable Ops
#define SCALAR_ARGS_0
#define SCALAR_ARGS_1 val1.to<scalar_t>()
#define SCALAR_ARGS_2 SCALAR_ARGS_1, val2.to<scalar_t>()
#define SCALAR_ARGS_3 SCALAR_ARGS_2, val3.to<scalar_t>()

#define SCALAR_ARGS_DECL_0
#define SCALAR_ARGS_DECL_1 const Scalar& val1
#define SCALAR_ARGS_DECL_2 SCALAR_ARGS_DECL_1, const Scalar& val2
#define SCALAR_ARGS_DECL_3 SCALAR_ARGS_DECL_2, const Scalar& val3

#define IPEX_INT_OR_FLOAT_CALLABLE_OPS(                              \
    op, callable, types, oprand_num, arg_num, callable_args_num)     \
  namespace impl {                                                   \
  IMPLEMENT_POINTWISE_##oprand_num##_CALLABLE_##callable_args_num(   \
      op,                                                            \
      callable)                                                      \
  }                                                                  \
                                                                     \
  Tensor& op(POINTWISE_ARGS_DECL_##arg_num COMMA_##callable_args_num \
                 SCALAR_ARGS_DECL_##callable_args_num) {             \
    IPEX_DISPATCH_##types(                                           \
        POINTWISE_ARG_FOR_TYPE_##arg_num.scalar_type(), #op, [&]() { \
          impl::op<scalar_t>(                                        \
              POINTWISE_ARGS_##arg_num COMMA_##callable_args_num     \
                  SCALAR_ARGS_##callable_args_num);                  \
        });                                                          \
    return arg1;                                                     \
  }

#define IPEX_CALLABLE_OPS(                                                   \
    op,                                                                      \
    callable,                                                                \
    arg_name,                                                                \
    out_num,                                                                 \
    types,                                                                   \
    oprand_num,                                                              \
    arg_num,                                                                 \
    callable_args_num)                                                       \
  namespace impl {                                                           \
  IMPLEMENT_POINTWISE_##oprand_num##_CALLABLE_##callable_args_num(           \
      op,                                                                    \
      callable)                                                              \
  }                                                                          \
                                                                             \
  Tensor& op(POINTWISE_##arg_name##_DECL_##arg_num COMMA_##callable_args_num \
                 SCALAR_ARGS_DECL_##callable_args_num COMMA_##out_num        \
                     ARGS_OUT_##out_num) {                                   \
    IPEX_DISPATCH_##types(                                                   \
        at::ScalarType::Half,                                                \
        at::ScalarType::BFloat16,                                            \
        POINTWISE_ARG_FOR_TYPE_##arg_num.scalar_type(),                      \
        #op,                                                                 \
        [&]() {                                                              \
          impl::op<scalar_t>(                                                \
              POINTWISE_ARGS_##arg_num COMMA_##callable_args_num             \
                  SCALAR_ARGS_##callable_args_num);                          \
        });                                                                  \
    return arg1;                                                             \
  }

// Unary
#define IPEX_OUT_ALL_CALLABLE_0_UNARY_OPS(op, callable) \
  IPEX_CALLABLE_OPS(op, callable, ARGS, 0, ALL_TYPES_AND2, 1, 2, 0)

#define IPEX_OUT_FLOAT_CALLABLE_0_UNARY_OPS(op, callable) \
  IPEX_CALLABLE_OPS(op, callable, ARGS, 0, FLOATING_TYPES_AND2, 1, 2, 0)

#define IPEX_OUT_FLOAT_AND_HALF_CALLABLE_0_UNARY_OPS(op, callable) \
  IPEX_INT_OR_FLOAT_CALLABLE_OPS(op, callable, FLOATING_TYPES_AND_HALF, 1, 2, 0)

#define IPEX_ALL_CALLABLE_1_UNARY_OPS(op, callable) \
  IPEX_CALLABLE_OPS(                                \
      op, callable, ARGS, 0, ALL_TYPES_AND2, 1, 11 /* inplace */, 1)

#define IPEX_FLOAT_CALLABLE_1_UNARY_OPS(op, callable) \
  IPEX_CALLABLE_OPS(                                  \
      op, callable, ARGS, FLOATING_TYPES_AND2, 1, 11 /* inplace */, 1)

#define IPEX_INT_CALLABLE_1_UNARY_OPS(op, callable) \
  IPEX_INT_OR_FLOAT_CALLABLE_OPS(                   \
      op, callable, INTEGRAL_TYPES, 1, 11 /* inplace */, 1)

#define IPEX_OUT_ALL_CALLABLE_1_UNARY_OPS(op, callable) \
  IPEX_CALLABLE_OPS(op, callable, ARGS, 0, ALL_TYPES_AND2, 1, 2, 1)

#define IPEX_OUT_ALL_CALLABLE_1_CONST_UNARY_OPS(op, callable) \
  IPEX_CALLABLE_OPS(op, callable, CONST_ARGS, 1, ALL_TYPES_AND2, 1, 2, 1)

#define IPEX_OUT_FLOAT_CALLABLE_1_UNARY_OPS(op, callable) \
  IPEX_CALLABLE_OPS(op, callable, ARGS, 0, FLOATING_TYPES_AND2, 1, 2, 1)

#define IPEX_OUT_INT_CALLABLE_1_UNARY_OPS(op, callable) \
  IPEX_INT_OR_FLOAT_CALLABLE_OPS(op, callable, INTEGRAL_TYPES, 1, 2, 1)

#define IPEX_ALL_CALLABLE_2_UNARY_OPS(op, callable) \
  IPEX_CALLABLE_OPS(                                \
      op, callable, ARGS, 0, ALL_TYPES_AND2, 1, 11 /* inplace */, 2)

#define IPEX_FLOAT_CALLABLE_2_UNARY_OPS(op, callable) \
  IPEX_CALLABLE_OPS(                                  \
      op, callable, ARGS, FLOATING_TYPES_AND2, 1, 11 /* inplace */, 2)

#define IPEX_INT_CALLABLE_2_UNARY_OPS(op, callable) \
  IPEX_INT_OR_FLOAT_CALLABLE_OPS(                   \
      op, callable, INTEGRAL_TYPES, 1, 11 /* inplace */, 2)

#define IPEX_OUT_ALL_CALLABLE_2_UNARY_OPS(op, callable) \
  IPEX_CALLABLE_OPS(op, callable, ARGS, 0, ALL_TYPES_AND2, 1, 2, 2)

#define IPEX_OUT_ALL_CALLABLE_2_CONST_UNARY_OPS(op, callable) \
  IPEX_CALLABLE_OPS(op, callable, CONST_ARGS, 1, ALL_TYPES_AND2, 1, 2, 2)

#define IPEX_OUT_FLOAT_CALLABLE_2_UNARY_OPS(op, callable) \
  IPEX_CALLABLE_OPS(op, callable, ARGS, 0, FLOATING_TYPES_AND2, 1, 2, 2)

#define IPEX_OUT_INT_CALLABLE_2_UNARY_OPS(op, callable) \
  IPEX_INT_OR_FLOAT_CALLABLE_OPS(op, callable, INTEGRAL_TYPES, 1, 2, 2)

// Binary
#define IPEX_OUT_ALL_CALLABLE_0_BINARY_OPS(op, callable) \
  IPEX_CALLABLE_OPS(op, callable, ARGS, 0, ALL_TYPES_AND2, 2, 3, 0)

#define IPEX_OUT_FLOAT_CALLABLE_0_BINARY_OPS(op, callable) \
  IPEX_CALLABLE_OPS(op, callable, ARGS, 0, FLOATING_TYPES_AND2, 2, 3, 0)

#define IPEX_OUT_INT_CALLABLE_0_BINARY_OPS(op, callable) \
  IPEX_INT_OR_FLOAT_CALLABLE_OPS(op, callable, INTEGRAL_TYPES, 2, 3, 0)

#define IPEX_OUT_ALL_CALLABLE_1_BINARY_OPS(op, callable) \
  IPEX_CALLABLE_OPS(op, callable, ARGS, 0, ALL_TYPES_AND2, 2, 3, 1)

#define IPEX_OUT_FLOAT_CALLABLE_1_BINARY_OPS(op, callable) \
  IPEX_CALLABLE_OPS(op, callable, ARGS, 0, FLOATING_TYPES_AND2, 2, 3, 1)

#define IPEX_OUT_INT_CALLABLE_1_BINARY_OPS(op, callable) \
  IPEX_INT_OR_FLOAT_CALLABLE_OPS(op, callable, INTEGRAL_TYPES, 2, 3, 1)

#define IPEX_OUT_ALL_CALLABLE_2_BINARY_OPS(op, callable) \
  IPEX_CALLABLE_OPS(op, callable, ARGS, 0, ALL_TYPES_AND2, 2, 3, 2)

#define IPEX_OUT_FLOAT_CALLABLE_2_BINARY_OPS(op, callable) \
  IPEX_CALLABLE_OPS(op, callable, ARGS, 0, FLOATING_TYPES_AND2, 2, 3, 2)

#define IPEX_OUT_INT_CALLABLE_2_BINARY_OPS(op, callable) \
  IPEX_INT_OR_FLOAT_CALLABLE_OPS(op, callable, INTEGRAL_TYPES, 2, 3, 2)

template <typename T>
struct TensorATan2Op {
  void operator()(T& out, T& a, T& b) const {
    out = Numerics<T>::atan2(a, b);
  }
};

template <typename T>
struct TensorSigmoidOp {
  void operator()(T& out, T& in) const {
    T one = (T)1.0;
    out = one / (one + Numerics<T>::exp(-static_cast<T>(in)));
  }

  void operator()(T& v) const {
    T one = (T)1.0;
    v = one / (one + Numerics<T>::exp(-static_cast<T>(v)));
  }
};

template <typename T>
struct TensorSigmoidGradOp {
  void operator()(T& gradInput, T& output, T& gradOutput) const {
    gradInput = gradOutput * (1.f - output) * output;
  }
};

template <>
struct TensorSigmoidGradOp<at::Half> {
  void operator()(at::Half& gradInput, at::Half& output, at::Half& gradOutput)
      const {
    float out = (float)output;
    float go = (float)gradOutput;
    gradInput = (at::Half)(go * (1.f - out) * out);
  }
};

/*
 * The following function was converted to DPCPP form from code that comes
 * with the following copyright notice. It has been released under the BSD
 * license.
 *
 * Cephes Math Library Release 2.8:  June, 2000
 * Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
 */
template <typename T>
struct TensorDigammaOp {
  void operator()(T& out, T& in) const {
    using compute_type = typename std::
        conditional<std::is_same<T, at::Half>::value, float, T>::type;
    static const double PI_f64 = 3.14159265358979323846;
    static const compute_type PSI_10 = 2.25175258906672110764;
    static const compute_type A[] = {
        8.33333333333333333333E-2,
        -2.10927960927960927961E-2,
        7.57575757575757575758E-3,
        -4.16666666666666666667E-3,
        3.96825396825396825397E-3,
        -8.33333333333333333333E-3,
        8.33333333333333333333E-2,
    };

    auto x = scalar_cast<compute_type>(in);
    if (x == 0) {
      out = std::numeric_limits<T>::infinity();
      return;
    }

    bool x_is_integer = (x == DPCPP::floor(x));
    compute_type result = 0;

    if (x < 0) {
      if (x_is_integer) {
        out = std::numeric_limits<T>::infinity();
        return;
      }

      // Rounding errors in tan's input can really affect the output
      // for extreme values, so we always perform this computation in double.
      result = scalar_cast<compute_type>(
          -PI_f64 / DPCPP::tan(PI_f64 * scalar_cast<double>(x)));
      x = 1 - x;
    }

    while (x < 10) {
      result -= 1 / x;
      x += 1;
    }

    if (x == 10) {
      out = scalar_cast<T>(result + PSI_10);
      return;
    }

    compute_type y = 0;
    if (x < 1.0e17) {
      compute_type z = 1.0 / (x * x);
      compute_type polevl_result = 0;
      for (int i = 0; i <= 6; i++) {
        polevl_result = polevl_result * z + A[i];
      }
      y = z * polevl_result;
    }

    out = scalar_cast<T>(DPCPP::log(x) - (0.5 / x) - y + result);
    return;
  }

  void operator()(T& v) const {
    using compute_type = typename std::
        conditional<std::is_same<T, at::Half>::value, float, T>::type;
    static const double PI_f64 = 3.14159265358979323846;
    static const compute_type PSI_10 = 2.25175258906672110764;
    static const compute_type A[] = {
        8.33333333333333333333E-2,
        -2.10927960927960927961E-2,
        7.57575757575757575758E-3,
        -4.16666666666666666667E-3,
        3.96825396825396825397E-3,
        -8.33333333333333333333E-3,
        8.33333333333333333333E-2,
    };

    auto x = scalar_cast<compute_type>(v);
    if (x == 0) {
      v = std::numeric_limits<T>::infinity();
      return;
    }

    bool x_is_integer = (x == DPCPP::floor(x));
    compute_type result = 0;

    if (x < 0) {
      if (x_is_integer) {
        v = std::numeric_limits<T>::infinity();
        return;
      }

      // Rounding errors in tan's input can really affect the output
      // for extreme values, so we always perform this computation in double.
      result = scalar_cast<compute_type>(
          -PI_f64 / DPCPP::tan(PI_f64 * scalar_cast<double>(x)));
      x = 1 - x;
    }

    while (x < 10) {
      result -= 1 / x;
      x += 1;
    }

    if (x == 10) {
      v = scalar_cast<T>(result + PSI_10);
      return;
    }

    compute_type y = 0;
    if (x < 1.0e17) {
      compute_type z = 1.0 / (x * x);
      compute_type polevl_result = 0;
      for (int i = 0; i <= 6; i++) {
        polevl_result = polevl_result * z + A[i];
      }
      y = z * polevl_result;
    }

    v = scalar_cast<T>(DPCPP::log(x) - (0.5 / x) - y + result);
    return;
  }
};

template <typename T>
struct TensorErfinvOp {
  void operator()(T& out, T& in) const {
    using compute_type = typename std::conditional<
        std::is_same<T, at::Half>::value ||
            std::is_same<T, at::BFloat16>::value,
        float,
        T>::type;
    compute_type z, num, dem;
    static const double PI_f64 = 3.14159265358979323846;
    static const compute_type a[4] = {
        0.886226899, -1.645349621, 0.914624893, -0.140543331};
    static const compute_type b[4] = {
        -2.118377725, 1.442710462, -0.329097515, 0.012229801};
    static const compute_type c[4] = {
        -1.970840454, -1.624906493, 3.429567803, 1.641345311};
    static const compute_type d[2] = {3.543889200, 1.637067800};

    auto x = scalar_cast<compute_type>(in);
    if (DPCPP::fabs(x) > 1.0) {
      out = scalar_cast<T>(NAN);
      return;
    }
    if (DPCPP::fabs(x) == 1.0) {
      out = scalar_cast<T>(
          (DPCPP::copysign(1.0, scalar_cast<double>(x))) *
          (std::numeric_limits<double>::infinity()));
      return;
    }
    if (DPCPP::fabs(x) <= 0.7) {
      z = x * x;
      num = (((a[3] * z + a[2]) * z + a[1]) * z + a[0]);
      dem =
          ((((b[3] * z + b[2]) * z + b[1]) * z + b[0]) * z +
           scalar_cast<compute_type>(1.0));
      out = x * num / dem;
    } else {
      z = scalar_cast<compute_type>(
          DPCPP::sqrt(-DPCPP::log((1.0 - DPCPP::fabs(x)) / 2.0)));
      num = ((c[3] * z + c[2]) * z + c[1]) * z + c[0];
      dem = (d[1] * z + d[0]) * z + scalar_cast<compute_type>(1.0);
      out = scalar_cast<T>(
          scalar_cast<compute_type>(
              DPCPP::copysign(1.0, scalar_cast<double>(x))) *
          num / dem);
    }
    out = out -
        scalar_cast<T>(
              (DPCPP::erf(scalar_cast<double>(out)) - x) /
              ((2.0 / DPCPP::sqrt(PI_f64)) * DPCPP::exp(-x * x)));
    out = out -
        scalar_cast<T>(
              (DPCPP::erf(scalar_cast<double>(out)) - x) /
              ((2.0 / DPCPP::sqrt(PI_f64)) * DPCPP::exp(-x * x)));
    return;
  }

  void operator()(T& v) const {
    using compute_type = typename std::conditional<
        std::is_same<T, at::Half>::value ||
            std::is_same<T, at::BFloat16>::value,
        float,
        T>::type;
    compute_type z, num, dem;
    static const double PI_f64 = 3.14159265358979323846;
    static const compute_type a[4] = {
        0.886226899, -1.645349621, 0.914624893, -0.140543331};
    static const compute_type b[4] = {
        -2.118377725, 1.442710462, -0.329097515, 0.012229801};
    static const compute_type c[4] = {
        -1.970840454, -1.624906493, 3.429567803, 1.641345311};
    static const compute_type d[2] = {3.543889200, 1.637067800};

    auto x = scalar_cast<compute_type>(v);
    if (DPCPP::fabs(x) > 1.0) {
      v = scalar_cast<T>(NAN);
      return;
    }
    if (DPCPP::fabs(x) == 1.0) {
      v = scalar_cast<T>(
          (DPCPP::copysign(1.0, scalar_cast<double>(x))) *
          (std::numeric_limits<double>::infinity()));
      return;
    }
    if (DPCPP::fabs(x) <= 0.7) {
      z = x * x;
      num = (((a[3] * z + a[2]) * z + a[1]) * z + a[0]);
      dem =
          ((((b[3] * z + b[2]) * z + b[1]) * z + b[0]) * z +
           scalar_cast<compute_type>(1.0));
      v = x * num / dem;
    } else {
      z = scalar_cast<compute_type>(
          DPCPP::sqrt(-DPCPP::log((1.0 - DPCPP::fabs(x)) / 2.0)));
      num = ((c[3] * z + c[2]) * z + c[1]) * z + c[0];
      dem = (d[1] * z + d[0]) * z + scalar_cast<compute_type>(1.0);
      v = scalar_cast<T>(
          scalar_cast<compute_type>(
              DPCPP::copysign(1.0, scalar_cast<double>(x))) *
          num / dem);
    }
    v = v -
        scalar_cast<T>(
            (DPCPP::erf(scalar_cast<double>(v)) - x) /
            ((2.0 / DPCPP::sqrt(PI_f64)) * DPCPP::exp(-x * x)));
    v = v -
        scalar_cast<T>(
            (DPCPP::erf(scalar_cast<double>(v)) - x) /
            ((2.0 / DPCPP::sqrt(PI_f64)) * DPCPP::exp(-x * x)));
    return;
  }
};

template <typename T, typename acc>
struct TensorReciprocalOp {
  void operator()(T& out, T& in) const {
    out = static_cast<acc>(1) / in;
  }

  void operator()(T& v) const {
    v = static_cast<acc>(1) / v;
  }
};

template <typename T>
struct TensorSignOp {
  void operator()(T& out, T& in) const {
    T orig = in;
    out = (orig > 0) - (orig < 0);
  }

  void operator()(T& v) const {
    T orig = v;
    v = (orig > 0) - (orig < 0);
  }
};

template <>
struct TensorSignOp<unsigned char> {
  void operator()(unsigned char& out, unsigned char& in) const {
    unsigned char orig = in;
    out = (orig == 0) ? 0 : 1;
  }

  void operator()(unsigned char& v) const {
    unsigned char orig = v;
    v = (orig == 0) ? 0 : 1;
  }
};

template <>
struct TensorSignOp<bool> {
  void operator()(bool& out, bool& in) const {
    out = in;
  }

  void operator()(bool& v) const {}
};

template <typename T>
struct TensorClampOp {
  TensorClampOp(T min, T max) : minValue(min), maxValue(max) {}
  void operator()(T& out, T& in) const {
    T val = Numerics<T>::lt(in, maxValue) ? in : maxValue;
    out = Numerics<T>::gt(minValue, val) ? minValue : val;
  }

  void operator()(T& v) const {
    T val = Numerics<T>::lt(v, maxValue) ? v : maxValue;
    v = Numerics<T>::gt(minValue, val) ? minValue : val;
  }

  const T minValue;
  const T maxValue;
};

template <typename T>
struct TensorCrossOp {
  TensorCrossOp(int64_t sx, int64_t sy, int64_t so) : sx(sx), sy(sy), so(so) {}

  void operator()(T& out, T& x, T& y) const {
    T val0 = Numerics<T>::sub(
        Numerics<T>::mul((&x)[1 * sx], (&y)[2 * sy]),
        Numerics<T>::mul((&x)[2 * sx], (&y)[1 * sy]));

    T val1 = Numerics<T>::sub(
        Numerics<T>::mul((&x)[2 * sx], (&y)[0 * sy]),
        Numerics<T>::mul((&x)[0 * sx], (&y)[2 * sy]));

    T val2 = Numerics<T>::sub(
        Numerics<T>::mul((&x)[0 * sx], (&y)[1 * sy]),
        Numerics<T>::mul((&x)[1 * sx], (&y)[0 * sy]));

    (&out)[0 * so] = val0;
    (&out)[1 * so] = val1;
    (&out)[2 * so] = val2;
  }

  const int64_t sx, sy, so;
};

template <typename T>
struct TensorLerpOp {
  TensorLerpOp(T w) : w(w) {}

  void operator()(T& out, T& a, T& b) const {
    out = Numerics<T>::add(a, Numerics<T>::mul(w, Numerics<T>::sub(b, a)));
  }

  const T w;
};

template <typename T>
struct TensorMaxOp {
  void operator()(T& out, T& in) const {
    out = Numerics<T>::gt(out, in) ? out : in;
  }

  void operator()(T& out, T& in1, T& in2) const {
    out = Numerics<T>::gt(in1, in2) ? in1 : in2;
  }
};

template <typename T>
struct TensorMinOp {
  void operator()(T& out, T& in) const {
    out = Numerics<T>::lt(out, in) ? out : in;
  }

  void operator()(T& out, T& in1, T& in2) const {
    out = Numerics<T>::lt(in1, in2) ? in1 : in2;
  }
};

template <typename T>
struct TensorMaxValueOp {
  TensorMaxValueOp(T v) : val(v) {}

  inline void operator()(T& out) const {
    out = Numerics<T>::lt(out, val) ? val : out; // this order propagates NaN
  }

  inline void operator()(T& out, T& in) const {
    out = Numerics<T>::lt(in, val) ? val : in; // this order propagates NaN
  }

  T val;
};

template <typename T>
struct TensorMinValueOp {
  TensorMinValueOp(T v) : val(v) {}

  void operator()(T& out) const {
    out = Numerics<T>::gt(out, val) ? val : out; // this order propagates NaN
  }

  void operator()(T& out, T& in) const {
    out = Numerics<T>::gt(in, val) ? val : in; // this order propagates NaN
  }

  T val;
};

template <typename T>
struct TensorAddCMulOp {
  TensorAddCMulOp(T v) : val(v) {}

  void operator()(T& out, T& in1, T& in2) const {
    out = Numerics<T>::add(
        out, Numerics<T>::mul(val, Numerics<T>::mul(in1, in2)));
  }

  T val;
};

template <typename T>
struct TensorAddCDivOp {
  TensorAddCDivOp(T v) : val(v) {}

  void operator()(T& out, T& in1, T& in2) const {
    out = Numerics<T>::add(
        out, Numerics<T>::mul(val, Numerics<T>::div(in1, in2)));
  }

  T val;
};

template <typename T>
struct TensorBitAndOp {
  void operator()(T& out, T& in) const {
    out &= in;
  }

  void operator()(T& out, T& in1, T& in2) const {
    out = in1 & in2;
  }
};

template <typename T>
struct TensorBitOrOp {
  void operator()(T& out, T& in) const {
    out |= in;
  }

  void operator()(T& out, T& in1, T& in2) const {
    out = in1 | in2;
  }
};

template <typename T>
struct TensorBitXorOp {
  void operator()(T& out, T& in) const {
    out ^= in;
  }

  void operator()(T& out, T& in1, T& in2) const {
    out = in1 ^ in2;
  }
};

template <typename T>
static typename std::enable_if<std::is_signed<T>::value, bool>::type modulo_wrap(
    T a,
    T b) {
  return (a != 0) && (a < 0) != (b < 0);
}

template <typename T>
typename std::enable_if<std::is_unsigned<T>::value, bool>::type modulo_wrap(
    T a,
    T b) {
  return false;
}

template <typename T>
struct TensorCRemainderOp {
  void operator()(T& out, T& in) const {
    T val = out % in;
    if (modulo_wrap(val, in)) {
      val += in;
    }
    out = val;
  }

  void operator()(T& out, T& in1, T& in2) const {
    T val = in1 % in2;
    if (modulo_wrap(val, in2)) {
      val += in2;
    }
    out = val;
  }
};

template <>
struct TensorCRemainderOp<float> {
  void operator()(float& out, float& in) const {
    out = in != 0.f ? out - in * DPCPP::floor(out / in) : NAN;
  }

  void operator()(float& out, float& in1, float& in2) const {
    out = in2 != 0.f ? in1 - in2 * DPCPP::floor(in1 / in2) : NAN;
  }
};

template <>
struct TensorCRemainderOp<double> {
  void operator()(double& out, double& in) const {
    out = in != 0. ? out - in * DPCPP::floor(out / in) : NAN;
  }

  void operator()(double& out, double& in1, double& in2) const {
    out = in2 != 0. ? in1 - in2 * DPCPP::floor(in1 / in2) : NAN;
  }
};

template <>
struct TensorCRemainderOp<at::Half> {
  void operator()(at::Half& out, at::Half& in) const {
    out = in != 0.f ? out - in * DPCPP::floor(float(out / in)) : NAN;
  }

  void operator()(at::Half& out, at::Half& in1, at::Half& in2) const {
    out = in2 != 0.f ? in1 - in2 * DPCPP::floor(float(in1 / in2)) : NAN;
  }
};

template <>
struct TensorCRemainderOp<at::BFloat16> {
  void operator()(at::BFloat16& out, at::BFloat16& in) const {
    out = in != 0.f ? out - in * DPCPP::floor(float(out / in)) : NAN;
  }

  void operator()(at::BFloat16& out, at::BFloat16& in1, at::BFloat16& in2)
      const {
    out = in2 != 0.f ? in1 - in2 * DPCPP::floor(float(in1 / in2)) : NAN;
  }
};

template <typename T>
struct TensorCFmodOp {
  void operator()(T& out, T& in) const {
    out = out % in;
  }

  void operator()(T& out, T& in1, T& in2) const {
    out = in1 % in2;
  }
};

template <>
struct TensorCFmodOp<float> {
  void operator()(float& out, float& in) const {
    out = DPCPP::fmod(out, in);
  }

  void operator()(float& out, float& in1, float& in2) const {
    out = DPCPP::fmod(in1, in2);
  }
};

template <>
struct TensorCFmodOp<double> {
  void operator()(double& out, double& in) const {
    out = DPCPP::fmod(out, in);
  }

  void operator()(double& out, double& in1, double& in2) const {
    out = DPCPP::fmod(in1, in2);
  }
};

template <>
struct TensorCFmodOp<at::Half> {
  void operator()(at::Half& out, at::Half& in) const {
    out = DPCPP::fmod(float(out), float(in));
  }

  void operator()(at::Half& out, at::Half& in1, at::Half& in2) const {
    out = DPCPP::fmod(float(in1), float(in2));
  }
};

template <>
struct TensorCFmodOp<at::BFloat16> {
  void operator()(at::BFloat16& out, at::BFloat16& in) const {
    out = DPCPP::fmod(float(out), float(in));
  }

  void operator()(at::BFloat16& out, at::BFloat16& in1, at::BFloat16& in2)
      const {
    out = DPCPP::fmod(float(in1), float(in2));
  }
};

template <typename T>
struct TensorCPowOp {
  void operator()(T& out, T& in) const {
    out = Numerics<T>::pow(out, in);
  }

  void operator()(T& out, T& in1, T& in2) const {
    out = Numerics<T>::pow(in1, in2);
  }
};

template <>
struct TensorCPowOp<float> {
  void operator()(float& out, float& in) const {
    out = DPCPP::pow(out, in);
  }

  void operator()(float& out, float& in1, float& in2) const {
    out = DPCPP::pow(in1, in2);
  }
};

template <>
struct TensorCPowOp<double> {
  void operator()(double& out, double& in) const {
    out = DPCPP::pow(out, in);
  }

  void operator()(double& out, double& in1, double& in2) const {
    out = DPCPP::pow(in1, in2);
  }
};

template <typename T, int StaticExp>
struct TensorPowOp {
  TensorPowOp(T v) : val(v) {}
  void operator()(T& out, T& in) const {
    if (StaticExp == 1) {
      out = in;
    } else if (StaticExp == 2) {
      out = Numerics<T>::mul(in, in);
    } else if (StaticExp == 3) {
      T square = Numerics<T>::mul(in, in);
      out = Numerics<T>::mul(square, in);
    } else {
      out = Numerics<T>::pow(in, val);
    }
  }

  void operator()(T& v) const {
    if (StaticExp == 1) {
      v = v;
    } else if (StaticExp == 2) {
      v = Numerics<T>::mul(v, v);
    } else if (StaticExp == 3) {
      v = Numerics<T>::mul(Numerics<T>::mul(v, v), v);
    } else {
      v = Numerics<T>::pow(v, val);
    }
  }

  const T val;
};

template <typename T>
struct TensorPowOp<T, -1> {
  TensorPowOp(T v) : val(v) {}
  void operator()(T& out, T& in) const {
    out = Numerics<T>::cinv(in);
  }

  void operator()(T& v) const {
    v = Numerics<T>::cinv(v);
  }

  const T val;
};

template <typename T>
struct TensorPowOp<T, -2> {
  TensorPowOp(T v) : val(v) {}
  void operator()(T& out, T& in) const {
    T square = Numerics<T>::mul(in, in);
    out = Numerics<T>::cinv(square);
  }

  void operator()(T& v) const {
    T square = Numerics<T>::mul(v, v);
    v = Numerics<T>::cinv(square);
  }

  const T val;
};

template <typename T>
struct TensorTPowOp {
  TensorTPowOp(T v) : val(v) {}

  void operator()(T& out, T& in) const {
    out = Numerics<T>::pow(val, in);
  }

  void operator()(T& v) const {
    v = Numerics<T>::pow(val, v);
  }

  const T val;
};

// TODO: support complex type
template <typename T>
struct TensorRealOp {
  void operator()(T& out, T& in) const {
    out = in;
  }
};

// TODO: support complex type
template <typename T>
struct TensorConjOp {
  void operator()(T& out, T& in) const {
    out = in;
  }
};

} // namespace AtenIpexTypeXPU
} // namespace at
