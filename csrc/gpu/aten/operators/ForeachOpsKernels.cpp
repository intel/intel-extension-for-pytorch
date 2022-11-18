#include <ATen/Dispatch.h>
#include <ATen/native/Fill.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/TensorIterator.h>
#include <aten/core/detail/IndexUtils.h>

#include <runtime/Utils.h>
#include "ATen/OpMathType.h"
#include "comm/ATDispatch.h"
#include "comm/ApplyUtils.h"
#include "comm/RegistrationDeclarations.h"

#include "ForeachFunctors.h"
#include "Loops.h"
#include "MultiTensorApply.h"
#include "comm/Numerics.h"

namespace at {

namespace {
bool check_complex(at::TensorList tensors) {
  return std::any_of(tensors.begin(), tensors.end(), [](const auto& t) {
    return at::isComplexType(t.scalar_type());
  });
}
} // namespace
namespace AtenIpexTypeXPU {
template <typename scalar_t, template <class> class Op>
std::vector<Tensor> foreach_unary_op(TensorList tensors) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors.size());
  for (const auto& t : tensors) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  tensor_lists.emplace_back(tensors.vec());
  tensor_lists.emplace_back(std::move(vec_res));

  using opmath_t = typename at::opmath_type<scalar_t>;
  multi_tensor_apply<2>(
      tensor_lists,
      UnaryOpFunctor<
          scalar_t,
          /* depth */ 2,
          /* r_args_depth */ 1,
          /* res_arg_index */ 1>(),
      Op<opmath_t>());
  return tensor_lists[1];
}

template <typename scalar_t, template <class> class Op>
void foreach_unary_op_(TensorList tensors) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(tensors.vec());
  using opmath_t = typename at::opmath_type<scalar_t>;
  multi_tensor_apply<1>(
      tensor_lists,
      UnaryOpFunctor<
          scalar_t,
          /* depth */ 1,
          /* r_args_depth */ 1,
          /* res_arg_index */ 0>(),
      Op<opmath_t>());
}

#define REGISTER_FOREACH_UNARY_OPS_1(func_name, macro, Type1)               \
  template <template <class> class Op>                                      \
  std::vector<Tensor> func_name(TensorList tensors) {                       \
    return macro(                                                           \
        Type1, tensors[0].scalar_type(), "foreach_unary_op_sycl", [&]() {   \
          return foreach_unary_op<scalar_t, Op>(tensors);                   \
        });                                                                 \
  }                                                                         \
                                                                            \
  template <template <class> class Op>                                      \
  void func_name##_(TensorList tensors) {                                   \
    macro(Type1, tensors[0].scalar_type(), "foreach_unary_op_sycl", [&]() { \
      foreach_unary_op_<scalar_t, Op>(tensors);                             \
    });                                                                     \
  }

#define REGISTER_FOREACH_UNARY_OPS_2(func_name, macro, Type1, Type2) \
  template <template <class> class Op>                               \
  std::vector<Tensor> func_name(TensorList tensors) {                \
    return macro(                                                    \
        Type1,                                                       \
        Type2,                                                       \
        tensors[0].scalar_type(),                                    \
        "foreach_unary_op_sycl",                                     \
        [&]() { return foreach_unary_op<scalar_t, Op>(tensors); });  \
  }                                                                  \
                                                                     \
  template <template <class> class Op>                               \
  void func_name##_(TensorList tensors) {                            \
    macro(                                                           \
        Type1,                                                       \
        Type2,                                                       \
        tensors[0].scalar_type(),                                    \
        "foreach_unary_op_sycl",                                     \
        [&]() { foreach_unary_op_<scalar_t, Op>(tensors); });        \
  }

#define REGISTER_FOREACH_UNARY_OPS_3(func_name, macro, Type1, Type2, Type3) \
  template <template <class> class Op>                                      \
  std::vector<Tensor> func_name(TensorList tensors) {                       \
    return macro(                                                           \
        Type1,                                                              \
        Type2,                                                              \
        Type3,                                                              \
        tensors[0].scalar_type(),                                           \
        "foreach_unary_op_sycl",                                            \
        [&]() { return foreach_unary_op<scalar_t, Op>(tensors); });         \
  }                                                                         \
                                                                            \
  template <template <class> class Op>                                      \
  void func_name##_(TensorList tensors) {                                   \
    macro(                                                                  \
        Type1,                                                              \
        Type2,                                                              \
        Type3,                                                              \
        tensors[0].scalar_type(),                                           \
        "foreach_unary_op_sycl",                                            \
        [&]() { foreach_unary_op_<scalar_t, Op>(tensors); });               \
  }

REGISTER_FOREACH_UNARY_OPS_1(
    floating_complex_half,
    IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1,
    ScalarType::Half)
REGISTER_FOREACH_UNARY_OPS_1(
    floating_half,
    IPEX_DISPATCH_FLOATING_TYPES_AND,
    ScalarType::Half)
REGISTER_FOREACH_UNARY_OPS_2(
    floating_complex_half_bfloat16,
    IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2,
    ScalarType::Half,
    ScalarType::BFloat16)
REGISTER_FOREACH_UNARY_OPS_2(
    all_types_half_complex_bfloat16,
    IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2,
    ScalarType::Half,
    ScalarType::BFloat16)
REGISTER_FOREACH_UNARY_OPS_2(
    floating_half_bfloat16,
    IPEX_DISPATCH_FLOATING_TYPES_AND2,
    ScalarType::Half,
    ScalarType::BFloat16)
REGISTER_FOREACH_UNARY_OPS_3(
    all_types_complex_bfloat16_half_bool,
    IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3,
    ScalarType::Half,
    ScalarType::BFloat16,
    ScalarType::Bool)

#define OP_CUSTOM_FUNCTOR(function, op_name, functor_name)                 \
  std::vector<Tensor> _foreach_##op_name(TensorList tensors) {             \
    at::native::check_foreach_api_restrictions(tensors);                   \
    if (!at::native::can_use_fast_route(tensors) ||                        \
        at::native::has_integral_tensor(tensors, /*includeBool */ true)) { \
      return at::native::foreach_tensor_##op_name##_slow(tensors);         \
    }                                                                      \
    return function<functor_name>(tensors);                                \
  }                                                                        \
  void _foreach_##op_name##_(TensorList tensors) {                         \
    at::native::check_foreach_api_restrictions(tensors);                   \
    if (!at::native::can_use_fast_route(tensors) ||                        \
        at::native::has_integral_tensor(tensors, /*includeBool */ true)) { \
      return at::native::foreach_tensor_##op_name##_slow_(tensors);        \
    }                                                                      \
    function##_<functor_name>(tensors);                                    \
  }

#define STD_FUNCTOR(op_name, functor_name) \
  template <typename T>                    \
  struct functor_name {                    \
    T operator()(T t) const {              \
      return std::op_name(t);              \
    }                                      \
  }

#define NUM_FUNCTOR(op_name, functor_name) \
  template <typename T>                    \
  struct functor_name {                    \
    T operator()(T t) const {              \
      return Numerics<T>::op_name(t);      \
    }                                      \
  }

#define OP(function, op_name, functor_name) \
  STD_FUNCTOR(op_name, functor_name);       \
  OP_CUSTOM_FUNCTOR(function, op_name, functor_name)

#define NUM_OP(function, op_name, functor_name) \
  NUM_FUNCTOR(op_name, functor_name);           \
  OP_CUSTOM_FUNCTOR(function, op_name, functor_name)

NUM_OP(floating_half_bfloat16, erfc, Erfc);
NUM_OP(floating_half_bfloat16, expm1, Expm1);
OP(floating_half, lgamma, Lgamma);
NUM_OP(floating_half_bfloat16, trunc, Truncf);
NUM_OP(floating_half_bfloat16, floor, Floor);
NUM_OP(floating_half_bfloat16, ceil, Ceil);

NUM_OP(floating_complex_half_bfloat16, acos, Acos);
NUM_OP(floating_complex_half_bfloat16, asin, Asin);
NUM_OP(floating_complex_half_bfloat16, atan, Atan);
NUM_OP(floating_complex_half_bfloat16, cosh, Cosh);
NUM_OP(floating_complex_half_bfloat16, tan, Tan);
NUM_OP(floating_complex_half_bfloat16, sin, Sin);
NUM_OP(floating_complex_half_bfloat16, sinh, Sinh);

NUM_OP(floating_complex_half_bfloat16, exp, Exp);
NUM_OP(floating_complex_half_bfloat16, tanh, Tanh);
NUM_OP(floating_complex_half_bfloat16, log, Log);
NUM_OP(floating_complex_half_bfloat16, log10, Log10);
NUM_OP(floating_complex_half_bfloat16, log2, Log2);
NUM_OP(floating_complex_half_bfloat16, cos, Cos);
NUM_OP(floating_complex_half_bfloat16, sqrt, Sqrt);

NUM_OP(floating_half_bfloat16, log1p, Log1p);
NUM_OP(floating_half_bfloat16, erf, Erf);

template <typename T>
struct Sigmoid {
  T one = T(1);
  T operator()(T t) const {
    return (one / (one + Numerics<T>::exp(-t)));
  }
};

template <typename T>
struct Round {
  T operator()(T t) const {
    return std::nearbyint(static_cast<float>(t));
  }
};

template <typename T>
struct Trunc {
  T operator()(T t) const {
    return t - Numerics<T>::trunc(t);
  }
};

template <typename T>
struct Reciprocal {
  T one = T(1);
  T operator()(T t) const {
    return (one / t);
  }
};

OP_CUSTOM_FUNCTOR(floating_half_bfloat16, sigmoid, Sigmoid)
OP_CUSTOM_FUNCTOR(floating_half_bfloat16, round, Round)
OP_CUSTOM_FUNCTOR(floating_half_bfloat16, frac, Trunc)
OP_CUSTOM_FUNCTOR(floating_complex_half_bfloat16, reciprocal, Reciprocal)

std::vector<Tensor> _foreach_neg(TensorList tensors) {
  at::native::check_foreach_api_restrictions(tensors);

  if (!at::native::can_use_fast_route(tensors)) {
    return at::native::foreach_tensor_neg_slow(tensors);
  }

  TORCH_CHECK(
      tensors[0].scalar_type() != kBool,
      "Negation, the `-` operator, on a bool tensor is not supported. "
      "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
  return all_types_half_complex_bfloat16<std::negate>(tensors);
}

void _foreach_neg_(TensorList tensors) {
  at::native::check_foreach_api_restrictions(tensors);

  if (!at::native::can_use_fast_route(tensors)) {
    return at::native::foreach_tensor_neg_slow_(tensors);
  }

  TORCH_CHECK(
      tensors[0].scalar_type() != kBool,
      "Negation, the `-` operator, on a bool tensor is not supported. "
      "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
  all_types_half_complex_bfloat16_<std::negate>(tensors);
}

template <typename T>
struct Abs {
  T operator()(T t) const {
    return Numerics<T>::abs(t);
  }
};

std::vector<Tensor> _foreach_abs(at::TensorList tensors) {
  at::native::check_foreach_api_restrictions(tensors);
  const bool has_complex = check_complex(tensors);
  if (!at::native::can_use_fast_route(tensors) || has_complex) {
    return at::native::foreach_tensor_abs_slow(tensors);
  }
  return all_types_complex_bfloat16_half_bool<Abs>(tensors);
}

void _foreach_abs_(at::TensorList tensors) {
  at::native::check_foreach_api_restrictions(tensors);
  const bool has_complex = check_complex(tensors);
  if (!at::native::can_use_fast_route(tensors) || has_complex) {
    return at::native::foreach_tensor_abs_slow_(tensors);
  }
  return all_types_complex_bfloat16_half_bool_<Abs>(tensors);
}

void _foreach_zero_(TensorList tensors) {
  at::native::check_foreach_api_restrictions(tensors);

  if (!at::native::can_use_fast_route(tensors)) {
    return at::native::foreach_tensor_zero_slow_(tensors);
  }
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(tensors.vec());

  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
      ScalarType::Half, tensors[0].scalar_type(), "foreach_zero_xpu_", [&]() {
        multi_tensor_apply<1>(
            tensor_lists,
            ZeroFunctor<
                scalar_t,
                /* depth */ 1,
                /* r_args_depth */ 1,
                /* res_arg_index */ 0>());
      });
}

} // namespace AtenIpexTypeXPU
} // namespace at