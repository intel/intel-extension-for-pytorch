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

#include <ATen/native/BinaryOps.h>
#include "ForeachFunctors.h"
#include "Loops.h"
#include "MultiTensorApply.h"
#include "comm/Numerics.h"
namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t, template <class> class Op>
std::vector<Tensor> foreach_binary_op(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors.size());
  for (const auto& t : tensors) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  tensor_lists.emplace_back(tensors.vec());
  tensor_lists.emplace_back(vec_res);
  using opmath_t = at::opmath_type<scalar_t>;
  multi_tensor_apply<2, opmath_t>(
      tensor_lists,
      scalars,
      BinaryOpScalarListFunctor<
          scalar_t,
          /* depth */ 2,
          /* r_args_depth */ 1,
          /* res_arg_index */ 1>(),

      Op<opmath_t>());

  return tensor_lists[1];
}

template <typename scalar_t, template <class> class Op>
void foreach_binary_op_(TensorList tensors, at::ArrayRef<Scalar> scalars) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(tensors.vec());

  using opmath_t = at::opmath_type<scalar_t>;
  multi_tensor_apply<1, opmath_t>(
      tensor_lists,
      scalars,
      BinaryOpScalarListFunctor<
          scalar_t,
          /* depth */ 1,
          /* r_args_depth */ 1,
          /* res_arg_index */ 0>(),
      Op<opmath_t>());
}

template <template <class> class Op>
std::vector<Tensor> all_types_complex_bool_half_bfloat16(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  return IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda",
      [&]() { return foreach_binary_op<scalar_t, Op>(tensors, scalars); });
}

template <template <class> class Op>
void all_types_complex_bool_half_bfloat16_(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalars); });
}

template <template <class> class Op>
std::vector<Tensor> all_types_half_bfloat16(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  return IPEX_DISPATCH_ALL_TYPES_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda",
      [&]() { return foreach_binary_op<scalar_t, Op>(tensors, scalars); });
}

template <template <class> class Op>
void all_types_half_bfloat16_(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalars); });
}

#define FOREACH_BINARY_OP_SCALARLIST(FUNCTION, NAME, OP, DIV_OP)              \
  void _foreach_##NAME##_(TensorList tensors, at::ArrayRef<Scalar> scalars) { \
    at::native::check_foreach_api_restrictions(tensors, scalars);             \
    if (!at::native::can_use_fast_route(tensors, scalars, DIV_OP)) {          \
      return at::native::foreach_tensor_##NAME##_scalarlist_kernel_slow_(     \
          tensors, scalars);                                                  \
    }                                                                         \
                                                                              \
    FUNCTION##_<OP>(tensors, scalars);                                        \
  }                                                                           \
                                                                              \
  std::vector<Tensor> _foreach_##NAME(                                        \
      TensorList tensors, at::ArrayRef<Scalar> scalars) {                     \
    at::native::check_foreach_api_restrictions(tensors, scalars);             \
    if (!at::native::can_use_fast_route(tensors, scalars, DIV_OP)) {          \
      return at::native::foreach_tensor_##NAME##_scalarlist_kernel_slow(      \
          tensors, scalars);                                                  \
    }                                                                         \
                                                                              \
    return FUNCTION<OP>(tensors, scalars);                                    \
  }

FOREACH_BINARY_OP_SCALARLIST(
    all_types_complex_bool_half_bfloat16,
    add,
    std::plus,
    /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST(
    all_types_complex_bool_half_bfloat16,
    mul,
    std::multiplies,
    /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST(
    all_types_complex_bool_half_bfloat16,
    div,
    std::divides,
    /*div_op*/ true);

// This does not use FOREACH_BINARY_OP_SCALARLIST because
// In the case of subtraction, we dont allow scalar to be boolean following the
// torch.sub logic
void _foreach_sub_(TensorList tensors, at::ArrayRef<Scalar> scalars) {
  at::native::check_foreach_api_restrictions(tensors, scalars);
  for (int i = 0; i < tensors.size(); i++) {
    at::native::sub_check(tensors[i], scalars[i]);
  }

  if (!at::native::can_use_fast_route({tensors}, scalars)) {
    return at::native::foreach_tensor_sub_scalarlist_kernel_slow_(
        tensors, scalars);
  }

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda_",
      [&]() { foreach_binary_op_<scalar_t, std::minus>(tensors, scalars); });
}

std::vector<Tensor> _foreach_sub(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  at::native::check_foreach_api_restrictions(tensors, scalars);
  for (int i = 0; i < tensors.size(); i++) {
    at::native::sub_check(tensors[i], scalars[i]);
  }

  if (!at::native::can_use_fast_route({tensors}, scalars)) {
    return at::native::foreach_tensor_sub_scalarlist_kernel_slow(
        tensors, scalars);
  }

  return IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda",
      [&]() {
        return foreach_binary_op<scalar_t, std::minus>(tensors, scalars);
      });
}

FOREACH_BINARY_OP_SCALARLIST(
    all_types_half_bfloat16,
    clamp_max,
    foreach_internal::maximum,
    false);
FOREACH_BINARY_OP_SCALARLIST(
    all_types_half_bfloat16,
    clamp_min,
    foreach_internal::minimum,
    false);

std::vector<Tensor> _foreach_minimum(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  return AtenIpexTypeXPU::_foreach_clamp_min(tensors, scalars);
}
void _foreach_minimum_(TensorList tensors, at::ArrayRef<Scalar> scalars) {
  AtenIpexTypeXPU::_foreach_clamp_min_(tensors, scalars);
}
std::vector<Tensor> _foreach_maximum(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  return AtenIpexTypeXPU::_foreach_clamp_max(tensors, scalars);
}
void _foreach_maximum_(TensorList tensors, at::ArrayRef<Scalar> scalars) {
  AtenIpexTypeXPU::_foreach_clamp_max_(tensors, scalars);
}

} // namespace AtenIpexTypeXPU
} // namespace at
