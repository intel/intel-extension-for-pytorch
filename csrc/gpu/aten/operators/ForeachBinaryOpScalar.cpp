#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
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
namespace AtenIpexTypeXPU {

template <typename scalar_t, template <class> class Op>
std::vector<Tensor> foreach_binary_op(
    TensorList tensors,
    const Scalar& scalar) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors.size());
  for (const auto& t : tensors) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  tensor_lists.emplace_back(tensors.vec());
  tensor_lists.emplace_back(std::move(vec_res));

  using opmath_t = at::opmath_type<scalar_t>;
  multi_tensor_apply<2>(
      tensor_lists,
      BinaryOpScalarFunctor<
          scalar_t,
          /* depth */ 2,
          /* r_args_depth */ 1,
          /* res_arg_index */ 1>(),
      Op<opmath_t>(),
      scalar.to<opmath_t>());
  return tensor_lists[1];
}

template <typename scalar_t, template <class> class Op>
void foreach_binary_op_(TensorList tensors, const Scalar& scalar) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(tensors.vec());
  using opmath_t = at::opmath_type<scalar_t>;
  multi_tensor_apply<1>(
      tensor_lists,
      BinaryOpScalarFunctor<
          scalar_t,
          /* depth */ 1,
          /* r_args_depth */ 1,
          /* res_arg_index */ 0>(),
      Op<opmath_t>(),
      scalar.to<opmath_t>());
}

template <template <class> class Op>
std::vector<Tensor> all_types_complex_bool_half_bfloat16(
    TensorList tensors,
    const Scalar& scalar) {
  return IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda",
      [&]() { return foreach_binary_op<scalar_t, Op>(tensors, scalar); });
}

template <template <class> class Op>
void all_types_complex_bool_half_bfloat16_(
    TensorList tensors,
    const Scalar& scalar) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalar); });
}

template <template <class> class Op>
std::vector<Tensor> all_types_half_bfloat16(
    TensorList tensors,
    const Scalar& scalar) {
  return IPEX_DISPATCH_ALL_TYPES_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda",
      [&]() { return foreach_binary_op<scalar_t, Op>(tensors, scalar); });
}

template <template <class> class Op>
void all_types_half_bfloat16_(TensorList tensors, const Scalar& scalar) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalar); });
}

#define FOREACH_BINARY_OP_SCALAR(FUNCTION, NAME, OP, DIVISION_OP)        \
  void _foreach_##NAME##_(TensorList tensors, const Scalar& scalar) {    \
    at::native::check_foreach_api_restrictions(tensors);                 \
    if (!at::native::can_use_fast_route(tensors, scalar, DIVISION_OP)) { \
      return at::native::foreach_tensor_##NAME##_scalar_kernel_slow_(    \
          tensors, scalar);                                              \
    }                                                                    \
                                                                         \
    FUNCTION##_<OP>(tensors, scalar);                                    \
  }                                                                      \
                                                                         \
  std::vector<Tensor> _foreach_##NAME(                                   \
      TensorList tensors, const at::Scalar& scalar) {                    \
    at::native::check_foreach_api_restrictions(tensors);                 \
    if (!at::native::can_use_fast_route(tensors, scalar, DIVISION_OP)) { \
      return at::native::foreach_tensor_##NAME##_scalar_kernel_slow(     \
          tensors, scalar);                                              \
    }                                                                    \
                                                                         \
    return FUNCTION<OP>(tensors, scalar);                                \
  }

FOREACH_BINARY_OP_SCALAR(
    all_types_complex_bool_half_bfloat16,
    add,
    std::plus,
    false);
FOREACH_BINARY_OP_SCALAR(
    all_types_complex_bool_half_bfloat16,
    mul,
    std::multiplies,
    false);

// In the case of division, integer inputs will result in float.
// Currently multi tensor apply can only return result of the same type as
// input.
FOREACH_BINARY_OP_SCALAR(
    all_types_complex_bool_half_bfloat16,
    div,
    std::divides,
    true);

// In the case of subtraction, we dont allow scalar to be boolean following the
// torch.sub logic
void _foreach_sub_(TensorList tensors, const Scalar& scalar) {
  at::native::check_foreach_api_restrictions(tensors);
  at::native::sub_check(tensors[0], scalar);

  if (!at::native::can_use_fast_route(tensors, scalar)) {
    return at::native::foreach_tensor_sub_scalar_kernel_slow_(tensors, scalar);
  }

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_",
      [&]() { foreach_binary_op_<scalar_t, std::minus>(tensors, scalar); });
}

std::vector<Tensor> _foreach_sub(TensorList tensors, const Scalar& scalar) {
  at::native::check_foreach_api_restrictions(tensors);
  at::native::sub_check(tensors[0], scalar);

  if (!at::native::can_use_fast_route(tensors, scalar)) {
    return at::native::foreach_tensor_sub_scalar_kernel_slow(tensors, scalar);
  }

  return IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar",
      [&]() {
        return foreach_binary_op<scalar_t, std::minus>(tensors, scalar);
      });
}

FOREACH_BINARY_OP_SCALAR(
    all_types_half_bfloat16,
    clamp_max,
    foreach_internal::maximum,
    false);
FOREACH_BINARY_OP_SCALAR(
    all_types_half_bfloat16,
    clamp_min,
    foreach_internal::minimum,
    false);

std::vector<Tensor> _foreach_minimum(TensorList tensors, const Scalar& scalar) {
  return AtenIpexTypeXPU::_foreach_clamp_min(tensors, scalar);
}
void _foreach_minimum_(TensorList tensors, const Scalar& scalar) {
  AtenIpexTypeXPU::_foreach_clamp_min_(tensors, scalar);
}
std::vector<Tensor> _foreach_maximum(TensorList tensors, const Scalar& scalar) {
  return AtenIpexTypeXPU::_foreach_clamp_max(tensors, scalar);
}
void _foreach_maximum_(TensorList tensors, const Scalar& scalar) {
  AtenIpexTypeXPU::_foreach_clamp_max_(tensors, scalar);
}

} // namespace AtenIpexTypeXPU
} // namespace at
