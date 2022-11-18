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

template <template <class> class Op>
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

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBFloat16,
      kHalf,
      kBool,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalarlist",
      [&]() {
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
      });
  return tensor_lists[1];
}

template <template <class> class Op>
void foreach_binary_op_(TensorList tensors, at::ArrayRef<Scalar> scalars) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(tensors.vec());

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBFloat16,
      kHalf,
      kBool,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalarlist_",
      [&]() {
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
      });
}

#define FOREACH_BINARY_OP_SCALARLIST(NAME, OP, DIV_OP)                        \
  void _foreach_##NAME##_(TensorList tensors, at::ArrayRef<Scalar> scalars) { \
    at::native::check_foreach_api_restrictions(tensors, scalars);             \
    if (!at::native::can_use_fast_route(tensors, scalars, DIV_OP)) {          \
      return at::native::foreach_tensor_##NAME##_scalarlist_kernel_slow_(     \
          tensors, scalars);                                                  \
    }                                                                         \
                                                                              \
    foreach_binary_op_<OP>(tensors, scalars);                                 \
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
    return foreach_binary_op<OP>(tensors, scalars);                           \
  }

FOREACH_BINARY_OP_SCALARLIST(add, std::plus, /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST(mul, std::multiplies, /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST(div, std::divides, /*div_op*/ true);

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

  foreach_binary_op_<std::minus>(tensors, scalars);
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

  return foreach_binary_op<std::minus>(tensors, scalars);
}

} // namespace AtenIpexTypeXPU
} // namespace at
