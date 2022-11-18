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

namespace AtenIpexTypeXPU {

template <template <class> class Op>
std::vector<Tensor> foreach_tensor_list_op(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());
  tensor_lists.emplace_back(std::move(vec_res));

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kBFloat16,
      kHalf,
      tensors1[0].scalar_type(),
      "foreach_binary_op_list",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<3>(
            tensor_lists,
            BinaryOpListAlphaFunctor<
                scalar_t,
                /* depth */ 3,
                /* r_args_depth */ 2,
                /* res_arg_index */ 2>(),
            Op<opmath_t>(),
            alpha.to<opmath_t>());
      });

  return tensor_lists[2];
}

template <template <class> class Op>
void foreach_tensor_list_op_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kBFloat16,
      kHalf,
      tensors1[0].scalar_type(),
      "foreach_binary_op_list_",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<2>(
            tensor_lists,
            BinaryOpListAlphaFunctor<
                scalar_t,
                /* depth */ 2,
                /* r_args_depth */ 2,
                /* res_arg_index */ 0>(),
            Op<opmath_t>(),
            alpha.to<opmath_t>());
      });
}

#define FOREACH_BINARY_OP_LIST(NAME, OP, DIVISION_OP)                       \
  void _foreach_##NAME##_(TensorList tensors1, TensorList tensors2) {       \
    at::native::check_foreach_api_restrictions(tensors1, tensors2);         \
    if (!at::native::can_use_fast_route(tensors1, tensors2, DIVISION_OP)) { \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow_(         \
          tensors1, tensors2);                                              \
    }                                                                       \
                                                                            \
    foreach_tensor_list_op_<OP>(tensors1, tensors2);                        \
  }                                                                         \
                                                                            \
  std::vector<Tensor> _foreach_##NAME(                                      \
      TensorList tensors1, TensorList tensors2) {                           \
    at::native::check_foreach_api_restrictions(tensors1, tensors2);         \
    if (!at::native::can_use_fast_route(tensors1, tensors2, DIVISION_OP)) { \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow(          \
          tensors1, tensors2);                                              \
    }                                                                       \
                                                                            \
    return foreach_tensor_list_op<OP>(tensors1, tensors2);                  \
  }

#define FOREACH_BINARY_OP_LIST_ALPHA(NAME, OP)                          \
  void _foreach_##NAME##_(                                              \
      TensorList tensors1, TensorList tensors2, const Scalar& alpha) {  \
    at::native::check_foreach_api_restrictions(tensors1, tensors2);     \
    if (!at::native::can_use_fast_route({tensors1, tensors2}, alpha)) { \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow_(     \
          tensors1, tensors2, alpha);                                   \
    }                                                                   \
                                                                        \
    foreach_tensor_list_op_<OP>(tensors1, tensors2, alpha);             \
  }                                                                     \
                                                                        \
  std::vector<Tensor> _foreach_##NAME(                                  \
      TensorList tensors1, TensorList tensors2, const Scalar& alpha) {  \
    at::native::check_foreach_api_restrictions(tensors1, tensors2);     \
    if (!at::native::can_use_fast_route({tensors1, tensors2}, alpha)) { \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow(      \
          tensors1, tensors2, alpha);                                   \
    }                                                                   \
                                                                        \
    return foreach_tensor_list_op<OP>(tensors1, tensors2, alpha);       \
  }

FOREACH_BINARY_OP_LIST_ALPHA(add, std::plus);
FOREACH_BINARY_OP_LIST_ALPHA(sub, std::minus);
FOREACH_BINARY_OP_LIST(mul, std::multiplies, /*division_op*/ false);
FOREACH_BINARY_OP_LIST(div, std::divides, /*division_op*/ true);

} // namespace AtenIpexTypeXPU
} // namespace at
