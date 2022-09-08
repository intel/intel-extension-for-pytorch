#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/TensorIterator.h>
#include <aten/core/detail/IndexUtils.h>

#include <runtime/Utils.h>
#include "ATen/OpMathType.h"
#include "comm/ATDispatch.h"
#include "comm/ApplyUtils.h"
#include "comm/RegistrationDeclarations.h"

#include <aten/operators/comm/Numerics.h>

#include "ForeachFunctors.h"
#include "Loops.h"
#include "MultiTensorApply.h"
#include "comm/Numerics.h"

namespace at {
namespace AtenIpexTypeXPU {

template <template <class> class Op>
std::vector<Tensor> foreach_pointwise_op(
    TensorList input,
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& scalar) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(input.size());
  for (const auto& t : input) {
    vec_res.emplace_back(at::empty_like(t));
  }

  tensor_lists.emplace_back(input.vec());
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());
  tensor_lists.emplace_back(std::move(vec_res));

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op_xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<4>(
            tensor_lists,
            at::AtenIpexTypeXPU::PointWiseOpScalarFunctor<scalar_t, 4, 3, 3>(),
            Op<opmath_t>(),
            scalar.to<opmath_t>());
      });

  return tensor_lists[3];
}

template <template <class> class Op>
void foreach_pointwise_op_(
    TensorList input,
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& scalar) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(input.vec());
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op__xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<3>(
            tensor_lists,
            at::AtenIpexTypeXPU::PointWiseOpScalarFunctor<scalar_t, 3, 3, 0>(),
            Op<opmath_t>(),
            scalar.to<opmath_t>());
      });
}

#define FOREACH_POINTWISE_OP_SCALAR(NAME, OP)                              \
  std::vector<Tensor> _foreach_##NAME(                                     \
      TensorList input,                                                    \
      TensorList tensors1,                                                 \
      TensorList tensors2,                                                 \
      const Scalar& scalar) {                                              \
    at::native::check_foreach_api_restrictions(input, tensors1, tensors2); \
                                                                           \
    if (!at::native::can_use_fast_route(                                   \
            {input, tensors1, tensors2}, scalar) ||                        \
        at::native::has_integral_tensor(input, /* includeBool */ true)) {  \
      return at::native::foreach_tensor_##NAME##_scalar_slow(              \
          input, tensors1, tensors2, scalar);                              \
    }                                                                      \
                                                                           \
    return foreach_pointwise_op<OP>(input, tensors1, tensors2, scalar);    \
  }                                                                        \
                                                                           \
  void _foreach_##NAME##_(                                                 \
      TensorList input,                                                    \
      TensorList tensors1,                                                 \
      TensorList tensors2,                                                 \
      const Scalar& scalar) {                                              \
    at::native::check_foreach_api_restrictions(input, tensors1, tensors2); \
                                                                           \
    if (!at::native::can_use_fast_route(                                   \
            {input, tensors1, tensors2}, scalar) ||                        \
        at::native::has_integral_tensor(input, /* includeBool */ true)) {  \
      return at::native::foreach_tensor_##NAME##_scalar_slow_(             \
          input, tensors1, tensors2, scalar);                              \
    }                                                                      \
                                                                           \
    foreach_pointwise_op_<OP>(input, tensors1, tensors2, scalar);          \
  }

FOREACH_POINTWISE_OP_SCALAR(addcmul, std::multiplies);
FOREACH_POINTWISE_OP_SCALAR(addcdiv, std::divides);

} // namespace AtenIpexTypeXPU
} // namespace at