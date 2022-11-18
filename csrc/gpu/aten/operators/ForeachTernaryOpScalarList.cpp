#include <ATen/Dispatch.h>
#include <ATen/native/Fill.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/TensorIterator.h>
#include <aten/core/detail/IndexUtils.h>
#include "oneapi/dpl/functional"

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

template <template <class> class Op>
void foreach_pointwise_op_(
    TensorList input,
    TensorList tensors1,
    TensorList tensors2,
    at::ArrayRef<Scalar> scalars) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.reserve(3);
  tensor_lists.emplace_back(input.vec());
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op__sycl",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<3, opmath_t>(
            tensor_lists,
            scalars,
            PointWiseOpScalarListFunctor<
                scalar_t,
                /* depth */ 3,
                /* r_args_depth */ 3,
                /* res_arg_index */ 0>(),
            Op<opmath_t>());
      });
}

template <template <class> class Op>
std::vector<Tensor> foreach_pointwise_op(
    TensorList input,
    TensorList tensors1,
    TensorList tensors2,
    at::ArrayRef<Scalar> scalars) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.reserve(4);
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(input.size());
  for (const auto& t : input) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  tensor_lists.emplace_back(input.vec());
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());
  tensor_lists.emplace_back(std::move(vec_res));

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

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op_sycl",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<4, opmath_t>(
            tensor_lists,
            scalars,
            PointWiseOpScalarListFunctor<
                scalar_t,
                /* depth */ 4,
                /* r_args_depth */ 3,
                /* res_arg_index */ 3>(),
            Op<opmath_t>());
      });

  return tensor_lists[3];
}

#define FOREACH_POINTWISE_OP_SCALARLIST(NAME, OP)                         \
  std::vector<Tensor> _foreach_##NAME(                                    \
      TensorList input,                                                   \
      TensorList tensors1,                                                \
      TensorList tensors2,                                                \
      at::ArrayRef<Scalar> scalars) {                                     \
    at::native::check_foreach_api_restrictions(                           \
        input, tensors1, tensors2, scalars);                              \
                                                                          \
    if (!at::native::can_use_fast_route(                                  \
            {input, tensors1, tensors2}, scalars) ||                      \
        at::native::has_integral_tensor(input, /* includeBool */ true)) { \
      return at::native::foreach_tensor_##NAME##_scalarlist_slow(         \
          input, tensors1, tensors2, scalars);                            \
    }                                                                     \
                                                                          \
    return foreach_pointwise_op<OP>(input, tensors1, tensors2, scalars);  \
  }                                                                       \
                                                                          \
  void _foreach_##NAME##_(                                                \
      TensorList input,                                                   \
      TensorList tensors1,                                                \
      TensorList tensors2,                                                \
      at::ArrayRef<Scalar> scalars) {                                     \
    at::native::check_foreach_api_restrictions(                           \
        input, tensors1, tensors2, scalars);                              \
                                                                          \
    if (!at::native::can_use_fast_route(                                  \
            {input, tensors1, tensors2}, scalars) ||                      \
        at::native::has_integral_tensor(input, /* includeBool */ true)) { \
      return at::native::foreach_tensor_##NAME##_scalarlist_slow_(        \
          input, tensors1, tensors2, scalars);                            \
    }                                                                     \
                                                                          \
    foreach_pointwise_op_<OP>(input, tensors1, tensors2, scalars);        \
  }
FOREACH_POINTWISE_OP_SCALAR(addcmul, oneapi::dpl::multiplies);
FOREACH_POINTWISE_OP_SCALAR(addcdiv, oneapi::dpl::divides);
FOREACH_POINTWISE_OP_SCALARLIST(addcmul, oneapi::dpl::multiplies);
FOREACH_POINTWISE_OP_SCALARLIST(addcdiv, oneapi::dpl::divides);

} // namespace AtenIpexTypeXPU
} // namespace at