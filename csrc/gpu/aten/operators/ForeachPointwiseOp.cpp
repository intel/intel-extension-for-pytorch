#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Fill.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/TensorIterator.h>
#include <aten/core/detail/IndexUtils.h>
#include <aten/operators/comm/Numerics.h>
#include <runtime/Utils.h>
#include "ATen/OpMathType.h"
#include "comm/ATDispatch.h"
#include "comm/ApplyUtils.h"
#include "comm/RegistrationDeclarations.h"
#include "utils/CustomOperatorRegistration.h"

#include <iostream>
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
    at::ArrayRef<Scalar> scalars) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.reserve(4);
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

std::vector<c10::Scalar> convert_tensor_to_scalar_list(
    const Tensor& scalarList_,
    int64_t expect_length) {
  std::vector<c10::Scalar> scalarList;
  TORCH_CHECK(
      scalarList_.device() == c10::kCPU,
      "Expected scalars to be on CPU, got ",
      scalarList_.device(),
      " instead.");
  TORCH_CHECK(
      scalarList_.is_contiguous(), "Expected scalars to be contiguous.");
  TORCH_CHECK(
      scalarList_.dim() == 1,
      "Expected packed scalar Tensor to be of dimension 1. Got ",
      scalarList_.dim(),
      " instead.");
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      kComplexHalf,
      kHalf,
      kBool,
      kBFloat16,
      scalarList_.scalar_type(),
      "convert_tensor_to_scalar_list",
      [&]() {
        const scalar_t* scalar_data = scalarList_.data_ptr<scalar_t>();
        TORCH_CHECK(
            (expect_length == scalarList_.size(0)),
            "Expected length of scalars to match input of length ",
            expect_length,
            " but got ",
            scalarList_.size(0),
            " instead.");
        for (int64_t i = 0; i < scalarList_.size(0); i++) {
          scalarList.push_back(c10::Scalar(scalar_data[i]));
        }
      });
  return scalarList;
}

#define FOREACH_POINTWISE_OP_TENSOR(NAME, OP)                             \
  std::vector<Tensor> _foreach_##NAME(                                    \
      TensorList input,                                                   \
      TensorList tensors1,                                                \
      TensorList tensors2,                                                \
      const Tensor& scalars_) {                                           \
    auto scalars = convert_tensor_to_scalar_list(scalars_, input.size()); \
    at::native::check_foreach_api_restrictions(                           \
        input, tensors1, tensors2, scalars);                              \
    if (!at::native::can_use_fast_route({input, tensors1, tensors2}) ||   \
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
      const Tensor& scalars_) {                                           \
    auto scalars = convert_tensor_to_scalar_list(scalars_, input.size()); \
    at::native::check_foreach_api_restrictions(                           \
        input, tensors1, tensors2, scalars);                              \
    if (!at::native::can_use_fast_route(                                  \
            {input, tensors1, tensors2}, scalars) ||                      \
        at::native::has_integral_tensor(input, /* includeBool */ true)) { \
      return at::native::foreach_tensor_##NAME##_scalarlist_slow_(        \
          input, tensors1, tensors2, scalars);                            \
    }                                                                     \
                                                                          \
    foreach_pointwise_op_<OP>(input, tensors1, tensors2, scalars);        \
  }

FOREACH_POINTWISE_OP_TENSOR(addcdiv, std::divides);
FOREACH_POINTWISE_OP_TENSOR(addcmul, std::multiplies);

} // namespace AtenIpexTypeXPU
} // namespace at