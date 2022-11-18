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

#include <aten/operators/comm/Numerics.h>

#include <iostream>
#include "ForeachFunctors.h"
#include "Loops.h"
#include "MultiTensorApply.h"
#include "comm/Numerics.h"

namespace at {
namespace AtenIpexTypeXPU {

#define FOREACH_MAXIMUM_MINIMUM_OP(NAME, OP)                               \
  std::vector<Tensor> _foreach_##NAME(                                     \
      TensorList tensors1, TensorList tensors2) {                          \
    at::native::check_foreach_api_restrictions(tensors1, tensors2);        \
    if (!at::native::can_use_fast_route({tensors1, tensors2}) ||           \
        at::native::has_bool_tensor(tensors1)) {                           \
      return at::native::foreach_tensor_##NAME##_slow(tensors1, tensors2); \
    }                                                                      \
    std::vector<std::vector<Tensor>> tensor_lists;                         \
    std::vector<Tensor> vec_res;                                           \
    vec_res.reserve(tensors1.size());                                      \
    for (const auto& t : tensors1) {                                       \
      vec_res.emplace_back(empty_like(t));                                 \
    }                                                                      \
                                                                           \
    tensor_lists.emplace_back(tensors1.vec());                             \
    tensor_lists.emplace_back(tensors2.vec());                             \
    tensor_lists.emplace_back(vec_res);                                    \
    IPEX_DISPATCH_ALL_TYPES_AND2(                                          \
        ScalarType::Half,                                                  \
        ScalarType::BFloat16,                                              \
        tensors1[0].scalar_type(),                                         \
        "foreach_maximum_minimum_op_xpu",                                  \
        [&]() {                                                            \
          using opmath_t = at::opmath_type<scalar_t>;                      \
          auto op = [](opmath_t a, opmath_t b) -> opmath_t {               \
            opmath_t c = a OP b ? a : b;                                   \
            if (Numerics<opmath_t>::isnan(a)) {                            \
              c = a;                                                       \
            }                                                              \
            return c;                                                      \
          };                                                               \
          multi_tensor_apply<3>(                                           \
              tensor_lists,                                                \
              BinaryOpListAlphaFunctor<scalar_t, 3, 2, 2>(),               \
              op,                                                          \
              opmath_t(1));                                                \
        });                                                                \
    return tensor_lists[2];                                                \
  }                                                                        \
  void _foreach_##NAME##_(TensorList self, TensorList other) {             \
    at::native::check_foreach_api_restrictions(self, other);               \
    if (!at::native::can_use_fast_route({self, other}) ||                  \
        at::native::has_bool_tensor(self)) {                               \
      return at::native::foreach_tensor_##NAME##_slow_(self, other);       \
    }                                                                      \
                                                                           \
    AT_DISPATCH_ALL_TYPES_AND2(                                            \
        kHalf,                                                             \
        kBFloat16,                                                         \
        self[0].scalar_type(),                                             \
        "foreach_maximum_minimum_op_xpu_",                                 \
        [&]() {                                                            \
          using opmath_t = at::opmath_type<scalar_t>;                      \
          std::vector<std::vector<at::Tensor>> tensor_lists{               \
              self.vec(), other.vec()};                                    \
          auto op = [](opmath_t a, opmath_t b) -> opmath_t {               \
            opmath_t c = a OP b ? a : b;                                   \
            if (Numerics<opmath_t>::isnan(a)) {                            \
              c = a;                                                       \
            }                                                              \
            return c;                                                      \
          };                                                               \
          multi_tensor_apply<2>(                                           \
              tensor_lists,                                                \
              BinaryOpListAlphaFunctor<scalar_t, 2, 2, 0>(),               \
              op,                                                          \
              opmath_t(1));                                                \
        });                                                                \
  }

FOREACH_MAXIMUM_MINIMUM_OP(maximum, >);
FOREACH_MAXIMUM_MINIMUM_OP(minimum, <);

} // namespace AtenIpexTypeXPU
} // namespace at