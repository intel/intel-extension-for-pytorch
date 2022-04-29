#include <ATen/ATen.h>
#include "comm/AccumulateType.h"

#include <core/Memory.h>
#include <runtime/Utils.h>
#include "comm/ApplyUtils.h"
#include "comm/RegistrationDeclarations.h"

#include "comm/ATDispatch.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void where_kernel(TensorIterator& iter, ScalarType condition_type) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, iter.dtype(), "where_dpcpp", [&] {
        if (condition_type == at::ScalarType::Byte) {
          dpcpp_kernel_for_tensor_iter(
              iter,
              [=](uint8_t cond_val, scalar_t self_val, scalar_t other_val)
                  -> scalar_t { return cond_val ? self_val : other_val; });
        } else {
          dpcpp_kernel_for_tensor_iter(
              iter,
              [=](bool cond_val, scalar_t self_val, scalar_t other_val)
                  -> scalar_t { return cond_val ? self_val : other_val; });
        }
      });
}

} // namespace impl

Tensor _s_where(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  TORCH_CHECK(
      self.dtype() == other.dtype(),
      "expected scalar type ",
      self.dtype(),
      " but found ",
      other.dtype());
  Tensor ret = at::empty(self.sizes(), self.options());
  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(ret)
                  .add_input(condition)
                  .add_input(self)
                  .add_input(other)
                  .build();
  impl::where_kernel(iter, condition.scalar_type());
  return ret;
}

Tensor isnan(const Tensor& self) {
  return self != self;
}

} // namespace AtenIpexTypeXPU
} // namespace at
