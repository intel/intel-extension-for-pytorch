#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/AccumulateType.h"
#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/zmath.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void conj_physical_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Bool,
      ScalarType::BFloat16,
      ScalarType::Half,
      iter.common_dtype(),
      "conj",
      [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          return at::AtenIpexTypeXPU::conj_impl(a);
        });
      });
}

} // namespace impl

Tensor& conj_physical_out(const Tensor& self, Tensor& result) {
  auto iter = TensorIterator::unary_op(result, self);
  impl::conj_physical_kernel(iter);
  return result;
}

Tensor& conj_physical_(Tensor& self) {
  if (!self.is_complex())
    return self;
  return at::AtenIpexTypeXPU::conj_physical_out(self, self);
}

} // namespace AtenIpexTypeXPU
} // namespace at
