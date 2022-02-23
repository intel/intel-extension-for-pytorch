#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void bitwise_not_kernel_dpcpp(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    dpcpp_kernel_for_tensor_iter(iter, [](bool a) -> bool { return !a; });
  } else {
    IPEX_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_not_dpcpp", [&]() {
      dpcpp_kernel_for_tensor_iter(
          iter, [](scalar_t a) -> scalar_t { return ~a; });
    });
  }
}

} // namespace impl

Tensor& bitwise_not_out(Tensor& out, const Tensor& self) {
  auto iter = TensorIterator::unary_op(out, self);
  impl::bitwise_not_kernel_dpcpp(iter);
#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names(out, self);
#endif
  return out;
}

Tensor bitwise_not(const Tensor& self) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::bitwise_not_out(result, self);
}

Tensor& bitwise_not_(Tensor& self) {
  return at::AtenIpexTypeXPU::bitwise_not_out(self, self);
}

} // namespace AtenIpexTypeXPU
} // namespace at
