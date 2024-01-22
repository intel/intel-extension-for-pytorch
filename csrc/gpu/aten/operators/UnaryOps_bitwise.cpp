#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
struct bitwise_not_kernel_dpcpp_functor {
  scalar_t operator()(scalar_t a) const {
    return ~a;
  }
};

struct bitwise_not_kernel_dpcpp_functor_2 {
  bool operator()(bool a) const {
    return !a;
  }
};

void bitwise_not_kernel_dpcpp(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    bitwise_not_kernel_dpcpp_functor_2 f;
    dpcpp_kernel_for_tensor_iter(iter, f);
  } else {
    IPEX_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_not_dpcpp", [&]() {
      bitwise_not_kernel_dpcpp_functor<scalar_t> f;
      dpcpp_kernel_for_tensor_iter(iter, f);
    });
  }
}

} // namespace impl

Tensor& bitwise_not_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  impl::bitwise_not_kernel_dpcpp(iter);
#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names(out, self);
#endif
  return out;
}

Tensor bitwise_not(const Tensor& self) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::bitwise_not_out(self, result);
}

Tensor& bitwise_not_(Tensor& self) {
  return at::AtenIpexTypeXPU::bitwise_not_out(self, self);
}

} // namespace AtenIpexTypeXPU
} // namespace at
