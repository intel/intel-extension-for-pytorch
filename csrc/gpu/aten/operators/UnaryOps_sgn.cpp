#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

namespace at {
namespace AtenIpexTypeXPU {

IPEX_UNARY_LOOPS_FUNC_ALL_ALL_COMPLEX(
    neg_out,
    Numerics<scalar_t>::neg,
    unary_op);

IPEX_UNARY_LOOPS_FUNC_COMPLEX(sgn_xpu, at::AtenIpexTypeXPU::sgn_impl, unary_op);

Tensor& sgn_out(const Tensor& self, Tensor& out) {
  if (self.is_complex()) {
    return at::AtenIpexTypeXPU::sgn_xpu(self, out);
  } else {
    return at::AtenIpexTypeXPU::sign_out(self, out);
  }
}

struct sign_out_functor {
  bool operator()(bool a) const {
    return a;
  }
};

template <typename scalar_t>
struct sign_out_functor_2 {
  scalar_t operator()(scalar_t a) const {
    scalar_t zero = scalar_t(0);
    return (zero < a) - (a < zero);
  }
};

Tensor& sign_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  if (iter.dtype() == ScalarType::Bool) {
    sign_out_functor f;
    dpcpp_kernel_for_tensor_iter(iter, f);
  } else {
    IPEX_DISPATCH_ALL_TYPES_AND2(
        ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "sign_xpu", [&] {
          sign_out_functor_2<scalar_t> f;
          dpcpp_kernel_for_tensor_iter(iter, f);
        });
  }
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
