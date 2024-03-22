#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>
#include "comm/RegistrationDeclarations.h"
#include "comm/ScalarOps.h"

#include "Loops.h"
#include "comm/zmath.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
struct DivTruncKernelDpcppFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a / b;
  }
};

template <typename scalar_t>
struct DivTruncKernelDpcppFunctor2 {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return trunc_impl(a / b);
  }
};

static void div_trunc_kernel_dpcpp(TensorIterator& iter) {
  if (isIntegralType(iter.dtype(), false)) {
    IPEX_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "div_trunc_dpcpp", [&] {
      DivTruncKernelDpcppFunctor<scalar_t> f;
      dpcpp_kernel_with_scalars(iter, f);
    });
  } else {
    IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        iter.dtype(),
        "div_trunc_dpcpp",
        [&]() {
          DivTruncKernelDpcppFunctor2<scalar_t> f;
          dpcpp_kernel_with_scalars(iter, f);
        });
  }
}

} // namespace impl

void div_trunc_kernel(TensorIterator& iter) {
  impl::div_trunc_kernel_dpcpp(iter);
}

} // namespace AtenIpexTypeXPU
} // namespace at
