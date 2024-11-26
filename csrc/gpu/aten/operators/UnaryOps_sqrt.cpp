#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include <oneDNN/oneDNN.h>
#include "Loops.h"
#include "LoopsTemplates.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
struct sqrt_kernel_xpu_functor {
  scalar_t operator()(scalar_t a) const {
    return std::sqrt(a);
  }
};

void sqrt_kernel_xpu(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "sqrt",
      [&]() {
        sqrt_kernel_xpu_functor<scalar_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
}

template <typename scalar_t>
struct rsqrt_kernel_xpu_functor {
  scalar_t operator()(scalar_t a) const {
    return Numerics<scalar_t>::rsqrt(a);
  }
};

void rsqrt_kernel_xpu(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "rsqrt",
      [&]() {
        rsqrt_kernel_xpu_functor<scalar_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
}

} // namespace AtenIpexTypeXPU
} // namespace at
