#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/TensorIterator.h>
#include <utils/DPCPP.h>
#include "comm/AccumulateType.h"
#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
struct HeavisideOutFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a == 0 ? b : static_cast<scalar_t>(a > 0);
  }
};

Tensor& heaviside_out(const Tensor& self, const Tensor& values, Tensor& out) {
  TORCH_CHECK(
      !self.is_complex() && !values.is_complex(),
      "heaviside is not yet implemented for complex tensors.");
  TORCH_CHECK(
      self.dtype() == values.dtype(),
      "heaviside is not yet implemented for tensors with different dtypes.");

  auto iter = TensorIterator::binary_op(out, self, values);
  IPEX_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBool, kBFloat16, iter.dtype(), "heaviside", [&]() {
        HeavisideOutFunctor<scalar_t> f;
        dpcpp_kernel_with_scalars(iter, f);
      });
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
