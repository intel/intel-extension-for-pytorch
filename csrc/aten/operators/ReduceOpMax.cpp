#include <ATen/Context.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/core/DimVector.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>

#include <c10/core/ScalarType.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "Reduce.h"
#include "ReduceOpsUtils.h"

using namespace xpu::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeXPU {

template <typename acc_t>
struct ReduceMaxOps {
  ReduceMaxOps() {}
  acc_t operator()(acc_t a, acc_t b) const {
    return (Numerics<acc_t>::gt(a, b) || Numerics<acc_t>::isnan(a)) ? a : b;
  }
};

template <
    typename scalar_t,
    typename acc_t = scalar_t,
    typename out_t = scalar_t>
void max_kernel_impl(TensorIterator& iter) {
  dpcpp_reduce_kernel<scalar_t, out_t>(
      iter,
      func_wrapper<scalar_t>(ReduceMaxOps<scalar_t>()),
      Numerics<scalar_t>::lower_bound());
}

static void max_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "max",
      [&]() { max_kernel_impl<scalar_t>(iter); });
}

Tensor& amax_out(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim) {
  TORCH_CHECK(
      self.scalar_type() == result.scalar_type(),
      "Illegal dtype for self, and out:",
      self.scalar_type(),
      result.scalar_type());
  if (self.numel() == 0) {
    zero_numel_check_dims(self, dim, "amax()");
  }
  auto iter = meta::make_reduction(
      "amax", result, self, dim, keepdim, self.scalar_type());
  if (iter.numel() != 0) {
    max_kernel(iter);
  }
  return result;
}

Tensor amax(const Tensor& self, IntArrayRef dim, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::amax_out(result, self, dim, keepdim);
}

Tensor max(const Tensor& self) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::amax_out(
      result, self, std::vector<int64_t>{}, false);
}

} // namespace AtenIpexTypeXPU
} // namespace at
