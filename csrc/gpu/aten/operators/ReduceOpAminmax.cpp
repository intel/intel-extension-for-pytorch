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
#include "comm/XPUPair.h"

#include "Reduce.h"
#include "ReduceOpsUtils.h"

using namespace torch_ipex::xpu::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeXPU {

Tensor& argmax_out(
    const Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    Tensor& result) {
  if (dim.has_value()) {
    auto dim_ = maybe_wrap_dim(dim.value(), self.dim());
    zero_numel_check_dims(self, dim_, "argmax");
  } else {
    TORCH_CHECK_INDEX(
        self.numel() != 0,
        "argmax",
        ": Expected reduction dim to be specified for input.numel() == 0.");
  }
  Tensor in;
  if (dim) {
    in = self;
  } else {
    in = self.reshape({-1});
    keepdim = false;
  }

  Tensor ignored = at::empty({0}, self.options());
  return std::get<1>(
      at::max_out(ignored, result, in, dim.value_or(0), keepdim));
}

Tensor argmax(const Tensor& self, c10::optional<int64_t> dim, bool keepdims) {
  Tensor result = at::empty({0}, self.options().dtype(at::kLong));
  return at::AtenIpexTypeXPU::argmax_out(self, dim, keepdims, result);
}

Tensor& argmin_out(
    const Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    Tensor& result) {
  if (dim.has_value()) {
    auto dim_ = maybe_wrap_dim(dim.value(), self.dim());
    zero_numel_check_dims(self, dim_, "argmin");
  } else {
    TORCH_CHECK_INDEX(
        self.numel() != 0,
        "argmin",
        ": Expected reduction dim to be specified for input.numel() == 0.");
  }
  Tensor in;
  if (dim) {
    in = self;
  } else {
    in = self.reshape({-1});
    keepdim = false;
  }

  Tensor ignored = at::empty({0}, self.options());
  return std::get<1>(
      at::min_out(ignored, result, in, dim.value_or(0), keepdim));
}

Tensor argmin(const Tensor& self, c10::optional<int64_t> dim, bool keepdims) {
  Tensor result = at::empty({0}, self.options().dtype(at::kLong));
  return at::AtenIpexTypeXPU::argmin_out(self, dim, keepdims, result);
}
} // namespace AtenIpexTypeXPU
} // namespace at
