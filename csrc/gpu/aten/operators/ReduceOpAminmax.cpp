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

template <typename scalar_t, typename acc_scalar_t, typename index_t>
struct MinMaxOps {
  using acc_t = std::pair<acc_scalar_t, acc_scalar_t>;
  inline acc_t reduce(acc_t acc, scalar_t data, int64_t idx) const {
    return combine(acc, {data, data});
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    auto min_val = (Numerics<acc_scalar_t>::isnan(a.first) || a.first < b.first)
        ? a.first
        : b.first;
    auto max_val =
        (Numerics<acc_scalar_t>::isnan(a.second) || a.second > b.second)
        ? a.second
        : b.second;

    return {min_val, max_val};
  }

  inline acc_t project(acc_t acc) const {
    return acc;
  }

  inline acc_t sg_shfl_down(acc_t arg, int offset) const {
    // FIXME:
    return arg;
  }
  inline acc_t translate_idx(acc_t acc, int64_t /*idx*/) const {
    return acc;
  }
};

template <typename scalar_t>
void _min_max_values_kernel_dpcpp_impl(TensorIterator& iter) {
  dpcpp_reduce_kernel<scalar_t, scalar_t>(
      iter,
      MinMaxOps<scalar_t, scalar_t, int32_t>{},
      std::pair<scalar_t, scalar_t>(
          std::numeric_limits<scalar_t>::max(),
          std::numeric_limits<scalar_t>::lowest()));
}

void aminmax_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "aminmax_elementwise_dpcpp",
      [&]() { _min_max_values_kernel_dpcpp_impl<scalar_t>(iter); });
}

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

void aminmax_out(const Tensor& self, Tensor& min_result, Tensor& max_result) {
  auto iter = meta::make_reduction(
      "aminmax",
      min_result,
      max_result,
      self,
      std::vector<int64_t>{},
      false,
      self.scalar_type()); // TensorIterator::binary_op(min_result, max_result,
                           // self);
  aminmax_kernel(iter);
}

void aminmax_dim_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& min_result,
    Tensor& max_result) {
  auto iter = meta::make_reduction(
      "aminmax_dim",
      min_result,
      max_result,
      self,
      dim,
      keepdim,
      self.scalar_type()); // TensorIterator::binary_op(min_result, max_result,
                           // self);
  aminmax_kernel(iter);
}

std::tuple<Tensor, Tensor> _aminmax(const Tensor& self) {
  TORCH_CHECK(
      !self.is_complex(), "max is not yet implemented for complex tensors.");
  TORCH_CHECK(self.numel() > 0, "operation does not have an identity.");
  Tensor min_result;
  Tensor max_result;
  at::AtenIpexTypeXPU::aminmax_out(self, min_result, max_result);
  return std::tuple<Tensor, Tensor>(min_result, max_result);
}

std::tuple<Tensor, Tensor> _aminmax(
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  TORCH_CHECK(
      !self.is_complex(), "max is not yet implemented for complex tensors.");
  TORCH_CHECK(self.numel() > 0, "operation does not have an identity.");
  Tensor min_result = at::empty_like(self);
  Tensor max_result = at::empty_like(self);
  at::AtenIpexTypeXPU::aminmax_dim_out(
      self, dim, keepdim, min_result, max_result);
  return std::tuple<Tensor, Tensor>(min_result, max_result);
}

::std::tuple<at::Tensor&, at::Tensor&> aminmax_out(
    const at::Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    at::Tensor& min,
    at::Tensor& max) {
  Tensor in;
  if (dim.has_value()) {
    in = self;
  } else {
    in = self.reshape({-1});
    keepdim = false;
  }
  at::AtenIpexTypeXPU::aminmax_dim_out(in, dim.value_or(0), keepdim, min, max);
  return std::tuple<Tensor&, Tensor&>(min, max);
}

} // namespace AtenIpexTypeXPU
} // namespace at
