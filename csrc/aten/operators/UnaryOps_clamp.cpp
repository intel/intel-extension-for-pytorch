#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"

#include "Loops.h"
#include "comm/ATDispatch.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor& clamp_max_out(const Tensor& self, const Scalar& max, Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "clamp_max_out",
      [&]() {
        auto val = max.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t in) -> scalar_t {
          return Numerics<scalar_t>::isnan(in)
              ? in
              : Numerics<scalar_t>::gt(in, val) ? val : in;
        });
      });
  return out;
}

Tensor& clamp_min_out(const Tensor& self, const Scalar& min, Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "clamp_min_out",
      [&]() {
        auto val = min.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t in) -> scalar_t {
          return Numerics<scalar_t>::isnan(in)
              ? in
              : Numerics<scalar_t>::lt(in, val) ? val : in;
        });
      });
  return out;
}

Tensor& clamp_min_max(
    const Tensor& self,
    const Scalar& min,
    const Scalar& max,
    Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "clamp_min_max",
      [&]() {
        auto minValue = min.to<scalar_t>();
        auto maxValue = max.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t in) -> scalar_t {
          auto val = Numerics<scalar_t>::lt(in, maxValue) ? in : maxValue;
          return Numerics<scalar_t>::isnan(in)
              ? in
              : Numerics<scalar_t>::gt(minValue, val) ? minValue : val;
        });
      });
  return out;
}

Tensor& clamp_out(
    const Tensor& self,
    const optional<Scalar>& min,
    const optional<Scalar>& max,
    Tensor& result) {
  if (min && max) {
    at::AtenIpexTypeXPU::clamp_min_max(self, *min, *max, result);
  } else if (max) {
    at::AtenIpexTypeXPU::clamp_max_out(self, *max, result);
  } else if (min) {
    at::AtenIpexTypeXPU::clamp_min_out(self, *min, result);
  } else {
    TORCH_CHECK(false, "At least one of 'min' or 'max' must not be None");
  }
  return result;
}

Tensor& clamp_(Tensor& self, optional<Scalar> min, optional<Scalar> max) {
  return at::AtenIpexTypeXPU::clamp_out(self, min, max, self);
}

Tensor clamp(const Tensor& self, optional<Scalar> min, optional<Scalar> max) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeXPU::clamp_out(self, min, max, result);
}

} // namespace AtenIpexTypeXPU
} // namespace at
