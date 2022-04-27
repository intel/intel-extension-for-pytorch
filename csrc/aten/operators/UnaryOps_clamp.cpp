#include <ATen/ATen.h>
#include <ATen/NumericUtils.h>
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

Tensor& clamp_out(
    const Tensor& self,
    const c10::optional<Tensor>& min,
    const c10::optional<Tensor>& max,
    Tensor& result) {
  if (min && max) {
    TORCH_CHECK(
        self.layout() == Layout::Strided,
        "torch.clamp only supports strided layout, got: ",
        self.layout());
    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(true)
                    .add_output(result)
                    .add_input(self)
                    .add_input(*min)
                    .add_input(*max)
                    .promote_inputs_to_common_dtype(true)
                    .cast_common_dtype_to_outputs(true)
                    .enforce_safe_casting_to_output(true)
                    .build();

    IPEX_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "clamp_min_max",
        [&]() {
          dpcpp_kernel_for_tensor_iter(
              iter,
              [=](scalar_t in, scalar_t min_val, scalar_t max_val) -> scalar_t {
                if (at::_isnan(in)) {
                  return in;
                } else {
                  return std::min(std::max(in, min_val), max_val);
                }
              });
        });
  } else if (max) {
    at::clamp_max_outf(self, *max, result);
  } else if (min) {
    at::clamp_min_outf(self, *min, result);
  } else {
    TORCH_CHECK(
        false, "torch.clamp: At least one of 'min' or 'max' must not be None");
  }
  return result;
}

Tensor& clamp_(Tensor& self, optional<Scalar> min, optional<Scalar> max) {
  return at::AtenIpexTypeXPU::clamp_out(self, min, max, self);
}

Tensor& clamp_(
    Tensor& self,
    const c10::optional<Tensor>& min,
    const c10::optional<Tensor>& max) {
  return at::clamp_outf(self, min, max, self);
}

Tensor clamp(const Tensor& self, optional<Scalar> min, optional<Scalar> max) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeXPU::clamp_out(self, min, max, result);
}

Tensor clamp(
    const Tensor& self,
    const c10::optional<Tensor>& min,
    const c10::optional<Tensor>& max) {
  Tensor result = at::empty({0}, self.options());
  return at::clamp_outf(self, min, max, result);
}

Tensor& clamp_max_out(const Tensor& self, const Tensor& max, Tensor& result) {
  TORCH_CHECK(
      self.layout() == Layout::Strided,
      "torch.clamp only supports strided layout, got: ",
      self.layout());
  auto iter = TensorIterator::borrowing_binary_op(result, self, max);
  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf, kBFloat16, iter.common_dtype(), "clamp_max_dpcpp", [&] {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t in, scalar_t max_val) -> scalar_t {
              if (at::_isnan(in)) {
                return in;
              } else {
                return std::min(in, max_val);
              }
            });
      });
  return result;
}

Tensor& clamp_min_out(const Tensor& self, const Tensor& min, Tensor& result) {
  TORCH_CHECK(
      self.layout() == Layout::Strided,
      "torch.clamp only supports strided layout, got: ",
      self.layout());
  auto iter = TensorIterator::borrowing_binary_op(result, self, min);
  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf, kBFloat16, iter.common_dtype(), "clamp_min_dpcpp", [&] {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t in, scalar_t min_val) -> scalar_t {
              if (at::_isnan(in)) {
                return in;
              } else {
                return std::max(in, min_val);
              }
            });
      });
  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at
