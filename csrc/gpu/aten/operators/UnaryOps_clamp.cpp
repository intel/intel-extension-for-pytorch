#include <ATen/ATen.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"
#include "comm/ATDispatch.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
struct clamp_max_out_functor {
  scalar_t operator()(scalar_t in) const {
    return Numerics<scalar_t>::isnan(in)  ? in
        : Numerics<scalar_t>::gt(in, val) ? val
                                          : in;
  }

  clamp_max_out_functor(scalar_t val) : val(val) {}

 private:
  scalar_t val;
};

Tensor& clamp_max_out(const Tensor& self, const Scalar& max, Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "clamp_max_out",
      [&]() {
        auto val = max.to<scalar_t>();
        clamp_max_out_functor<scalar_t> f(val);
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  return out;
}

template <typename scalar_t>
struct clamp_min_out_functor {
  scalar_t operator()(scalar_t in) const {
    return Numerics<scalar_t>::isnan(in)  ? in
        : Numerics<scalar_t>::lt(in, val) ? val
                                          : in;
  }

  clamp_min_out_functor(scalar_t val) : val(val) {}

 private:
  scalar_t val;
};

Tensor& clamp_min_out(const Tensor& self, const Scalar& min, Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "clamp_min_out",
      [&]() {
        auto val = min.to<scalar_t>();
        clamp_min_out_functor<scalar_t> f(val);
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  return out;
}

template <typename scalar_t>
struct clamp_min_max_functor {
  scalar_t operator()(scalar_t in) const {
    auto val = Numerics<scalar_t>::lt(in, maxValue) ? in : maxValue;
    return Numerics<scalar_t>::isnan(in)        ? in
        : Numerics<scalar_t>::gt(minValue, val) ? minValue
                                                : val;
  }

  clamp_min_max_functor(scalar_t minValue, scalar_t maxValue)
      : minValue(minValue), maxValue(maxValue) {}

 private:
  scalar_t minValue;
  scalar_t maxValue;
};

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
        clamp_min_max_functor<scalar_t> f(minValue, maxValue);
        dpcpp_kernel_for_tensor_iter(iter, f);
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

template <typename scalar_t>
struct clamp_out_functor {
  scalar_t operator()(scalar_t in, scalar_t min_val, scalar_t max_val) const {
    if (Numerics<scalar_t>::isnan(in)) {
      return in;
    } else {
      return Numerics<scalar_t>::min(
          Numerics<scalar_t>::max(in, min_val), max_val);
    }
  }
};

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
          clamp_out_functor<scalar_t> f;
          dpcpp_kernel_for_tensor_iter(iter, f);
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

Tensor& clamp_(
    Tensor& self,
    const c10::optional<Tensor>& min,
    const c10::optional<Tensor>& max) {
  return at::clamp_outf(self, min, max, self);
}

Tensor clamp(
    const Tensor& self,
    const c10::optional<Tensor>& min,
    const c10::optional<Tensor>& max) {
  Tensor result = at::empty({0}, self.options());
  return at::clamp_outf(self, min, max, result);
}

template <typename scalar_t>
struct clamp_max_out_dpcpp_functor {
  scalar_t operator()(scalar_t in, scalar_t max_val) const {
    if (Numerics<scalar_t>::isnan(in)) {
      return in;
    } else {
      return Numerics<scalar_t>::min(in, max_val);
    }
  }
};

Tensor& clamp_max_out(const Tensor& self, const Tensor& max, Tensor& result) {
  TORCH_CHECK(
      self.layout() == Layout::Strided,
      "torch.clamp only supports strided layout, got: ",
      self.layout());
  auto iter = TensorIterator::borrowing_binary_op(result, self, max);
  IPEX_DISPATCH_ALL_TYPES_AND2(
      kHalf, kBFloat16, iter.common_dtype(), "clamp_max_dpcpp", [&] {
        clamp_max_out_dpcpp_functor<scalar_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  return result;
}

template <typename scalar_t>
struct clamp_min_out_dpcpp_functor {
  scalar_t operator()(scalar_t in, scalar_t min_val) const {
    if (Numerics<scalar_t>::isnan(in)) {
      return in;
    } else {
      return Numerics<scalar_t>::max(in, min_val);
    }
  }
};

Tensor& clamp_min_out(const Tensor& self, const Tensor& min, Tensor& result) {
  TORCH_CHECK(
      self.layout() == Layout::Strided,
      "torch.clamp only supports strided layout, got: ",
      self.layout());
  auto iter = TensorIterator::borrowing_binary_op(result, self, min);
  IPEX_DISPATCH_ALL_TYPES_AND2(
      kHalf, kBFloat16, iter.common_dtype(), "clamp_min_dpcpp", [&] {
        clamp_min_out_dpcpp_functor<scalar_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at
