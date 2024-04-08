#include <ATen/ATen.h>
#include <ATen/NumericUtils.h>
#include <ATen/OpMathType.h>
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

template <typename scalar_t, typename opmath_t>
struct clamp_max_out_functor {
  scalar_t operator()(scalar_t in_) const {
    auto in = static_cast<opmath_t>(in_);
    return Numerics<opmath_t>::isnan(in) ? in
                                         : Numerics<opmath_t>::min(val, in);
  }

  clamp_max_out_functor(opmath_t val) : val(val) {}

 private:
  opmath_t val;
};

Tensor& clamp_max_out(const Tensor& self, const Scalar& max, Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self.to(out.scalar_type()));
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "clamp_max_out",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto val = max.to<opmath_t>();
        clamp_max_out_functor<scalar_t, opmath_t> f(val);
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  return out;
}

template <typename scalar_t, typename opmath_t>
struct clamp_min_out_functor {
  scalar_t operator()(scalar_t in_) const {
    auto in = static_cast<opmath_t>(in_);
    return Numerics<opmath_t>::isnan(in) ? in
                                         : Numerics<opmath_t>::max(in, val);
  }

  clamp_min_out_functor(opmath_t val) : val(val) {}

 private:
  opmath_t val;
};

Tensor& clamp_min_out(const Tensor& self, const Scalar& min, Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self.to(out.scalar_type()));
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "clamp_min_out",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto val = min.to<opmath_t>();
        clamp_min_out_functor<scalar_t, opmath_t> f(val);
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  return out;
}

template <typename scalar_t, typename opmath_t>
struct clamp_min_max_functor {
  scalar_t operator()(scalar_t in_) const {
    auto in = static_cast<opmath_t>(in_);
    if (Numerics<opmath_t>::isnan(in)) {
      return in_;
    }
    if (Numerics<opmath_t>::isnan(min_value)) {
      return min_value;
    }
    if (Numerics<opmath_t>::isnan(max_value)) {
      return max_value;
    }

    if (min_value > max_value) {
      return max_value;
    }
    return (Numerics<opmath_t>::min(
        Numerics<opmath_t>::max(in, min_value), max_value));
  }

  clamp_min_max_functor(opmath_t minValue, opmath_t maxValue)
      : min_value(minValue), max_value(maxValue) {}

 private:
  opmath_t min_value;
  opmath_t max_value;
};

Tensor& clamp_min_max(
    const Tensor& self,
    const Scalar& min,
    const Scalar& max,
    Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self.to(out.scalar_type()));
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "clamp_min_max",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto min_value = min.to<opmath_t>();
        auto max_value = max.to<opmath_t>();
        clamp_min_max_functor<scalar_t, opmath_t> f(min_value, max_value);
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
    if (at::_isnan(in)) {
      return in;
    }
    if (at::_isnan(min_val)) {
      return min_val;
    }
    if (at::_isnan(max_val)) {
      return max_val;
    }
    // If min is greater than max torch.clamp(..., min, max) sets
    // all elements in input to the value of max.
    if (min_val > max_val) {
      return max_val;
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
        iter.common_dtype(),
        "clamp_min_max",
        [&]() {
          clamp_out_functor<scalar_t> f;
          dpcpp_kernel_for_tensor_iter(iter, f);
        });
    result = iter.output();
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
  // for type promotion, result tensor need to be undefined.
  Tensor result;
  return at::clamp_outf(self, min, max, result);
}

template <typename scalar_t>
struct clamp_max_out_dpcpp_functor {
  scalar_t operator()(scalar_t in, scalar_t max_val_) const {
    using opmath_t = at::opmath_type<scalar_t>;
    auto max_val = static_cast<opmath_t>(max_val_);
    if (Numerics<opmath_t>::isnan(in)) {
      return in;
    } else if (Numerics<opmath_t>::isnan(max_val)) {
      return max_val;
    } else {
      return Numerics<scalar_t>::min(static_cast<opmath_t>(in), max_val);
    }
  }
};

struct clamp_max_out_dpcpp_bool_functor {
  bool operator()(bool in, bool max_val) const {
    return in && max_val;
  }
};

Tensor& clamp_max_out(const Tensor& self, const Tensor& max, Tensor& result) {
  TORCH_CHECK(
      self.layout() == Layout::Strided,
      "torch.clamp only supports strided layout, got: ",
      self.layout());
  auto iter = TensorIterator::borrowing_binary_op(result, self, max);

  if (iter.dtype() == ScalarType::Bool) {
    clamp_max_out_dpcpp_bool_functor f;
    opmath_symmetric_gpu_kernel_with_scalars<bool>(iter, f);
  } else {
    IPEX_DISPATCH_ALL_TYPES_AND2(
        kHalf, kBFloat16, iter.common_dtype(), "clamp_max_dpcpp", [&] {
          clamp_max_out_dpcpp_functor<scalar_t> f;
          dpcpp_kernel_for_tensor_iter(iter, f);
        });
  }
  result = iter.output();
  return result;
}

template <typename scalar_t>
struct clamp_min_out_dpcpp_functor {
  scalar_t operator()(scalar_t in, scalar_t min_val_) const {
    using opmath_t = at::opmath_type<scalar_t>;
    auto min_val = static_cast<opmath_t>(min_val_);
    if (Numerics<opmath_t>::isnan(in)) {
      return in;
    } else if (Numerics<opmath_t>::isnan(min_val)) {
      return min_val;
    } else {
      return Numerics<opmath_t>::max(static_cast<opmath_t>(in), min_val);
    }
  }
};

struct clamp_min_out_dpcpp_bool_functor {
  bool operator()(bool in, bool min_val) const {
    return (in || min_val);
  }
};

Tensor& clamp_min_out(const Tensor& self, const Tensor& min, Tensor& result) {
  TORCH_CHECK(
      self.layout() == Layout::Strided,
      "torch.clamp only supports strided layout, got: ",
      self.layout());
  auto iter = TensorIterator::borrowing_binary_op(result, self, min);

  if (iter.dtype() == ScalarType::Bool) {
    clamp_min_out_dpcpp_bool_functor f;
    opmath_symmetric_gpu_kernel_with_scalars<bool>(iter, f);
  } else {
    IPEX_DISPATCH_ALL_TYPES_AND2(
        kHalf, kBFloat16, iter.common_dtype(), "clamp_min_dpcpp", [&] {
          clamp_min_out_dpcpp_functor<scalar_t> f;
          dpcpp_kernel_for_tensor_iter(iter, f);
        });
  }
  result = iter.output();

  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at
