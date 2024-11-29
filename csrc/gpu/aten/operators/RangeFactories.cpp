#include <ATen/NativeFunctions.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include "comm/AccumulateType.h"

#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include <cmath>
#include <limits>

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename T, typename accT = T>
class LinspaceOp {
 public:
  LinspaceOp(accT start, accT step) : start_(start), step_(step) {}
  T operator()(ptrdiff_t index) {
    accT increment = step_ * static_cast<accT>(index);
    accT value = start_ + increment;
    return static_cast<T>(value);
  }

  const accT start_, step_;
};

template <typename T, typename accT = T>
struct LogspaceOp {
  LogspaceOp(accT start, accT step, accT base)
      : start_(start), step_(step), base_(base) {}
  T operator()(ptrdiff_t index) {
    accT increment = step_ * static_cast<accT>(index);
    accT value = Numerics<accT>::pow(base_, start_ + increment);
    return static_cast<T>(value);
  }

  const accT start_, step_, base_;
};

template <typename scalar_t, typename func_t>
struct DpcppElementwiseKernelWithIndexImplFunctor {
  void operator()(sycl::item<1> item_id) const {
    auto idx = item_id.get_linear_id();
    out_ptr[idx] = f(idx);
  }
  DpcppElementwiseKernelWithIndexImplFunctor(scalar_t* out_ptr_, func_t f_)
      : out_ptr(out_ptr_), f(f_) {}

 private:
  scalar_t* out_ptr;
  func_t f;
};

// TODO: move it to the loops for more generic supporting.
template <typename scalar_t, typename func_t>
void dpcpp_elementwise_kernel_with_index_impl(
    scalar_t* out_ptr,
    int64_t N,
    func_t f) {
  if (N > std::numeric_limits<int>::max()) {
    dpcpp_elementwise_kernel_with_index_impl(
        out_ptr + std::numeric_limits<int>::max(),
        N - std::numeric_limits<int>::max(),
        f);
  }
  int64_t thread_number = std::min(N, (int64_t)std::numeric_limits<int>::max());
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgh) {
    DpcppElementwiseKernelWithIndexImplFunctor<scalar_t, func_t> kfn(
        out_ptr, f);
    cgh.parallel_for<decltype(kfn)>(sycl::range</*dim=*/1>(thread_number), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename func_t>
void dpcpp_elementwise_kernel_with_index(at::Tensor& output, func_t f) {
  using scalar_t = typename function_traits<func_t>::result_type;
  int64_t N = output.numel();
  if (N == 0) {
    return;
  }

  return dpcpp_elementwise_kernel_with_index_impl(
      output.data_ptr<scalar_t>(), N, f);
}

template <typename scalar_t>
struct linspace_dpcpp_out_functor {
  scalar_t operator()(int64_t ind) const {
    if (ind < halfway) {
      return scalar_start + (step * ind);
    }

    return scalar_end - step * (steps - ind - 1);
  }

  linspace_dpcpp_out_functor(
      scalar_t scalar_start,
      scalar_t scalar_end,
      int64_t steps,
      float step,
      const int64_t halfway)
      : scalar_start(scalar_start),
        scalar_end(scalar_end),
        steps(steps),
        step(step),
        halfway(halfway) {}

 private:
  scalar_t scalar_start;
  scalar_t scalar_end;
  int64_t steps;
  float step;
  const int64_t halfway;
};

template <typename scalar_t>
struct linspace_dpcpp_out_functor_2 {
  scalar_t operator()(int64_t ind) const {
    if (ind < halfway) {
      return scalar_start + (step * ind);
    }

    return scalar_end - step * (steps - ind - 1);
  }

  linspace_dpcpp_out_functor_2(
      scalar_t scalar_start,
      scalar_t scalar_end,
      int64_t steps,
      scalar_t step,
      const int64_t halfway)
      : scalar_start(scalar_start),
        scalar_end(scalar_end),
        steps(steps),
        step(step),
        halfway(halfway) {}

 private:
  scalar_t scalar_start;
  scalar_t scalar_end;
  int64_t steps;
  scalar_t step;
  const int64_t halfway;
};

Tensor& linspace_dpcpp_out(
    Tensor& result,
    Scalar start,
    Scalar end,
    c10::optional<int64_t> optional_steps) {
  const auto steps = optional_steps.value_or(100);
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  if (!optional_steps.has_value()) {
    TORCH_WARN_ONCE(
        "Not providing a value for linspace's steps is deprecated and will "
        "throw a runtime error in a future release. This warning will appear "
        "only once per process.");
  }

  if (result.numel() != steps) {
    result.resize_({steps});
  }

  bool is_contiguous = result.is_contiguous();
  Tensor r = !is_contiguous
      ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT)
      : result;

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    r.fill_(start);
  } else if (isIntegralType(r.scalar_type(), 0)) {
    IPEX_DISPATCH_INTEGRAL_TYPES(r.scalar_type(), "linspace_xpu", [&]() {
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      // Cast `end` and `start` to `float`, since range can be larger than
      // scalar_t for integral types
      float step =
          (static_cast<float>(scalar_end) - static_cast<float>(scalar_start)) /
          (steps - 1);
      const int64_t halfway = steps / 2;
      linspace_dpcpp_out_functor<scalar_t> f(
          scalar_start, scalar_end, steps, step, halfway);
      dpcpp_elementwise_kernel_with_index(r, f);
    });
  } else {
    IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kHalf, kBFloat16, r.scalar_type(), "linspace_xpu", [&]() {
          scalar_t scalar_start = start.to<scalar_t>();
          scalar_t scalar_end = end.to<scalar_t>();
          scalar_t step =
              (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
          const int64_t halfway = steps / 2;
          linspace_dpcpp_out_functor_2<scalar_t> f(
              scalar_start, scalar_end, steps, step, halfway);
          dpcpp_elementwise_kernel_with_index(r, f);
        });
  }

  if (!is_contiguous) {
    result.copy_(r);
  }

  return result;
}

template <typename scalar_t>
struct logspace_dpcpp_out_functor {
  scalar_t operator()(int64_t ind) const {
    if (ind < halfway) {
      return std::pow(scalar_base, scalar_start + step * ind);
    }
    return std::pow(scalar_base, scalar_end - step * (steps - ind - 1));
  }

  logspace_dpcpp_out_functor(
      scalar_t scalar_start,
      scalar_t scalar_end,
      float scalar_base,
      int64_t steps,
      float step,
      const int64_t halfway)
      : scalar_start(scalar_start),
        scalar_end(scalar_end),
        scalar_base(scalar_base),
        steps(steps),
        step(step),
        halfway(halfway) {}

 private:
  scalar_t scalar_start;
  scalar_t scalar_end;
  float scalar_base;
  int64_t steps;
  float step;
  const int64_t halfway;
};

template <typename scalar_t>
struct logspace_dpcpp_out_functor_2 {
  scalar_t operator()(int64_t ind) const {
    if (ind < halfway) {
      return std::pow(scalar_base, scalar_start + step * ind);
    }
    return std::pow(scalar_base, scalar_end - step * (steps - ind - 1));
  }

  logspace_dpcpp_out_functor_2(
      scalar_t scalar_start,
      scalar_t scalar_end,
      scalar_t scalar_base,
      int64_t steps,
      scalar_t step,
      const int64_t halfway)
      : scalar_start(scalar_start),
        scalar_end(scalar_end),
        scalar_base(scalar_base),
        steps(steps),
        step(step),
        halfway(halfway) {}

 private:
  scalar_t scalar_start;
  scalar_t scalar_end;
  scalar_t scalar_base;
  int64_t steps;
  scalar_t step;
  const int64_t halfway;
};

Tensor& logspace_dpcpp_out(
    Tensor& result,
    Scalar start,
    Scalar end,
    c10::optional<int64_t> optional_steps,
    double base) {
  const auto steps = optional_steps.value_or(100);
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  if (!optional_steps.has_value()) {
    TORCH_WARN_ONCE(
        "Not providing a value for logspace's steps is deprecated and will "
        "throw a runtime error in a future release. This warning will appear "
        "only once per process.");
  }

  if (result.numel() != steps) {
    result.resize_({steps});
  }

  bool is_contiguous = result.is_contiguous();
  Tensor r = !is_contiguous
      ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT)
      : result;

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    if (isComplexType(r.scalar_type())) {
      r.fill_(std::pow(base, start.to<c10::complex<double>>()));
    } else {
      r.fill_(std::pow(base, start.to<double>()));
    }
  } else if (isIntegralType(r.scalar_type(), /*includeBool=*/false)) {
    IPEX_DISPATCH_INTEGRAL_TYPES(r.scalar_type(), "logspace_xpu", [&]() {
      float scalar_base =
          static_cast<float>(base); // Use float to avoid promotion to double
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      float step = static_cast<float>(scalar_end - scalar_start) / (steps - 1);
      const int64_t halfway = steps / 2;
      logspace_dpcpp_out_functor<scalar_t> f(
          scalar_start, scalar_end, scalar_base, steps, step, halfway);
      dpcpp_elementwise_kernel_with_index(r, f);
    });
  } else {
    IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kHalf, kBFloat16, r.scalar_type(), "logspace_xpu", [&]() {
          scalar_t scalar_base = static_cast<scalar_t>(base);
          scalar_t scalar_start = start.to<scalar_t>();
          scalar_t scalar_end = end.to<scalar_t>();
          scalar_t step =
              (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
          const int64_t halfway = steps / 2;
          logspace_dpcpp_out_functor_2<scalar_t> f(
              scalar_start, scalar_end, scalar_base, steps, step, halfway);
          dpcpp_elementwise_kernel_with_index(r, f);
        });
  }

  if (!is_contiguous) {
    result.copy_(r);
  }

  return result;
}
} // namespace impl
Tensor& linspace_out(
    const Scalar& start,
    const Scalar& end,
    int64_t steps,
    Tensor& out) {
  impl::linspace_dpcpp_out(out, start, end, steps);
  return out;
}

Tensor& logspace_out(
    const Scalar& start,
    const Scalar& end,
    int64_t steps,
    double base,
    Tensor& out) {
  impl::logspace_dpcpp_out(out, start, end, steps, base);
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
