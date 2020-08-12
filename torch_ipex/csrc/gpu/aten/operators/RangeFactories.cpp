#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>
#include <utils/AccumulateType.h>

#include <core/DPCPP.h>
#include <core/Memory.h>
#include <utils/Algorithm.h>
#include <utils/Numerics.h>
#include <utils/ATDispatch.h>

#include <cmath>
#include <limits>

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

DPCPP_DEF_K1(linspace_dpcpp_ker);
DPCPP_DEF_K1(logspace_dpcpp_ker);
DPCPP_DEF_K1(range_dpcpp_ker);
DPCPP_DEF_K1(arange_dpcpp_ker);

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

Tensor& linspace_dpcpp_out(
    Tensor& result,
    Scalar start,
    Scalar end,
    int64_t steps) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  if (result.numel() != steps) {
    result.resize_({steps});
  }
  Tensor r = result.is_contiguous() ? result : result.contiguous();

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    r.fill_(start);
  } else {
    IPEX_DISPATCH_FLOATING_TYPES(r.scalar_type(), "linspace_dpcpp", [&]() {
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      scalar_t step =
          (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
      LinspaceOp<scalar_t> linspace_method(scalar_start, step);
      auto dpcpp_queue = dpcppGetCurrentQueue();
      auto cgf = DPCPP_Q_CGF(cgh) {
        auto r_data =
            get_buffer<dpcpp_discard_w_mode>(cgh, r.data_ptr<scalar_t>());
        // kernel function per work-item
        auto kfn = DPCPP_Q_KFN() {
          auto ptr = get_pointer(r_data);
          dpcpp_tabulate(ptr, ptr + steps, linspace_method);
        };
        // kick off kernel
        // (TODO) single_task need replaced due to low efficiency
        cgh.single_task<DPCPP_K(linspace_dpcpp_ker, scalar_t)>(kfn);
      };

      // submit to DPCPP queue
      DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
    });
  }

  if (!result.is_contiguous()) {
    result.copy_(r);
  }
  return result;
}

Tensor& logspace_dpcpp_out(
    Tensor& result,
    Scalar start,
    Scalar end,
    int64_t steps,
    double base) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  if (result.numel() != steps) {
    result.resize_({steps});
  }
  Tensor r = result.is_contiguous() ? result : result.contiguous();

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    r.fill_(Numerics<double>::pow(10.0, start.to<double>()));
  } else {
    IPEX_DISPATCH_FLOATING_TYPES(r.scalar_type(), "logspace_dpcpp", [&]() {
      scalar_t scalar_base = static_cast<scalar_t>(base);
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      scalar_t step =
          (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
      LogspaceOp<scalar_t> logspace_method(scalar_start, step, scalar_base);
      auto dpcpp_queue = dpcppGetCurrentQueue();
      auto cgf = DPCPP_Q_CGF(cgh) {
        auto r_data =
            get_buffer<dpcpp_discard_w_mode>(cgh, r.data_ptr<scalar_t>());
        // kernel function per work-item
        auto kfn = DPCPP_Q_KFN() {
          auto ptr = get_pointer(r_data);
          dpcpp_tabulate(ptr, ptr + steps, logspace_method);
        };
        // kick off kernel
        // (TODO) single_task need replaced due to low efficiency
        cgh.single_task<DPCPP_K(logspace_dpcpp_ker, scalar_t)>(kfn);
      };

      // submit to DPCPP queue
      DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
    });
  }

  if (!result.is_contiguous()) {
    result.copy_(r);
  }

  return result;
}

Tensor& range_dpcpp_out(Tensor& result, Scalar start, Scalar end, Scalar step) {
  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, result.scalar_type(), "range_dpcpp", [&]() {
        using accscalar_t = acc_type<scalar_t>;
        auto xstart = start.to<accscalar_t>();
        auto xend = end.to<accscalar_t>();
        auto xstep = step.to<accscalar_t>();

        TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
        TORCH_CHECK(
            std::isfinite(static_cast<double>(xstart)) &&
                std::isfinite(static_cast<double>(xend)),
            "unsupported range: ",
            xstart,
            " -> ",
            xend);
        TORCH_CHECK(
            ((xstep > 0) && (xend >= xstart)) ||
                ((xstep < 0) && (xend <= xstart)),
            "upper bound and larger bound inconsistent with step sign");
        int64_t size = static_cast<int64_t>(((xend - xstart) / xstep) + 1);
        if (result.numel() != size) {
          result.resize_({size});
        }
        Tensor r = result.is_contiguous() ? result : result.contiguous();
        LinspaceOp<scalar_t, accscalar_t> linspace_method(xstart, xstep);
        auto dpcpp_queue = dpcppGetCurrentQueue();

        // command group functions
        auto cgf = DPCPP_Q_CGF(cgh) {
          auto r_data = get_buffer<dpcpp_r_mode>(cgh, r.data_ptr<scalar_t>());

          // kernel function per work-item
          auto kfn = DPCPP_Q_KFN() {
            auto ptr = get_pointer(r_data);
            dpcpp_tabulate(ptr, ptr + size, linspace_method);
          };
          // kick off kernel
          // (TODO) single_task need replaced due to low efficiency
          cgh.single_task<DPCPP_K(range_dpcpp_ker, scalar_t)>(kfn);
        };

        // submit to DPCPP queue
        DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);

        if (!result.is_contiguous()) {
          result.copy_(r);
        }
      });

  return result;
}

Tensor& arange_dpcpp_out(
    Tensor& result,
    Scalar start,
    Scalar end,
    Scalar step) {
  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, result.scalar_type(), "arange_dpcpp", [&]() {
        using accscalar_t = acc_type<scalar_t>;
        auto xstart = start.to<accscalar_t>();
        auto xend = end.to<accscalar_t>();
        auto xstep = step.to<accscalar_t>();

        double size_d;
        if (std::is_same<scalar_t, int64_t>::value) {
          size_d = std::ceil(
              static_cast<double>(
                  end.to<accscalar_t>() - start.to<accscalar_t>()) /
              step.to<accscalar_t>());
        } else {
          size_d = std::ceil(
              static_cast<double>(end.to<double>() - start.to<double>()) /
              step.to<double>());
        }

        TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
        TORCH_CHECK(
            std::isfinite(static_cast<double>(xstart)) &&
                std::isfinite(static_cast<double>(xend)),
            "unsupported range: ",
            xstart,
            " -> ",
            xend);
        TORCH_CHECK(
            ((xstep > 0) && (xend >= xstart)) ||
                ((xstep < 0) && (xend <= xstart)),
            "upper bound and larger bound inconsistent with step sign");

        TORCH_CHECK(
            size_d >= 0 &&
                size_d <=
                    static_cast<double>(std::numeric_limits<int64_t>::max()),
            "invalid size, possible overflow?");
        int64_t size = static_cast<int64_t>(size_d);

        if (result.numel() != size) {
          result.resize_({size});
        }
        LinspaceOp<scalar_t, accscalar_t> linspace_method(xstart, xstep);
        auto dpcpp_queue = dpcppGetCurrentQueue();

        // command group functions
        auto cgf = DPCPP_Q_CGF(cgh) {
          auto acc =
              get_buffer<dpcpp_r_mode>(cgh, result.data_ptr<scalar_t>());

          // kernel function per work-item
          auto kfn = DPCPP_Q_KFN() {
            auto ptr = get_pointer(acc);
            dpcpp_tabulate(ptr, ptr + size, linspace_method);
          };
          // kick off kernel
          // (TODO) single_task need replaced due to low efficiency
          cgh.single_task<DPCPP_K(arange_dpcpp_ker, scalar_t)>(kfn);
        };

        // submit to DPCPP queue
        DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
      });
  return result;
}

} // namespace impl

Tensor& linspace_out(Tensor& out, Scalar start, Scalar end, int64_t steps) {
  impl::linspace_dpcpp_out(out, start, end, steps);
  return out;
}

Tensor& logspace_out(
    Tensor& out,
    Scalar start,
    Scalar end,
    int64_t steps,
    double base) {
  impl::logspace_dpcpp_out(out, start, end, steps, base);
  return out;
}

Tensor& range_out(Tensor& out, Scalar start, Scalar end, Scalar step) {
  impl::range_dpcpp_out(out, start, end, step);
  return out;
}

Tensor& arange_out(Tensor& out, Scalar start, Scalar end, Scalar step) {
  impl::arange_dpcpp_out(out, start, end, step);
  return out;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
