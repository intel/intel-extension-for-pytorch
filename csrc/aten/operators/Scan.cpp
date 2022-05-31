#include <ATen/ATen.h>

#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include <utils/Helpers.h>
#include "Scan.h"
#include "comm/ATDispatch.h"
#include "comm/MathReduce.h"
#include "comm/Numerics.h"
#include "comm/PSTLFunctions.h"
#include "comm/RegistrationDeclarations.h"

using namespace at::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <
    ScanType Type,
    class InputIt,
    class OutputIt,
    class T,
    class BinaryFunction>
static inline void _scan_kernel(
    InputIt input,
    OutputIt output,
    const int64_t problem,
    const int64_t stride,
    const int64_t batch,
    T init,
    BinaryFunction func) {
  scan_config cfg = {input, output, batch, problem, stride, init, Type, func};

  // 0. recursive convergence
  if (problem <= cfg.problem_wg_range_) {
    cfg.set_carrier(nullptr);
    launch_group_scan(cfg);
    return;
  }

  // 1. inclusive scan in each chunk
  Tensor carrier_holder = at::empty({cfg.carrier_size()}, map_options<T>());
  auto carrier = carrier_holder.data_ptr<T>();
  cfg.set_carrier(carrier);
  launch_group_scan(cfg);

  // 2. recursion for carrier
  _scan_kernel<EXCLUSIVE_TYPE>(
      carrier,
      carrier,
      /* carrier problem */ cfg.problem_glb_range_ / cfg.problem_wg_range_,
      /* carrier stride */ stride,
      /* carrier batch */ batch,
      init,
      func);

  // 3. accumulate among all chunk
  accumulate_carrier(cfg);

  return;
}

} // namespace impl

template <
    ScanType Type,
    typename scalar_t,
    typename oscalar_t,
    class BinaryFunction>
void scan(
    Tensor& self_,
    const Tensor& input_,
    int dimension,
    scalar_t init,
    BinaryFunction func) {
  self_.resize_as_(input_);
  if (input_.dim() == 0) {
    self_.fill_(input_);
    return;
  } else if (input_.numel() == 0) {
    self_.zero_();
    return;
  }

  dimension = maybe_wrap_dim(dimension, input_.dim());
  TORCH_CHECK(
      dimension >= 0 && dimension < input_.dim(),
      "dimension ",
      dimension,
      " out of range");

  auto self = self_.contiguous();
  auto input = input_.contiguous();

  auto N = self.numel();
  int64_t problem = input.size(dimension);
  int64_t stride = input.stride(dimension);
  int64_t batch = N / (problem * stride);
  auto input_ptr = input.data_ptr<scalar_t>();
  auto output_ptr = self.data_ptr<oscalar_t>();

  impl::_scan_kernel<Type>(
      input_ptr, output_ptr, problem, stride, batch, init, func);

  self_.copy_(self);
}

Tensor& cumsum_out(
    const Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype,
    Tensor& out) {
  if (self.dtype() == at::ScalarType::Bool) {
    IPEX_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Bool, out.scalar_type(), "cumsum", [&]() {
          scan<INCLUSIVE_TYPE, bool, scalar_t>(
              out,
              self,
              dim,
              ScalarConvert<float, bool>::to(0.0),
              AddOp<bool>());
        });
  } else {
    IPEX_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half, self.scalar_type(), "cumsum", [&]() {
          scan<INCLUSIVE_TYPE, scalar_t, scalar_t>(
              out,
              self,
              dim,
              ScalarConvert<float, scalar_t>::to(0.0),
              AddOp<scalar_t>());
        });
  }
  return out;
}

Tensor& cumprod_out(
    const Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype,
    Tensor& out) {
  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, self.scalar_type(), "cumprod", [&]() {
        scan<INCLUSIVE_TYPE, scalar_t, scalar_t>(
            out,
            self,
            dim,
            ScalarConvert<float, scalar_t>::to(1.0),
            MulOp<scalar_t>());
      });
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
