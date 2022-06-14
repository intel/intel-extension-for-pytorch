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
    typename T,
    class InputInfo,
    class OutputInfo,
    class BinaryFunction>
static inline void _scan_kernel(
    InputInfo& input_info,
    OutputInfo& output_info,
    int dim_after_collapse,
    T init,
    BinaryFunction func) {
  auto cfg = ScanConfig<InputInfo, OutputInfo, T, BinaryFunction>::make_config(
      input_info, output_info, dim_after_collapse, init, Type, func);

  // 0. recursive convergence
  if (cfg.problem_ <= cfg.problem_wg_range_) {
    cfg.set_carrier(nullptr);
    launch_group_scan(cfg);
    return;
  }

  // 1. inclusive scan in each chunk
  Tensor carrier_holder = at::empty(
      {cfg.batch_, cfg.problem_glb_range_ / cfg.problem_wg_range_, cfg.stride_},
      map_options<T>());
  TensorInfo<T, int64_t> carrier_info =
      getTensorInfo<T, int64_t>(carrier_holder);
  cfg.set_carrier(carrier_info.data);
  launch_group_scan(cfg);

  // 2. recursion for carrier
  _scan_kernel<EXCLUSIVE_TYPE>(carrier_info, carrier_info, 1, init, func);

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
    Tensor& self,
    const Tensor& input,
    int dimension,
    scalar_t init,
    BinaryFunction func) {
  self.resize_as_(input);
  if (input.dim() == 0) {
    self.fill_(input);
    return;
  } else if (input.numel() == 0) {
    self.zero_();
    return;
  }

  dimension = maybe_wrap_dim(dimension, input.dim());
  TORCH_CHECK(
      dimension >= 0 && dimension < input.dim(),
      "dimension ",
      dimension,
      " out of range");

  TensorInfo<scalar_t, int64_t> input_info =
      getTensorInfo<scalar_t, int64_t>(input);
  int dim_after_collapse = input_info.collapseDims(dimension);

  TensorInfo<oscalar_t, int64_t> output_info =
      getTensorInfo<oscalar_t, int64_t>(self);
  output_info.collapseDims(dimension);

  impl::_scan_kernel<Type>(
      input_info, output_info, dim_after_collapse, init, func);
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
