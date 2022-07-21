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

template <typename acc_t>
struct ReduceProdOps {
  ReduceProdOps() {}
  acc_t operator()(acc_t a, acc_t b) const {
    return a * b;
  }
};

template <
    typename scalar_t,
    typename acc_t = scalar_t,
    typename out_t = scalar_t>
void prod_kernel_impl(TensorIterator& iter) {
  dpcpp_reduce_kernel<scalar_t, out_t>(
      iter, func_wrapper<out_t>(ReduceProdOps<acc_t>()), 1);
}

static void prod_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "prod",
      [&]() { prod_kernel_impl<scalar_t>(iter); });
}

Tensor& prod_out_impl(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dims,
    bool keepdim,
    c10::optional<ScalarType> opt_dtype) {
  ScalarType dtype = get_dtype(result, self, opt_dtype, true);
  auto iter = meta::make_reduction("prod", result, self, dims, keepdim, dtype);
  if (iter.numel() == 0) {
    result.fill_(1);
  } else {
    prod_kernel(iter);
  }
  return result;
}

Tensor& prod_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    c10::optional<ScalarType> dtype,
    Tensor& result) {
  return at::AtenIpexTypeXPU::prod_out_impl(
      result, self, {dim}, keepdim, dtype);
}

Tensor prod(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  Tensor result;
  return at::AtenIpexTypeXPU::prod_out_impl(
      result, self, {dim}, keepdim, dtype);
}

Tensor prod(const Tensor& self, c10::optional<ScalarType> dtype) {
  Tensor result;
  return at::AtenIpexTypeXPU::prod_out_impl(result, self, {}, false, dtype);
}

} // namespace AtenIpexTypeXPU
} // namespace at
