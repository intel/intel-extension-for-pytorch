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
struct ReduceAddOps {
  ReduceAddOps() {}
  acc_t operator()(acc_t a, acc_t b) const {
    return a + b;
  }
};

template <
    typename scalar_t,
    typename acc_t = scalar_t,
    typename out_t = scalar_t>
void sum_kernel_impl(TensorIterator& iter) {
  dpcpp_reduce_kernel<scalar_t, out_t>(
      iter, func_wrapper<out_t>(ReduceAddOps<acc_t>()));
}

void sum_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "sum",
      [&]() {
        using accscalar_t = acc_type<scalar_t>;
        sum_kernel_impl<scalar_t, accscalar_t>(iter);
      });
}

Tensor& sum_out(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> opt_dtype) {
  ScalarType dtype = get_dtype(result, self, opt_dtype, true);
  auto iter = meta::make_reduction("sum", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.zero_();
  } else {
    sum_kernel(iter);
  }
  return result;
}

Tensor sum(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  Tensor result;
  return at::AtenIpexTypeXPU::sum_out(result, self, dim, keepdim, dtype);
}

Tensor sum(const Tensor& self, c10::optional<ScalarType> dtype) {
  return at::AtenIpexTypeXPU::sum(self, std::vector<int64_t>{}, false, dtype);
}

template <
    typename scalar_t,
    typename acc_t = scalar_t,
    typename out_t = scalar_t>
void nansum_kernel_impl(TensorIterator& iter) {
  dpcpp_reduce_kernel<scalar_t, out_t>(iter, NanSumOps<acc_t, out_t>{});
}

Tensor& nansum_out(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> opt_dtype,
    Tensor& result) {
  TORCH_CHECK(
      !c10::isComplexType(self.scalar_type()),
      "nansum does not support complex inputs");
  // For integral types, use existing sum as
  // integral types don't have `Nan`.
  if (c10::isIntegralType(self.scalar_type(), true)) {
    return at::sum_out(result, self, dim, keepdim, opt_dtype);
  }

  ScalarType dtype = get_dtype_from_result(result, opt_dtype);
  auto iter = meta::make_reduction("nansum", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result = result.zero_();
  } else {
    IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "nansum",
        [&]() {
          using accscalar_t = acc_type<scalar_t>;
          nansum_kernel_impl<scalar_t, accscalar_t>(iter);
        });
  }
  return result;
}

Tensor nansum(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> opt_dtype) {
  ScalarType dtype = get_dtype_from_self(self, opt_dtype, true);
  Tensor result = create_reduction_result(self, dim, keepdim, dtype);
  return at::AtenIpexTypeXPU::nansum_out(self, dim, keepdim, dtype, result);
}

Tensor nansum(const Tensor& self, c10::optional<ScalarType> dtype) {
  return at::AtenIpexTypeXPU::nansum(
      self, std::vector<int64_t>{}, false, dtype);
}

} // namespace AtenIpexTypeXPU
} // namespace at
