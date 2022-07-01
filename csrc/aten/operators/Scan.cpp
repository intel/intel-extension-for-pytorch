#include <ATen/ATen.h>

#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include <utils/Helpers.h>
#include "PSTLFunctions.h"
#include "Scan.h"
#include "comm/ATDispatch.h"
#include "comm/MathReduce.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

using namespace at::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

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
