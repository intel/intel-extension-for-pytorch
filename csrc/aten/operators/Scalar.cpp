#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include "comm/ATDispatch.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

Scalar _local_scalar_dense_dpcpp(const Tensor& self) {
  Scalar r;
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Bool,
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "_local_scalar_dense_dpcpp",
      [&] {
        scalar_t value;
        dpcppMemcpy(
            &value, self.data_ptr<scalar_t>(), sizeof(scalar_t), DeviceToHost);
        r = Scalar(value);
      });
  return r;
}

} // namespace impl

Scalar _local_scalar_dense(const Tensor& self) {
  return impl::_local_scalar_dense_dpcpp(self);
}

} // namespace AtenIpexTypeXPU
} // namespace at
