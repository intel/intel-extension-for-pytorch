#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <core/Memory.h>
#include <core/DPCPPUtils.h>
#include <core/Context.h>


using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

Scalar _local_scalar_dense_dpcpp(const Tensor& self) {
  Scalar r;
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Bool, at::ScalarType::Half, self.scalar_type(), "_local_scalar_dense_dpcpp", [&] {
        scalar_t value;
        dpcppMemcpy(&value, self.data_ptr<scalar_t>(), sizeof(scalar_t), DeviceToHost);
        r = Scalar(value);
      });
  return r;
}

} // namespace impl

Scalar _local_scalar_dense(const Tensor & self) {
  return impl::_local_scalar_dense_dpcpp(self);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
