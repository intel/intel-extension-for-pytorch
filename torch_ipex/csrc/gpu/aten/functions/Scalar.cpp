#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <core/SYCLMemory.h>
#include <core/SYCLUtils.h>
#include <core/SYCLContext.h>


namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

Scalar _local_scalar_dense_sycl(const Tensor& self) {
  Scalar r;
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Bool, at::ScalarType::Half, self.scalar_type(), "_local_scalar_dense_sycl", [&] {
        scalar_t value;
        c10::sycl::syclMemcpy(&value, self.data_ptr<scalar_t>(), sizeof(scalar_t), c10::sycl::DeviceToHost);
        r = Scalar(value);
      });
  return r;
}

} // namespace impl

Scalar _local_scalar_dense(const Tensor & self) {
  return impl::_local_scalar_dense_sycl(self);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
