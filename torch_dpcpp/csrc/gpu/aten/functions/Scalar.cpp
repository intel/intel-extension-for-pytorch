#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <c10/dpcpp/SYCLMemory.h>
#include <c10/dpcpp/SYCLUtils.h>

#include <core/SYCLContext.h>


namespace at {
namespace native {

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

}} // at::native
