#include <ATen/ATen.h>
#include <ATen/dpcpp/SYCLContext.h>

#include <ATen/native/dpcpp/Resize.h>

namespace at { namespace native {

Tensor& resize_sycl_(Tensor& self, IntArrayRef size) {
  auto * self_ = self.unsafeGetTensorImpl();
  resize_impl_sycl_(self_, size, /*strides=*/c10::nullopt);
  self_->maybe_zero_dim(size.size() == 0);
  return self;
}

Tensor& resize_as_sycl_(Tensor& self, const Tensor& the_template) {
  return resize_sycl_(self, the_template.sizes());
}


}}

