#include <ATen/ATen.h>

#include <core/SYCLContext.h>
#include <core/TensorImplUtils.h>

namespace at { namespace native {

Tensor& resize_sycl_(Tensor& self, IntArrayRef size) {
  auto * self_ = self.unsafeGetTensorImpl();
  TensorImpl_resizeImpl(self_, size, /*strides=*/c10::nullopt);
  self_->maybe_zero_dim(size.size() == 0);
  return self;
}

Tensor& resize_as_sycl_(Tensor& self, const Tensor& the_template) {
  return resize_sycl_(self, the_template.sizes());
}

}}


namespace at { namespace AtenIpexTypeDPCPP {
Tensor& resize_(Tensor & self, IntArrayRef size, c10::optional<MemoryFormat> memory_format){
  at::native::resize_sycl_(self, size);
  return self;
}

Tensor& resize_as_(Tensor& self, const Tensor& the_template, c10::optional<MemoryFormat> memory_format){
  return resize_(self, the_template.sizes(), memory_format);
}
}}