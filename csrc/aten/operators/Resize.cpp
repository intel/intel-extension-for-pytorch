#include <ATen/ATen.h>

#include <core/TensorImplUtils.h>
#include "comm/RegistrationDeclarations.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

const Tensor& resize_(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
  auto* self_ = self.unsafeGetTensorImpl();
  TensorImpl_resizeImpl(self_, size, /*strides=*/c10::nullopt);

  if (optional_memory_format.has_value()) {
    auto memory_format = optional_memory_format.value();
    TORCH_CHECK(
        memory_format != MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    self_->empty_tensor_restride(memory_format);
  }

  return self;
}

const Tensor& resize_as_(const Tensor& self, const Tensor& the_template) {
  return impl::resize_(self, the_template.sizes(), c10::nullopt);
}

} // namespace impl

const Tensor& resize_(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> memory_format) {
  impl::resize_(self, size, memory_format);
  return self;
}

const Tensor& resize_as_(
    const Tensor& self,
    const Tensor& the_template,
    c10::optional<MemoryFormat> memory_format) {
  return at::AtenIpexTypeXPU::resize_(
      self, the_template.sizes(), memory_format);
}
} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {
const Tensor& resize_(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> memory_format) {
  at::AtenIpexTypeXPU::impl::resize_(self, size, memory_format);
  return self;
}
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
