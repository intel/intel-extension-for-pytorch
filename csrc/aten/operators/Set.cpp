#include <ATen/core/TensorBody.h>
#include <ATen/native/Resize.h>
#include <core/Allocator.h>
#include <tensor/Tensor.h>
#include "comm/RegistrationDeclarations.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor& set_(Tensor& self, Storage source) {
  int64_t new_size =
      static_cast<int64_t>(source.nbytes() / self.dtype().itemsize());
  return self.set_(source, 0, new_size, {});
}

Tensor& set_(
    Tensor& self,
    Storage source,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  at::native::checkSetStorage(self, source, storage_offset, size, stride);

  self.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  c10::optional<IntArrayRef> stride_opt = stride.data() != nullptr
      ? c10::optional<IntArrayRef>(stride)
      : c10::nullopt;
  resize_impl(self.unsafeGetTensorImpl(), size, stride_opt);
  return self;
}

Tensor& set_(Tensor& self, const Tensor& source) {
  if (!self.is_same(source)) {
    return self.set_(
        source.storage(),
        source.storage_offset(),
        source.sizes(),
        source.strides());
  }
  return self;
}

Tensor& set_(Tensor& self) {
  ScalarType type = self.scalar_type();
  Storage storage(
      c10::Storage::use_byte_size_t(),
      0,
      xpu::dpcpp::getDeviceAllocator(),
      true);
  self.set_(storage, 0, {0}, {});
  TORCH_INTERNAL_ASSERT(type == self.scalar_type());
  return self;
}

bool is_set_to(const Tensor& self, const Tensor& tensor) {
  if (self.storage().unsafeGetStorageImpl() ==
          tensor.storage().unsafeGetStorageImpl() &&
      self.storage_offset() == tensor.storage_offset() &&
      self.dim() == tensor.dim()) {
    for (int64_t d = 0; d < self.dim(); ++d) {
      if (self.size(d) != tensor.size(d) ||
          self.stride(d) != tensor.stride(d)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

} // namespace AtenIpexTypeXPU
} // namespace at
