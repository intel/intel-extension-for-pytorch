#include <core/TensorImplUtils.h>

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {

Tensor& set_(Tensor& self, Storage source) {
  TensorImpl_setStorage(
      TensorImpl_Unwrap(self),
      source.unsafeGetStorageImpl(),
      0,
      {static_cast<int64_t>(source.size())},
      {});
  return self;
}

Tensor& set_(
    Tensor& self,
    Storage source,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  TensorImpl_setStorage(
      TensorImpl_Unwrap(self),
      source.unsafeGetStorageImpl(),
      storage_offset,
      size,
      stride);
  return self;
}

Tensor& set_(Tensor& self, const Tensor& source) {
  TensorImpl_set(TensorImpl_Unwrap(self), TensorImpl_Unwrap(source));
  return self;
}

Tensor& set_(Tensor& self) {
  TensorImpl_setStorage(TensorImpl_Unwrap(self), NULL, 0, {0}, {});
  return self;
}

bool is_set_to(const Tensor& self, const Tensor& tensor) {
  return TensorImpl_isSetTo(TensorImpl_Unwrap(self), TensorImpl_Unwrap(tensor));
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
