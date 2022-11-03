#include <ATen/InitialTensorOptions.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/quantized/QTensorImpl.h>
#include <core/TensorImplUtils.h>
#include <core/detail/ListUtils.h>
#include <oneDNN/oneDNN.h>
#include <quantized/QTensor.h>
#include <quantized/Quantizer.h>
#include <utils/DPCPP.h>

using namespace at::native;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

using namespace at::AtenIpexTypeQuantizedXPU;

Tensor new_qtensor(
    IntArrayRef sizes,
    const TensorOptions& options,
    QuantizerPtr quantizer) {
  auto memory_format =
      options.memory_format_opt().value_or(MemoryFormat::Contiguous);

  at::Allocator* allocator = xpu::dpcpp::getDeviceAllocator();

  at::DispatchKey tensorDispatchKey = options.computeDispatchKey();
  native::check_size_nonnegative(sizes);
  int64_t nelements = xpu::dpcpp::detail::prod_intlist(sizes);
  auto dtype = options.dtype();
  TORCH_CHECK(
      isQIntType(typeMetaToScalarType(dtype)),
      dtype,
      " is not supported in new_qtensor on xpu device.");
  int64_t size_bytes = nelements * dtype.itemsize();
  auto storage = c10::make_intrusive<StorageImpl>(
      StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /*resizable=*/true);
  auto tensor = detail::make_tensor<QTensorImpl>(
      storage, at::DispatchKeySet(tensorDispatchKey), dtype, quantizer);

  at::get_qtensorimpl(tensor)->set_sizes_contiguous(sizes);
  at::get_qtensorimpl(tensor)->empty_tensor_restride(memory_format);

  return tensor;
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {

using namespace at::AtenIpexTypeXPU;

int64_t q_zero_point(const Tensor& self) {
  return at::native::q_zero_point_quant(self);
}

double q_scale(const Tensor& self) {
  return at::native::q_scale_quant(self);
}

QScheme qscheme(const Tensor& self) {
  return at::native::qscheme_quant(self);
}

Tensor q_per_channel_scales(const Tensor& self) {
  return at::native::q_per_channel_scales(self);
}

Tensor q_per_channel_zero_points(const Tensor& self) {
  return at::native::q_per_channel_zero_points(self);
}

int64_t q_per_channel_axis(const Tensor& self) {
  return at::native::q_per_channel_axis(self);
}

Tensor& set_quantizer_(Tensor& self, ConstQuantizerPtr quantizer) {
  get_qtensorimpl(self)->set_quantizer_(quantizer);
  return self;
}

Tensor& set_(
    Tensor& self,
    Storage storage,
    int64_t storage_offset,
    IntArrayRef sizes,
    IntArrayRef strides) {
  auto* self_ = self.unsafeGetTensorImpl();
  self_->set_storage_keep_dtype(storage);
  self_->set_storage_offset(storage_offset);
  self_->set_sizes_and_strides(sizes, strides);
  return self;
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
