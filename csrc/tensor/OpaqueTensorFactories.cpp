#include <ATen/ATen.h>
#include <ATen/quantized/QTensorImpl.h>
#include <core/Allocator.h>
#include <tensor/OpaqueTensorFactories.h>

namespace at {
namespace AtenIpexTypeXPU {

Tensor empty_opaque_tensor(
    DPCPPTensorContext::Meta meta,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format) {
  auto* allocator = xpu::dpcpp::getDeviceAllocator();
  auto dtype = options.dtype();

  // Here the opaque tensor is allocated on full size of block layout from the
  // memory descriptor.
  int64_t size_bytes = meta.get_size();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /*resizeable=*/true);

  auto tensor = detail::make_tensor<TensorImpl>(
      storage_impl, c10::DispatchKey::XPU, dtype);
  // should not work for an opaque tensor
  if (meta.dims().size() != 1 || meta.dims().at(0) != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(meta.dims());
  }

  auto memory_format =
      optional_memory_format.value_or(MemoryFormat::Contiguous);
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);

  auto ctx = (DPCPPTensorContext*)tensor.unsafeGetTensorImpl()
                 ->storage()
                 .unsafeGetStorageImpl()
                 ->data_ptr()
                 .get_context();
  ctx->set_meta(meta);
  return tensor;
}

Tensor empty_opaque_qtensor(
    DPCPPTensorContext::Meta meta,
    c10::optional<MemoryFormat> optional_memory_format,
    QuantizerPtr quantizer) {
  auto* allocator = xpu::dpcpp::getDeviceAllocator();
  auto dtype = scalarTypeToTypeMeta(quantizer->scalar_type());

  // Here the opaque tensor is allocated on full size of block layout from the
  // memory descriptor.
  int64_t size_bytes = meta.get_size();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /*resizeable=*/true);

  Tensor tensor;
  at::DispatchKey tensorDispatchKey = c10::DispatchKey::QuantizedXPU;
  tensor = detail::make_tensor<QTensorImpl>(
      storage_impl, at::DispatchKeySet(tensorDispatchKey), dtype, quantizer);
  // should not work for an opaque tensor
  if (meta.dims().size() != 1 || meta.dims().at(0) != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(meta.dims());
  }

  auto memory_format =
      optional_memory_format.value_or(MemoryFormat::Contiguous);
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);

  auto ctx = (DPCPPTensorContext*)tensor.unsafeGetTensorImpl()
                 ->storage()
                 .unsafeGetStorageImpl()
                 ->data_ptr()
                 .get_context();
  ctx->set_meta(meta);

  std::vector<float> scales;
  if (tensor.qscheme() == kPerTensorAffine) {
    scales.push_back(static_cast<float>(tensor.q_scale()));
  } else {
    for (int i = 0; i < tensor.q_per_channel_scales().numel(); i++) {
      scales.push_back(tensor.q_per_channel_scales()[i].item<float>());
    }
  }
  ctx->set_scales(scales);

  return tensor;
}

inline bool need_to_plain(const Tensor& tensor) {
  if (!tensor.defined())
    return false;

  if ((tensor.options().backend() != at::Backend::XPU &&
       tensor.options().backend() != at::Backend::QuantizedXPU) ||
      !DPCPPTensorConvertor::is_opaque_tensor(tensor))
    return false;

  if (tensor.is_sparse())
    return false;

  auto tensor_ctx = DPCPPTensorContext::get_tensor_ctx(tensor);
  if (tensor_ctx.is_plain())
    return false;

  return true;
}

Tensor to_plain_if_needed(const Tensor& tensor) {
  if (!need_to_plain(tensor))
    return tensor;
  const OptionalDeviceGuard device_guard(device_of(tensor));

  return DPCPPTensorConvertor::to_plain(tensor);
}

Tensor to_plain_if_needed_(const Tensor& tensor) {
  if (!need_to_plain(tensor))
    return tensor;

  auto plain = to_plain_if_needed(tensor);
  auto plain_ctx = (DPCPPTensorContext*)plain.unsafeGetTensorImpl()
                       ->storage()
                       .unsafeGetStorageImpl()
                       ->data_ptr()
                       .release_context();
  DPCPPTensorContext::set_tensor_ctx(tensor, std::move(*plain_ctx));
  return tensor;
}

std::vector<Tensor> to_plain_if_needed(TensorList tensors) {
  std::vector<Tensor> _tensors;
  for (auto tensor : tensors) {
    _tensors.push_back(to_plain_if_needed(tensor));
  }
  return _tensors;
}

MaterializedITensorListRef to_plain_if_needed(
    MaterializedITensorListRef tensors) {
  if (!Settings::I().is_onednn_layout_enabled())
    return tensors;
  MaterializedITensorListRef _tensors;
  for (auto tensor : tensors) {
    _tensors.push_back(std::reference_wrapper<const at::Tensor>(tensor.get()));
  }
  return _tensors;
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {
using AtenIpexTypeXPU::DPCPPTensorContext;
using AtenIpexTypeXPU::DPCPPTensorConvertor;
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
