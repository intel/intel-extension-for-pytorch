#include <ATen/ATen.h>
#include <ATen/quantized/QTensorImpl.h>
#include <tensor/Context.h>

#include <ATen/aten_ipex_type_dpcpp.h>

namespace at {
namespace AtenIpexTypeDPCPP {

Tensor empty_opaque_tensor(
    DPCPPTensorContext::Meta meta,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format) {
  auto* allocator = at::dpcpp::getDPCPPDeviceAllocator();
  int64_t nelements = DPCPPTensorContext(nullptr, meta).padded_size();
  auto dtype = options.dtype();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      dtype,
      nelements,
      allocator->allocate(nelements * dtype.itemsize()),
      allocator,
      /*resizeable=*/true);

  auto tensor = detail::make_tensor<TensorImpl>(
      storage_impl, c10::DispatchKey::DPCPPTensorId);
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
  auto* allocator = at::dpcpp::getDPCPPDeviceAllocator();
  int64_t nelements = DPCPPTensorContext(nullptr, meta).padded_size();
  auto dtype = scalarTypeToTypeMeta(quantizer->scalar_type());
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      dtype,
      nelements,
      allocator->allocate(nelements * dtype.itemsize()),
      allocator,
      /*resizeable=*/true);

  Tensor tensor;
  at::DispatchKey tensorDispatchKey = c10::DispatchKey::QuantizedDPCPPTensorId;
  tensor = detail::make_tensor<QTensorImpl>(
      storage_impl, at::DispatchKeySet(tensorDispatchKey), quantizer);
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
  if (!lazy_reorder_enabled())
    return false;

  if (!tensor.defined())
    return false;

  if (tensor.options().backend() != at::Backend::DPCPP ||
      !DPCPPTensorConvertor::is_opaque_tensor(tensor))
    return false;
  return true;
}

Tensor to_plain_if_needed(const Tensor& tensor) {
  if (!need_to_plain(tensor))
    return tensor;

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
  if (!lazy_reorder_enabled())
    return tensors.vec();

  std::vector<Tensor> _tensors;
  for (auto tensor : tensors) {
    _tensors.push_back(to_plain_if_needed(tensor));
  }
  return _tensors;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
