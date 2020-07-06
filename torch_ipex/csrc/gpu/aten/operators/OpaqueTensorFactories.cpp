#include <ATen/ATen.h>
#include <tensor/Context.h>

#include <ATen/aten_ipex_type_dpcpp.h>


namespace at {
namespace AtenIpexTypeDPCPP {

Tensor empty_opaque_tensor(
    DPCPPTensorContext::Meta& meta,
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

  auto ctx = (DPCPPTensorContext*)tensor.unsafeGetTensorImpl()->
      storage().unsafeGetStorageImpl()->data_ptr().get_context();
  ctx->set_meta(meta);
  return tensor;
}

Tensor to_plain_if_needed(const Tensor& tensor) {
  if (tensor.options().backend() != at::Backend::DPCPP ||
      !DPCPPTensorConvertor::is_opaque_tensor(tensor))
    return tensor;

  auto _tensor = at::AtenIpexTypeDPCPP::empty(
      tensor.sizes(), tensor.options(), c10::nullopt);
  if (DPCPPTensorConvertor::convert(_tensor, tensor))
    return _tensor;
  else
    return tensor;
}

TensorList to_plain_if_needed(TensorList tensors) {
  std::vector<Tensor> _tensors;
  for(auto tensor : tensors) {
    _tensors.push_back(to_plain_if_needed(tensor));
  }
  return {_tensors};
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
