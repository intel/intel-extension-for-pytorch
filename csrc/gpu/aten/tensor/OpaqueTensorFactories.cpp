#include <ATen/ATen.h>
#include <ATen/native/Resize.h>
#include <ATen/quantized/QTensorImpl.h>
#include <core/Allocator.h>
#include <core/MemoryFormat.h>
#include <tensor/OpaqueTensorFactories.h>
#include <torch/library.h>

namespace at {
namespace AtenIpexTypeXPU {

// Definitions in opaque solution:
// - Interpretable - Interpretable to PyTorch Tensor meta.
// - Compatible    - Can share storage between PyTorch Tensor aliases/metas.
//                   Compatible plain metas share same storage.
//
// - Plain meta    - PyTorch interpretable memory layout or data type.
// - Opaque meta   - PyTorch non-interpretable memory layout or data type.
//
// - Tensor meta   - Data member of PyTorch Tensor, including a plain meta.
// - CTX meta      - Data member of DPCPPTensorContext,
//                   including an opaque meta and its corresponding plain meta.
// A XPU opaque tensor will be,
// Tensor +
//        - sizes   /* tensor meta, is plain meta */
//        - strides /* tensor meta, is plain meta */
//        - dtype   /* tensor meta, is plain meta */
//        - DataPtr + /* DPCPPTensorContext */
//                  - meta      /* opaque meta */
//                  - aten_meta /* plain meta, is compatible with tensor meta */
//
// Opaque implementation needs ensure,
// #1. Compatible between Tensor meta and CTX plain meta. Not necessary to
//     keep them same.
// #2. Interpretable CTX plain meta to PyTorch Tensor.
// Opaque cases,
// - memory layout - oneDNN convolution blocked layout.
// - data type     - oneDNN max_pooling int32_t indices.
//
// API manual: Create an opaque Tensor by,
// #1. creating a PyTorch Tensor, using shape in CTX meta and memory_format.
// #2. creating a CTX meta, recording both plain meta and opaque meta.
// To decouple Tensor meta and CTX meta. Use CTX meta to retrieve
// original storage, not Tensor meta to avoid retrieving plain from
// a reinterpreted and sharing-storage Tensor or a raw storage.
// We don't record plain data type, since data type is same between Tensor
// meta and CTX plain meta.
//
// Limitation:
// Cannot create an opaque tensor basing on a plain tensor with a plain meta,
// which interprets partial of a storage. Substitution could not retrieve whole
// storage basing on a partial plain meta recorded in CTX.
//
// XPU opaque tensor usage:
// #1. Persistent tensor. Use the API to create an opaque CTX, and invade
//     persistent Tensor wrapper with the CTX. In such cases, sometimes, except
//     for calling the func to create an opaque CTX, users need record
//     compatible plain meta to make the storage interpretable to persistent
//     Tensor by calling
//     `ctx.set_aten_meta(compatible_shape, compatible_stride)`.
// #2. New tensor creation. Use the API to create a new Tensor with an opaque
//     CTX. Interpretable plain meta is recorded by default. The new Tensor
//     is contiguous follow
//
// A case of #1, sometimes, cannot use PyTorch memory format to create a
// compatible PyTorch 4D tensor with 5D opaque meta which is needed by oneDNN.
// E.g. group_conv. Need more sentences to make the storage compatible and
// interpretable to Tensor meta when converting back to plain,
//
// auto t_5d = empty_opaque_tensor(md_5d, options, c10::nullptr);
// auto ctx_5d = release_tensor_ctx(t_5d);
// ctx_5d->set_aten_meta(compatible_stride(t_4d)); /* override plain meta */
// set_tensor_ctx(t_4d, ctx_5d);                   /* ctor Tensor wrapper */
//
// Another case of #1, like persistent tensor is not contiguous. In the func,
// cannot create a tensor with non-contiguous strides.
//
// API users are responsible for give an compatible and interpretable plain meta
// in CTX for case #1.
//
Tensor empty_opaque_tensor(
    DPCPPTensorContext::Meta meta,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format) {
  auto* allocator = xpu::dpcpp::getDeviceAllocator();
  auto dtype = options.dtype();

  at::detail::check_size_nonnegative(meta.dims());

  int64_t size_bytes = meta.get_size();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /*resizeable=*/true);

  auto tensor = detail::make_tensor<TensorImpl>(
      storage_impl, c10::DispatchKey::XPU, dtype);
  if (meta.dims().size() != 1 || meta.dims().at(0) != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(meta.dims());
  }

  auto memory_format =
      optional_memory_format.value_or(MemoryFormat::Contiguous);
  if (memory_format == CHANNELSLAST1D_DPCPP) {
    TORCH_CHECK(
        meta.dims().size() == 3,
        "required rank 3 tensor to use channels_last_1d format");
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
        meta.dims(),
        xpu::dpcpp::get_channels_last_strides_1d_dpcpp(meta.dims()));
  } else {
    tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  }

  auto ctx = (DPCPPTensorContext*)tensor.unsafeGetTensorImpl()
                 ->storage()
                 .unsafeGetStorageImpl()
                 ->data_ptr()
                 .get_context();
  ctx->set_meta(meta);
  ctx->set_aten_meta({tensor.sizes().vec(), tensor.strides().vec()});
  return tensor;
}

Tensor empty_opaque_qtensor(
    DPCPPTensorContext::Meta meta,
    c10::optional<MemoryFormat> optional_memory_format,
    QuantizerPtr quantizer) {
  at::detail::check_size_nonnegative(meta.dims());

  auto* allocator = xpu::dpcpp::getDeviceAllocator();
  auto dtype = scalarTypeToTypeMeta(quantizer->scalar_type());

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
  if (meta.dims().size() != 1 || meta.dims().at(0) != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(meta.dims());
  }

  auto memory_format =
      optional_memory_format.value_or(MemoryFormat::Contiguous);
  if (memory_format == CHANNELSLAST1D_DPCPP) {
    TORCH_CHECK(
        meta.dims().size() == 3,
        "required rank 3 tensor to use channels_last_1d format");
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
        meta.dims(),
        xpu::dpcpp::get_channels_last_strides_1d_dpcpp(meta.dims()));
  } else {
    tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  }

  auto ctx = (DPCPPTensorContext*)tensor.unsafeGetTensorImpl()
                 ->storage()
                 .unsafeGetStorageImpl()
                 ->data_ptr()
                 .get_context();
  ctx->set_meta(meta);
  ctx->set_aten_meta({tensor.sizes().vec(), tensor.strides().vec()});

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

Tensor to_plain(const Tensor& tensor) {
  return to_plain_if_needed(tensor);
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

std::vector<Tensor> to_plain_if_needed(MaterializedITensorListRef tensors) {
  std::vector<Tensor> _tensors;
  for (auto tensor : tensors) {
    _tensors.push_back(to_plain_if_needed(tensor.get()));
  }
  return _tensors;
}

} // namespace AtenIpexTypeXPU

namespace {
TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def("to_plain(Tensor tensor) -> Tensor");
  m.impl("to_plain", c10::DispatchKey::XPU, at::AtenIpexTypeXPU::to_plain);
}
} // namespace

namespace AtenIpexTypeQuantizedXPU {

using AtenIpexTypeXPU::DPCPPTensorContext;
using AtenIpexTypeXPU::DPCPPTensorConvertor;

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
