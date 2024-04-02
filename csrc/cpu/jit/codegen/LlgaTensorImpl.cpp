#include "LlgaTensorImpl.h"
#include "codegen/onednn/runtime.h"

#include <c10/core/CPUAllocator.h>

namespace torch_ipex {
namespace jit {
namespace fuser {
namespace onednn {

LlgaTensorImpl::LlgaTensorImpl(
    c10::Storage&& storage,
    const caffe2::TypeMeta& data_type,
    const LlgaTensorDesc& desc)
    : TensorImpl(
          std::move(storage),
          c10::DispatchKeySet(c10::DispatchKey::MkldnnCPU),
          data_type),
      desc_(desc) {
  set_sizes_and_strides(desc.sizes(), desc.strides());
  refresh_numel();
}

bool LlgaTensorImpl::has_storage() const {
  return true;
}

at::Tensor empty_llga(
    const LlgaTensorDesc& desc,
    const at::TensorOptions& options) {
  auto sizes = desc.sizes();
  auto nbytes = desc.storage_size();

  auto allocator = at::GetCPUAllocator();
  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      nbytes,
      allocator->allocate(nbytes),
      allocator,
      /*resizable=*/false);

  return at::detail::make_tensor<LlgaTensorImpl>(
      std::move(storage_impl), options.dtype(), desc);
}

const LlgaTensorDesc& get_llga_desc(const at::Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(
      tensor.is_mkldnn(), "get_llga_desc expects Mkldnn tensor input");
  return static_cast<LlgaTensorImpl*>(tensor.unsafeGetTensorImpl())->desc();
}

dnnl::graph::tensor llga_from_aten_tensor(const at::Tensor& tensor) {
  return {
      get_llga_desc(tensor).logical_tensor(),
      Engine::getEngine(),
      tensor.data_ptr()};
}

using data_type = dnnl::graph::logical_tensor::data_type;
data_type getLlgaDataType(at::ScalarType dt) {
  switch (dt) {
    case at::ScalarType::Float:
      return data_type::f32;
    case at::ScalarType::BFloat16:
      return data_type::bf16;
    case at::ScalarType::Bool:
      return data_type::boolean;
    case at::kInt:
      return data_type::s32;
    case at::ScalarType::QInt8:
      return data_type::s8;
    case at::ScalarType::QUInt8:
      return data_type::u8;
    default:
      return data_type::undef;
  }
}

at::Tensor LlgaTensorImpl::llga_to_aten_tensor(LlgaTensorImpl* llgaImpl) {
  auto aten_tensor = at::detail::make_tensor<TensorImpl>(
      std::move(llgaImpl->storage_),
      c10::DispatchKeySet(c10::DispatchKey::CPU),
      llgaImpl->data_type_);
  auto impl = aten_tensor.unsafeGetTensorImpl();
  impl->set_storage_offset(llgaImpl->storage_offset_);
  impl->set_sizes_and_strides(llgaImpl->sizes(), llgaImpl->strides());
  return aten_tensor;
}

at::Tensor LlgaTensorImpl::llga_to_aten_tensor(
    LlgaTensorImpl* llgaImpl,
    at::QuantizerPtr quantizer) {
  auto aten_tensor = at::detail::make_tensor<at::QTensorImpl>(
      std::move(llgaImpl->storage_),
      c10::DispatchKeySet(c10::DispatchKey::QuantizedCPU),
      llgaImpl->data_type_,
      quantizer);
  auto impl = aten_tensor.unsafeGetTensorImpl();
  impl->set_storage_offset(llgaImpl->storage_offset_);
  impl->set_sizes_and_strides(llgaImpl->sizes(), llgaImpl->strides());
  return aten_tensor;
}

LlgaTensorDesc LlgaTensorDesc::supplementTensorInfo(const at::Tensor& t) const {
  if (t.is_mkldnn()) {
    // if input tensor is of mkldnn, it's originated from an upstream
    // LLGA partition which carries opaque layout info
    return get_llga_desc(t).tid(tid_);
  } else {
    // if input tensor is not an mkldnn tensor, use default layout
    auto sizes = t.sizes().vec();
    auto strides = t.strides().vec();
    auto dtype = getLlgaDataType(t.scalar_type());
    TORCH_CHECK(
        dtype != data_type::undef, "Not support data type ", t.scalar_type());
    return {tid_, sizes, strides, dtype, property_type_, is_scalar_tensor_};
  }
}

LlgaTensorDesc LlgaTensorDesc::convertDimsToUnknown() {
  if (!is_dimensionality_unknown() && !is_opaque()) {
    for (int i = 0; i < sizes_.size(); i++) {
      sizes_[i] = INT64_MIN;
      strides_[i] = INT64_MIN;
    }
  }
  return {tid_, sizes_, strides_, dtype_, property_type_, is_scalar_tensor_};
}

at::ScalarType LlgaTensorDesc::aten_scalar_type() const {
  switch (dtype_) {
    case data_type::f32:
      return at::ScalarType::Float;
    case data_type::bf16:
      return at::ScalarType::BFloat16;
    case data_type::boolean:
      return at::ScalarType::Bool;
    case data_type::s32:
      return at::kInt;
    case data_type::s8:
      return at::ScalarType::QInt8;
    case data_type::u8:
      return at::ScalarType::QUInt8;
    default:
      TORCH_CHECK(false, "Invalid data type ", static_cast<size_t>(dtype_));
  }
}
} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch_ipex
