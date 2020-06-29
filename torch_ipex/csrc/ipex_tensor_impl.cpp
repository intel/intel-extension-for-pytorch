#include "torch_ipex/csrc/ipex_tensor_impl.h"

#include <c10/core/ScalarType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

#include "utils.h"

namespace torch_ipex {

namespace {

thread_local c10::Device g_current_device(at::DeviceType::DPCPP, 0);

struct IPEXGuardImpl : public c10::impl::DeviceGuardImplInterface {
  at::DeviceType type() const override { return at::DeviceType::DPCPP; }

  c10::Device exchangeDevice(c10::Device device) const override {
    std::swap(g_current_device, device);
    return device;
  }

  c10::Device getDevice() const override { return g_current_device; }

  void setDevice(c10::Device device) const override {
    g_current_device = device;
  }

  void uncheckedSetDevice(c10::Device device) const noexcept override {
    g_current_device = device;
  }

  c10::Stream getStream(c10::Device device) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, device);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, g_current_device);
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    c10::DeviceIndex dev_count = -1;
    get_device_count(g_current_device, &dev_count);
    return dev_count;
  }
};

C10_REGISTER_GUARD_IMPL(DPCPP, IPEXGuardImpl);

}  // namespace


IPEXTensorImpl::IPEXTensorImpl(at::Tensor tensor, at::Storage storage, at::DispatchKey type_id) :
    c10::TensorImpl(std::move(storage), type_id),
    m_src_data_tensor(std::move(tensor)) {}

IPEXTensorImpl::IPEXTensorImpl(at::Storage storage, at::DispatchKey type_id) :
    c10::TensorImpl(std::move(storage), type_id) {}

IPEXTensorImpl::IPEXTensorImpl(at::DispatchKeySet type_set, const caffe2::TypeMeta& data_type, c10::optional<c10::Device> device_opt) :
    c10::TensorImpl(type_set, data_type, device_opt) {}

void IPEXTensorImpl::set_dpcpp_tensor_id() {
  this->key_set_.add(at::DispatchKey::VariableTensorId);
}

void IPEXTensorImpl::reset_data_type(at::ScalarType dst_type) {
  this->data_type_ = at::scalarTypeToTypeMeta(dst_type);
}

void IPEXTensorImpl::copy_auto_grad(c10::TensorImpl *src_impl) {
  if (! src_impl->requires_grad()) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(! this->requires_grad());
    return;
  }

  if (! this->requires_grad())
    this->set_requires_grad(true);

  this->grad() = src_impl->grad();
}

void IPEXTensorImpl::copy_meta_info(const c10::TensorImpl *src_impl) {
  // Port from copy_tensor_metadata of TensorImpl.cpp and bypass some fields: storage_, device_opt_, type_set_ and reserved_.
  // NOTE: All these fields is specifically ignored except reserved_. Because there is no public interface to access it. Tthe
  //       field may impact performance. Tensor resize will check the flag. "If tensor is reserved then don't claim its memeory
  //       unless capacity() is smaller than new size"
  /*
  dest_impl->storage_ = src_impl->storage_;
  dest_impl->device_opt_ = src_impl->device_opt_;
  dest_impl->reserved_ = src_impl->reserved_;
  dest_impl->type_set_ = src_impl->type_set();
  */
  this->sizes_ = src_impl->sizes();
  this->strides_ = src_impl->strides();
  this->storage_offset_ = src_impl->storage_offset();
  this->data_type_ = src_impl->dtype();
  this->is_contiguous_ = src_impl->is_contiguous();
  this->is_channels_last_contiguous_ = src_impl->is_contiguous(at::MemoryFormat::ChannelsLast);
  this->is_channels_last_ = src_impl->is_strides_like_channels_last();
  this->is_channels_last_3d_ = src_impl->is_strides_like_channels_last_3d();
  this->is_channels_last_3d_contiguous_ = src_impl->is_contiguous(at::MemoryFormat::ChannelsLast3d);
  this->is_non_overlapping_and_dense_ = src_impl->is_non_overlapping_and_dense();
  this->is_wrapped_number_ = src_impl->is_wrapped_number();
  this->set_version_counter(src_impl->version_counter().current_version());
  bool allow_tensor_metadata_change_ = src_impl->allow_tensor_metadata_change();
  this->set_allow_tensor_metadata_change(allow_tensor_metadata_change_);
  if (src_impl->named_tensor_meta() != nullptr) {
    this->set_named_tensor_meta(src_impl->named_tensor_meta()->clone());
  }
  this->refresh_numel();
}

static inline void checkInBoundsForStorage(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    int64_t storage_offset,
    const at::Storage& new_storage) {
  // Port from setStrided in Aten/native/Resize.h
  int64_t storage_size = at::detail::computeStorageSize(size, stride);
  if (storage_size == 0) {
    // NB: (a tensor with arbitrary 0 dims)'s storage can have any numel.
    return;
  }
  int64_t new_storage_size = new_storage.numel();
  TORCH_CHECK(
      storage_offset + storage_size <= new_storage_size,
      "setStorage: sizes ", size, ", strides ", stride, ","
      " and storage offset ", storage_offset,
      " requiring a storage size of ", storage_size + storage_offset,
      " are out of bounds for storage with numel ", new_storage_size);
}

void IPEXTensorImpl::set_strided(at::IntArrayRef size, at::IntArrayRef stride, int64_t storage_offset) {
  // Port from setStrided in Aten/native/Resize.h
  checkInBoundsForStorage(size, stride, storage_offset_, this->storage());

  // In backprop phase, grad variable might be detached (in accumulate_grad.h)
  // and forbids the metadata to be modified.
  // Here we force the metadata changeable, so that we can modify its strides.
  bool orig_allow_tensor_metadata_change = this->allow_tensor_metadata_change();
  this->set_allow_tensor_metadata_change(true);

  /* storage offset */
  TORCH_CHECK(storage_offset >= 0, "Tensor: invalid storage offset ", storage_offset);
  this->set_storage_offset(storage_offset);

  /* size and stride */
  AT_ASSERT(size.size() == stride.size());
  if (this->sizes() == size && this->strides() == stride) {
    return;
  }
  this->set_sizes_and_strides(size, stride);

  // Restore allow_tensor_metadata_change
  this->set_allow_tensor_metadata_change(orig_allow_tensor_metadata_change);
}

// Keep data tensor in case the IPEXTensorImpl is shallow-copy from a stack variable.
void IPEXTensorImpl::keep_source_data_tensor(at::Tensor data_tensor) {
  this->m_src_data_tensor = data_tensor;
}

// The storage of tensor impl cannot be accessed out of TensorImpl class. So we need to expose an interface
// to set storage data ptr.
void IPEXTensorImpl::set_storage_data_ptr(c10::DataPtr data_ptr) {
  this->storage_.set_data_ptr(std::move(data_ptr));
}

c10::Device IPEXTensorImpl::GetCurrentAtenDevice() {
  return g_current_device;
}

c10::Device IPEXTensorImpl::SetCurrentAtenDevice(c10::Device device) {
  std::swap(g_current_device, device);
  return device;
}

void IPEXTensorImpl::CopySizeStridesAndOffset(c10::TensorImpl *dest_impl, const c10::TensorImpl *src_impl) {
  dest_impl->set_sizes_and_strides(src_impl->sizes(), src_impl->strides());
  dest_impl->set_storage_offset(src_impl->storage_offset());
}

void IPEXTensorImpl::CopyMetadata(c10::TensorImpl *dest_impl, const c10::TensorImpl *src_impl) {
  if (dest_impl->dim() == 0) {
    dest_impl->set_wrapped_number(src_impl->is_wrapped_number());
  }

  dest_impl->set_version_counter(src_impl->version_counter().current_version());

  bool allow_tensor_metadata_change_ = src_impl->allow_tensor_metadata_change();
  dest_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change_);

  if (src_impl->named_tensor_meta() != nullptr) {
    dest_impl->set_named_tensor_meta(src_impl->named_tensor_meta()->clone());
  }
}

}  // namespace torch_ipex
